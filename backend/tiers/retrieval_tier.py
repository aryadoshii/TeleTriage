"""
Retrieval-Augmented Generation (RAG) Tier — Phase 2 implementation.

Tier 2 retrieves the top-scoring chunks from the corpus (hybrid BM25 +
dense search, fused with RRF, reranked with a cross-encoder) and then
SYNTHESIZES an answer from them via an LLM constrained to that retrieved
context — genuine RAG, not extractive lookup (an earlier design returned
a matched KB entry's `answer` field verbatim; superseded).

  scripts/synth_qa.py + scripts/build_synth_kb.py still exist as a
  documented alternative corpus path (database/synth_kb.jsonl) but are
  off the critical path — tier 2 runs directly over
  database/real_kb.jsonl's raw 3GPP chunks, no synthetic Q&A needed.

Architecture:

  ┌─────────────────────────────────────────────────────────────────┐
  │                         Query                                   │
  └─────────┬──────────────────────┬────────────────────────────────┘
            │                      │
     ┌──────▼──────┐        ┌──────▼──────┐
     │    BM25     │        │    Dense    │
     │   (sparse)  │        │ (BGE-small) │
     │  top_k=20   │        │  top_k=20   │
     └──────┬──────┘        └──────┬──────┘
            │                      │
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │  Reciprocal Rank     │   Σ 1/(k + rankᵢ) — no score
            │  Fusion (RRF)        │   normalisation needed.
            │  top_k=10            │   k=60 (Cormack et al. 2009)
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │  Cross-encoder       │   Joint (query, doc) encoding.
            │  reranker            │   Sigmoid(logit) → confidence.
            │  (bge-reranker-base) │
            │  top_k=max(3, K)     │   K = synthesis_top_k (default 5).
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐   Gate on reranker logit BEFORE
            │  Confidence gate     │   spending an LLM call — delegate
            │  (min_rerank_score)  │   to tier 3 on a miss, no synthesis.
            └──────────┬───────────┘
                       ▼ (pass)
            ┌──────────────────────┐
            │  RAG synthesis       │   Groq llama-3.3-70b-versatile —
            │  (top-K chunks as    │   see synthesis_model config field),
            │   grounding context) │   reusing tier 3's FallbackClient.
            └──────────┬───────────┘
                       ▼
                   TierResult
                (answer + citations)

Startup note:
  __init__ loads the BM25 + FAISS indexes from disk (built by
  scripts/build_index.py).  If the index files don't exist it raises
  FileNotFoundError — build the indexes first.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.config import get_config
from backend.generation.llm_client import (
    AllBackendsFailedError,
    FallbackClient,
    make_llm_client,
)
from backend.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    Embedder,
    Hit,
    reciprocal_rank_fusion,
)
from backend.tiers.base import BaseTier
from backend.types import Query, TierName, TierResult

# Max answer chars carried per candidate into TierResult.details["candidates"].
# Mirrors llm_client.MAX_CONTEXT_CHARS's spirit (bound token spend) but
# applied per-candidate here so 3 candidates can't blow the prompt budget
# on their own before build_user_prompt even gets to apply its own cap.
_CANDIDATE_ANSWER_CHARS = 800

# Generation params for tier 2's synthesis call — stable implementation
# details of the prompt itself, not deployer-tunable (unlike
# synthesis_top_k/synthesis_model). Same low temperature as tier 3: we
# want consistent, reproducible troubleshooting advice, not creative
# variation.
_SYNTHESIS_MAX_TOKENS = 512
_SYNTHESIS_TEMPERATURE = 0.2

# Same model as tier 3 (RetrievalTierConfig.synthesis_model), not a
# smaller one — a cheaper model's free-tier TPM budget throttled too
# often to be viable; see config.yaml's synthesis_model comment.
#
# Structure mirrors generative_tier.py's SYSTEM_PROMPT exactly (root
# cause / diagnostic steps / remediation) so tier 2 and tier 3 read like
# the same engineer answered. The forbidden-phrasing rule below exists
# because without it, the model's default instinct is to summarize
# excerpts one by one ("Excerpt 2 states that...") instead of
# synthesizing a diagnosis — telling it to just "answer using the
# excerpts" wasn't enough on its own to stop that.
SYNTHESIS_SYSTEM_PROMPT = """You are a telecom network engineer diagnosing a fault using ONLY the \
3GPP specification excerpts provided below.

Structure your answer EXACTLY as:
1. Most likely root cause (one sentence)
2. Diagnostic steps (numbered list, max 5 steps)
3. Remediation (numbered list, max 3 steps)

Rules:
- Answer using ONLY the provided excerpts — never fall back on outside \
knowledge or training data to fill gaps.
- Ground every claim in the excerpts and cite the source spec inline for \
each one, e.g. "(3GPP TS 36.300)".
- SYNTHESIZE across the excerpts into the structure above — never \
structure the answer by walking through them in order. Phrasing like \
"Excerpt 3 states..." or "Excerpt 2 mentions..." is forbidden; write as \
if you already know the material, citing the spec, not the excerpt \
number.
- Never restate the question as if it were a finding or a conclusion.
- If the excerpts only partially cover the question, answer the covered \
part in the structure above and state plainly what's missing — do not \
guess at the rest.
- If the excerpts contain nothing relevant to the question at all, say \
so explicitly instead of guessing.
- Be concise and technical. No preamble."""


def _build_candidates(reranked: list[Hit]) -> list[dict]:
    """
    Package up to 3 reranked candidates as grounding context for
    downstream tiers — carried into the next tier's Query.metadata by
    backend/router.py when retrieval's confidence gate isn't met.

    Populated on both of RetrievalTier.answer()'s return paths (not just
    delegation) since the runner-ups are also useful for eval tooling
    inspecting why the winner won, and cost nothing extra to include —
    the cross-encoder already scored every fused candidate before
    picking one.
    """
    return [
        {
            "doc_id": hit.doc_id,
            "question": hit.doc.get("question", ""),
            "answer": hit.doc.get("answer", "")[:_CANDIDATE_ANSWER_CHARS],
            "logit": hit.score,
        }
        for hit in reranked[:3]
    ]


def _build_synthesis_prompt(
    query_text: str, chunks: list[Hit], chunk_chars: int | None = None
) -> str:
    """
    Build the user-turn prompt for tier 2's synthesis call: the query
    plus every chunk passed to the synthesizer, each labelled with its
    source spec and chunk id so the model can cite them — and so a human
    reading the prompt in a debug log can trace every claim back to a
    specific chunk.

    Deliberately labelled "[spec, chunk id]:", NOT "Excerpt N [...]:" —
    numbering them made the model echo that framing back into its
    answers ("Excerpt 3 states..."); removing it (on top of
    SYNTHESIS_SYSTEM_PROMPT's explicit prohibition) removed the stimulus.

    chunk_chars: per-chunk truncation. None (default) means no
    truncation — chunks are already bounded to ~2048 chars at scrape
    time. See RetrievalTierConfig.synthesis_chunk_chars.
    """
    def _chunk_text(hit: Hit) -> str:
        text = hit.doc.get("answer", "")
        return text[:chunk_chars] if chunk_chars is not None else text

    excerpts = "\n\n".join(
        f"[{hit.doc.get('source', 'unknown spec')}, chunk {hit.doc_id}]:\n{_chunk_text(hit)}"
        for hit in chunks
    )
    return (
        f"Question: {query_text}\n\n"
        f"{excerpts}\n\n"
        f"Diagnose the fault using ONLY the excerpts above, in the "
        f"required root-cause / diagnostic-steps / remediation "
        f"structure. Cite the source spec for each claim, not the "
        f"excerpt's position in this list. If the excerpts don't fully "
        f"cover the question, answer what they do cover and state "
        f"plainly what's missing."
    )


def _build_citations(chunks: list[Hit]) -> list[dict]:
    """
    Provenance record of exactly which chunks were passed to the
    synthesizer, for TierResult.details["citations"] — what the CLI and
    dashboard use to show "this answer was built from these sources."

    Distinct from _build_candidates()'s "candidates" field: candidates is
    a general top-3 preview populated on every answer() path (including
    the delegation path, where no LLM was ever called); citations only
    exists when synthesis actually happened, and covers however many
    chunks synthesis_top_k configured — not capped at 3.
    """
    return [
        {
            "spec_id": hit.doc.get("spec_id", ""),
            "chunk_id": hit.doc_id,
            "source": hit.doc.get("source", ""),
        }
        for hit in chunks
    ]


def load_index_manifest(index_dir: Path) -> dict | None:
    """
    Load database/indexes/manifest.json if present.

    The manifest is written by scripts/build_index.py and records which KB
    file, doc count, and embedder the live indexes were actually built
    from — the ground truth for "what is retrieval actually searching,"
    independent of whatever config/config.yaml currently points at.

    Returns None for indexes built before this feature existed, or if the
    file is missing/corrupt — callers must treat that as "unknown," not
    an error, so older indexes keep working.

    Standalone module-level function (not a method) so callers that only
    want the manifest — `teletriage info`, the dashboard's mismatch check —
    can read it without paying for a full RetrievalTier (embedder + FAISS +
    cross-encoder model loads).
    """
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return None


class RetrievalTier(BaseTier):
    tier_name = TierName.RETRIEVAL

    def __init__(self) -> None:
        cfg = get_config()
        self.min_confidence = cfg.retrieval_tier.min_confidence
        self._cfg = cfg.retrieval_tier

        # Embedder is needed at both index-build time and query time.
        # We load it once here so it's shared between the DenseRetriever
        # (which needs it for query encoding) and any future components.
        self._embedder = Embedder(self._cfg.embedder_model)

        # Load pre-built indexes (see scripts/build_index.py).
        index_dir = cfg.resolve_path(cfg.paths.faiss_index_dir)
        self._bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
        self._dense = DenseRetriever.load(index_dir, self._embedder)

        # Cross-encoder is loaded fresh (no serialised state needed).
        self._reranker = CrossEncoderReranker(self._cfg.reranker_model)

        # Which KB these indexes were actually built from — None if the
        # indexes predate manifest.json. Never raises on absence.
        self.index_manifest = load_index_manifest(index_dir)

        # Same FallbackClient machinery as tier 3, but authenticated with
        # tier 2's OWN Groq key (cfg.secrets.groq_api_key) — tier 3 uses
        # groq_api_key_tier3, a separate TPM pool. See Secrets.
        # groq_api_key_tier3 in backend/config.py. No network calls
        # happen until generate() is actually invoked.
        self._synthesis_client: FallbackClient = make_llm_client(
            groq_api_key=cfg.secrets.groq_api_key,
            google_api_key=cfg.secrets.google_api_key,
            groq_model=self._cfg.synthesis_model,
            gemini_model=cfg.generative_tier.gemini_model,
            local_model=cfg.generative_tier.local_model,
            backend_order=self._cfg.backend_order,
            client_name="tier2-synthesis",
        )

    def debug_search(self, query_text: str, top_k: int = 10) -> list[Hit]:
        """
        Run the full BM25 -> dense -> RRF -> rerank pipeline and return the
        top_k reranked hits directly — bypassing answer()'s confidence
        gate entirely and its synthesis call.

        For tooling/observability only — scripts/label_relevance.py needs
        to show a human reviewer more candidates than answer() itself
        ever surfaces, to build the relevance judgements
        backend/evaluation/retrieval_metrics.py scores against. Never
        called from the production query path.
        """
        bm25_hits = self._bm25.search(query_text, k=self._cfg.bm25_top_k)
        dense_hits = self._dense.search(query_text, k=self._cfg.dense_top_k)
        fused = reciprocal_rank_fusion(
            [bm25_hits, dense_hits],
            k=self._cfg.rrf_k,
            top_k=self._cfg.rerank_top_k,
        )
        if not fused:
            return []
        return self._reranker.rerank(query_text, fused, top_k=top_k)

    def answer(self, query: Query) -> TierResult:
        start = self._now()

        # ── Step 1: Sparse retrieval (BM25) ───────────────────────────────
        # Fast exact-match / keyword-weighted scoring.  Handles rare telecom
        # acronyms (TEID, HARQ, SCTP) that dense models may conflate.
        bm25_hits = self._bm25.search(query.text, k=self._cfg.bm25_top_k)

        # ── Step 2: Dense retrieval (bi-encoder + FAISS) ──────────────────
        # Semantic similarity.  Handles paraphrases and vocabulary mismatches
        # ("voice call choppy" → matches "one-way audio in VoLTE").
        dense_hits = self._dense.search(query.text, k=self._cfg.dense_top_k)

        # ── Step 3: Reciprocal Rank Fusion ────────────────────────────────
        # Combines the two ranked lists without needing to normalise scores.
        # score = Σ 1/(60 + rank_i).  Returns up to rerank_top_k candidates.
        fused = reciprocal_rank_fusion(
            [bm25_hits, dense_hits],
            k=self._cfg.rrf_k,
            top_k=self._cfg.rerank_top_k,
        )

        if not fused:
            return self._empty_result(self._now() - start)

        # ── Step 4: Cross-encoder reranking ───────────────────────────────
        # Joint (query, doc) encoding for precise relevance scoring.
        # Returns raw logits; positive logit ≈ relevant.
        #
        # top_k = max(3, synthesis_top_k): one rerank call serves both
        # _build_candidates' fixed top-3 preview AND however many chunks
        # synthesis_top_k wants for the RAG prompt below — the cross-
        # encoder already scores every fused candidate before sorting, so
        # asking for more back than the historical minimum of 1 is free.
        rerank_top_k = max(3, self._cfg.synthesis_top_k)
        reranked = self._reranker.rerank(
            query.text,
            fused,
            top_k=rerank_top_k,
        )

        if not reranked:
            return self._empty_result(self._now() - start)

        best = reranked[0]
        confidence = self._reranker.confidence_from_logit(best.score)
        candidates = _build_candidates(reranked)

        # ── Step 5: Confidence gate — BEFORE calling the LLM ───────────────
        # Gate stays exactly where it was: on the reranker logit, not on
        # anything the synthesizer produces. Ordering matters — we must
        # not spend an LLM call synthesizing from retrieval we already
        # know is bad; delegate to tier 3 immediately instead.
        if best.score < self._cfg.min_rerank_score:
            return TierResult(
                tier=self.tier_name,
                answer=None,
                confidence=confidence,
                latency_sec=self._now() - start,
                details={
                    "reason": "low_rerank_score",
                    "best_doc_id": best.doc_id,
                    "rerank_logit": best.score,
                    "rerank_confidence": confidence,
                    "candidates": candidates,
                },
            )

        # ── Step 6: RAG synthesis ──────────────────────────────────────────
        # Gate passed — synthesize an answer from the top synthesis_top_k
        # reranked chunks, grounded ONLY in their text (see
        # SYNTHESIS_SYSTEM_PROMPT above). This replaces the old extractive
        # behaviour (returning best.doc["answer"] verbatim) — see this
        # module's docstring for why.
        synthesis_chunks = reranked[: self._cfg.synthesis_top_k]
        user_prompt = _build_synthesis_prompt(
            query.text, synthesis_chunks, chunk_chars=self._cfg.synthesis_chunk_chars,
        )

        try:
            output = self._synthesis_client.generate(
                system=SYNTHESIS_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=_SYNTHESIS_MAX_TOKENS,
                temperature=_SYNTHESIS_TEMPERATURE,
            )
        except AllBackendsFailedError as exc:
            # Tier 2 isn't the last tier — unlike tier 3, it can safely
            # delegate rather than surface a hard error. Retrieval itself
            # succeeded; only synthesis failed, so let tier 3 take a fresh
            # shot (it may reuse these candidates as grounding context via
            # the router's retrieved_context wiring — see router.py).
            return TierResult(
                tier=self.tier_name,
                answer=None,
                confidence=confidence,
                latency_sec=self._now() - start,
                details={
                    "reason": "synthesis_failed",
                    "error": str(exc),
                    "best_doc_id": best.doc_id,
                    "rerank_logit": best.score,
                    "rerank_confidence": confidence,
                    "candidates": candidates,
                },
            )

        return TierResult(
            tier=self.tier_name,
            answer=output.text,
            confidence=confidence,
            latency_sec=self._now() - start,
            details={
                "doc_id": best.doc_id,
                "rerank_logit": best.score,
                "rerank_confidence": confidence,
                "bm25_candidates": len(bm25_hits),
                "dense_candidates": len(dense_hits),
                "fused_candidates": len(fused),
                "candidates": candidates,
                "citations": _build_citations(synthesis_chunks),
                "synthesis_backend": output.backend,
                "synthesis_model": output.model,
                "synthesis_tokens_used": output.tokens_used,
                # Non-zero means latency_sec above includes at least one
                # retry, not pure inference time (see GroqClient).
                "synthesis_retry_count": output.retry_count,
            },
        )

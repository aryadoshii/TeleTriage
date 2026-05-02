"""
Retrieval-Augmented Generation (RAG) Tier — Phase 2 implementation.

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
            │  final_top_k=1       │
            └──────────┬───────────┘
                       ▼
                   TierResult

Startup note:
  __init__ loads the BM25 + FAISS indexes from disk (built by
  scripts/build_index.py).  If the index files don't exist it raises
  FileNotFoundError — build the indexes first.
"""
from __future__ import annotations

from teletriage.config import get_config
from teletriage.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    Embedder,
    reciprocal_rank_fusion,
)
from teletriage.tiers.base import BaseTier
from teletriage.types import Query, TierName, TierResult


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
        reranked = self._reranker.rerank(
            query.text,
            fused,
            top_k=self._cfg.final_top_k,
        )

        if not reranked:
            return self._empty_result(self._now() - start)

        best = reranked[0]
        confidence = self._reranker.confidence_from_logit(best.score)

        # ── Step 5: Confidence gate ────────────────────────────────────────
        # If the reranker doesn't think the best hit is relevant, delegate
        # to the generative tier rather than returning a bad answer.
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
                },
            )

        return TierResult(
            tier=self.tier_name,
            answer=best.doc["answer"],
            confidence=confidence,
            latency_sec=self._now() - start,
            details={
                "doc_id": best.doc_id,
                "rerank_logit": best.score,
                "rerank_confidence": confidence,
                "bm25_candidates": len(bm25_hits),
                "dense_candidates": len(dense_hits),
                "fused_candidates": len(fused),
            },
        )

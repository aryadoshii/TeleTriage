"""
Tests for Phase 2: retrieval modules and RetrievalTier.

Test organisation:
  TestTokenizer      — _tokenize helper (pure function, no models)
  TestBM25Retriever  — build / search / save / load  (no ML models)
  TestRRF            — reciprocal_rank_fusion math    (pure function)
  TestSigmoid        — _sigmoid helper                (pure function)
  TestEmbedder       — encode shape + normalisation   (loads BGE-small)
  TestDenseRetriever — build / search / save / load   (loads BGE-small)
  TestReranker       — rerank ordering + logit        (loads bge-reranker)
  TestRetrievalTier  — end-to-end integration         (loads both models,
                        requires pre-built indexes)

Tests that load ML models are marked @pytest.mark.slow.
Tests that require pre-built indexes are marked @pytest.mark.integration
and will skip gracefully if the indexes haven't been built yet.

Run only fast tests:
    uv run pytest -m "not slow and not integration"

Run everything (after `uv run python scripts/build_index.py`):
    uv run pytest
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.retrieval._types import Hit
from backend.retrieval.bm25 import BM25Retriever, _tokenize
from backend.retrieval.hybrid import reciprocal_rank_fusion
from backend.retrieval.reranker import _sigmoid


# ─── Fixtures ────────────────────────────────────────────────────────────────


SAMPLE_DOCS = [
    {
        "id": "kb001",
        "question": "How do I diagnose intermittent packet loss on an LTE bearer?",
        "answer": "Capture at multiple points: UE-side tcpdump, eNB S1-U mirror.",
        "tags": ["packet-loss", "lte"],
        "source": "3GPP TS 36.300",
        "spec_id": "36.300",
    },
    {
        "id": "kb002",
        "question": "What causes RSRQ degradation in dense urban deployments?",
        "answer": "RSRQ = N * RSRP / RSSI. Interfering cells raise RSSI.",
        "tags": ["rsrq", "interference"],
        "source": "3GPP TS 36.214",
        "spec_id": "36.214",
    },
    {
        "id": "kb003",
        "question": "Why would a UE reattach repeatedly every few minutes?",
        "answer": "T3412 timer misconfigured or UE losing coverage.",
        "tags": ["attach", "tau"],
        "source": "3GPP TS 24.301",
        "spec_id": "24.301",
    },
    {
        "id": "kb004",
        "question": "How to troubleshoot one-way audio in VoLTE calls?",
        "answer": "Check NAT, SBC media anchoring, and RTP stream direction.",
        "tags": ["volte", "audio"],
        "source": "RFC 3550",
        "spec_id": "3550",
    },
    {
        "id": "kb005",
        "question": "Troubleshoot BGP flapping on core router.",
        "answer": "Check TCP 179 reachability, hold timer, MD5 auth key mismatch.",
        "tags": ["bgp", "routing"],
        "source": "RFC 4271",
        "spec_id": "4271",
    },
]


# ─── TestTokenizer ────────────────────────────────────────────────────────────


class TestTokenizer:
    """_tokenize is a critical function: same tokenizer must be used at
    index time and query time.  If these diverge, BM25 gets no token overlap."""

    def test_lowercases(self):
        assert _tokenize("LTE Bearer") == ["lte", "bearer"]

    def test_strips_punctuation(self):
        # Hyphens, slashes, periods all become spaces
        assert _tokenize("S1-U/S5") == ["s1", "u", "s5"]

    def test_collapses_whitespace(self):
        assert _tokenize("packet   loss") == ["packet", "loss"]

    def test_strips_leading_trailing(self):
        assert _tokenize("  RSRQ  ") == ["rsrq"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_alphanumeric_preserved(self):
        # 3GPP identifiers like "S1AP" should be a single token
        assert _tokenize("S1AP") == ["s1ap"]

    def test_underscore_preserved(self):
        # Underscore counts as a word character in \w
        tokens = _tokenize("bm25_score")
        assert "bm25_score" in tokens


# ─── TestBM25Retriever ────────────────────────────────────────────────────────


class TestBM25Retriever:
    """No ML models — all pure BM25 arithmetic on a tiny corpus."""

    def test_build_returns_correct_length(self):
        r = BM25Retriever.build(SAMPLE_DOCS)
        assert len(r) == len(SAMPLE_DOCS)

    def test_search_returns_hits(self):
        r = BM25Retriever.build(SAMPLE_DOCS)
        hits = r.search("packet loss LTE bearer", k=5)
        assert len(hits) > 0
        assert hits[0].doc_id == "kb001"  # exact vocabulary match

    def test_search_top_result_is_highest_score(self):
        r = BM25Retriever.build(SAMPLE_DOCS)
        hits = r.search("BGP flapping router", k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_k(self):
        r = BM25Retriever.build(SAMPLE_DOCS)
        hits = r.search("LTE bearer", k=2)
        assert len(hits) <= 2

    def test_search_zero_score_docs_excluded(self):
        r = BM25Retriever.build(SAMPLE_DOCS)
        # A query with no vocabulary overlap with any doc returns empty
        hits = r.search("xyzzy frobnicator", k=5)
        # All BM25 scores will be 0 — should return empty list
        assert all(h.score > 0 for h in hits)

    def test_search_rank_field_is_zero_indexed(self):
        r = BM25Retriever.build(SAMPLE_DOCS)
        hits = r.search("VoLTE audio RTP", k=3)
        for i, h in enumerate(hits):
            assert h.rank == i

    def test_hit_carries_doc(self):
        r = BM25Retriever.build(SAMPLE_DOCS)
        hits = r.search("packet loss LTE", k=1)
        assert hits[0].doc["id"] == hits[0].doc_id
        assert "answer" in hits[0].doc

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        r = BM25Retriever.build(SAMPLE_DOCS)
        r.save(tmp_path / "bm25.pkl")

        loaded = BM25Retriever.load(tmp_path / "bm25.pkl")
        assert len(loaded) == len(r)

        hits_original = r.search("packet loss LTE", k=3)
        hits_loaded = loaded.search("packet loss LTE", k=3)

        assert [h.doc_id for h in hits_original] == [h.doc_id for h in hits_loaded]
        assert [round(h.score, 6) for h in hits_original] == [
            round(h.score, 6) for h in hits_loaded
        ]

    def test_empty_corpus(self):
        r = BM25Retriever.build([])
        hits = r.search("packet loss", k=5)
        assert hits == []

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        r = BM25Retriever.build(SAMPLE_DOCS)
        deep = tmp_path / "a" / "b" / "c" / "bm25.pkl"
        r.save(deep)  # should not raise even though parents don't exist
        assert deep.exists()


# ─── TestRRF ──────────────────────────────────────────────────────────────────


def _make_hits(doc_ids: list[str]) -> list[Hit]:
    """Build a ranked list of Hit stubs from a list of doc IDs."""
    return [
        Hit(doc_id=d, score=float(10 - i), doc={"id": d}, rank=i)
        for i, d in enumerate(doc_ids)
    ]


class TestRRF:
    """Pure math — no models, no disk I/O."""

    def test_single_list_preserves_order(self):
        hits = _make_hits(["a", "b", "c"])
        fused = reciprocal_rank_fusion([hits])
        assert [h.doc_id for h in fused] == ["a", "b", "c"]

    def test_two_lists_consensus_boosts_shared_top(self):
        # Both lists agree on "a" being first → a should win
        list1 = _make_hits(["a", "b", "c"])
        list2 = _make_hits(["a", "c", "b"])
        fused = reciprocal_rank_fusion([list1, list2])
        assert fused[0].doc_id == "a"

    def test_rrf_score_formula(self):
        # Manually verify RRF score for a two-list scenario with k=60
        # doc "x" is rank 1 in list 1, rank 2 in list 2
        # expected score = 1/(60+1) + 1/(60+2)
        list1 = _make_hits(["x", "y"])
        list2 = _make_hits(["z", "x"])
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        x_hit = next(h for h in fused if h.doc_id == "x")
        expected = 1 / (60 + 1) + 1 / (60 + 2)
        assert abs(x_hit.score - expected) < 1e-10

    def test_doc_only_in_one_list(self):
        # "exclusive_a" appears only in list1, "exclusive_b" only in list2
        list1 = _make_hits(["exclusive_a"])
        list2 = _make_hits(["exclusive_b"])
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        ids = [h.doc_id for h in fused]
        assert "exclusive_a" in ids
        assert "exclusive_b" in ids

    def test_top_k_truncation(self):
        hits = _make_hits(["a", "b", "c", "d", "e"])
        fused = reciprocal_rank_fusion([hits], top_k=3)
        assert len(fused) == 3

    def test_fused_ranks_are_zero_indexed_and_sequential(self):
        list1 = _make_hits(["a", "b", "c"])
        list2 = _make_hits(["b", "a", "c"])
        fused = reciprocal_rank_fusion([list1, list2])
        for i, h in enumerate(fused):
            assert h.rank == i

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_k_affects_scores_not_ordering_for_uniform_input(self):
        # If both retrievers rank docs identically, changing k shouldn't
        # change the ordering (scores change but relative order is preserved)
        hits = _make_hits(["a", "b", "c"])
        fused_60 = reciprocal_rank_fusion([hits, hits], k=60)
        fused_10 = reciprocal_rank_fusion([hits, hits], k=10)
        assert [h.doc_id for h in fused_60] == [h.doc_id for h in fused_10]

    def test_doc_that_appears_in_both_lists_gets_higher_score(self):
        # "shared" appears in both; "exclusive" only in one
        list1 = _make_hits(["shared", "exclusive"])
        list2 = _make_hits(["shared"])
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        shared_hit = next(h for h in fused if h.doc_id == "shared")
        exclusive_hit = next(h for h in fused if h.doc_id == "exclusive")
        # shared: 1/61 + 1/61 = 0.0328
        # exclusive: 1/62 + 0 = 0.0161
        assert shared_hit.score > exclusive_hit.score


# ─── TestSigmoid ──────────────────────────────────────────────────────────────


class TestSigmoid:
    def test_zero_input_returns_half(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_approaches_zero(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        # σ(x) + σ(-x) = 1
        for x in [1.0, 2.5, 5.0, 10.0]:
            assert _sigmoid(x) + _sigmoid(-x) == pytest.approx(1.0, abs=1e-12)

    def test_known_value(self):
        # σ(1) = e/(1+e) ≈ 0.7311
        assert _sigmoid(1.0) == pytest.approx(1 / (1 + math.exp(-1)), rel=1e-9)

    def test_no_overflow_for_extreme_values(self):
        # These would overflow with naive 1/(1+exp(-x)) for x=-800
        assert 0.0 <= _sigmoid(-800.0) <= 1.0
        assert 0.0 <= _sigmoid(800.0) <= 1.0


# ─── TestEmbedder ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestEmbedder:
    """Loads the actual BGE-small model (~30 MB).  Skips if unavailable."""

    @pytest.fixture(scope="class")
    def embedder(self):
        from backend.retrieval.embedder import Embedder
        return Embedder("BAAI/bge-small-en-v1.5")

    def test_dim_is_set(self, embedder):
        assert embedder.dim == 384  # BGE-small output dim

    def test_encode_corpus_shape(self, embedder):
        vecs = embedder.encode_corpus(["hello world", "LTE bearer"])
        assert vecs.shape == (2, 384)
        assert vecs.dtype == np.float32

    def test_corpus_vectors_are_unit_norm(self, embedder):
        vecs = embedder.encode_corpus(["packet loss", "RSRQ degradation"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_query_shape(self, embedder):
        vec = embedder.encode_query("intermittent packet loss LTE")
        assert vec.shape == (1, 384)
        assert vec.dtype == np.float32

    def test_query_vector_is_unit_norm(self, embedder):
        vec = embedder.encode_query("BGP flapping")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_similar_queries_produce_close_vectors(self, embedder):
        # "packet loss" and "packets dropping" should be more similar to each
        # other than to "BGP routing"
        v1 = embedder.encode_query("packet loss")
        v2 = embedder.encode_query("packets dropping")
        v3 = embedder.encode_query("BGP routing table")
        sim12 = float(v1 @ v2.T)
        sim13 = float(v1 @ v3.T)
        assert sim12 > sim13


# ─── TestDenseRetriever ───────────────────────────────────────────────────────


@pytest.mark.slow
class TestDenseRetriever:
    """Loads BGE-small to build a tiny in-memory FAISS index."""

    @pytest.fixture(scope="class")
    def dense(self):
        from backend.retrieval.dense import DenseRetriever
        from backend.retrieval.embedder import Embedder
        embedder = Embedder("BAAI/bge-small-en-v1.5")
        return DenseRetriever.build(SAMPLE_DOCS, embedder)

    def test_len_matches_corpus(self, dense):
        assert len(dense) == len(SAMPLE_DOCS)

    def test_search_returns_hits(self, dense):
        hits = dense.search("intermittent packet loss LTE", k=5)
        assert len(hits) > 0

    def test_top_result_is_most_relevant(self, dense):
        hits = dense.search("intermittent packet loss on LTE bearer", k=5)
        assert hits[0].doc_id == "kb001"

    def test_scores_descending(self, dense):
        hits = dense.search("VoLTE audio one-way", k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_cosine_similarities(self, dense):
        # After L2 normalisation, inner product is in [-1, 1]
        hits = dense.search("LTE packet loss", k=5)
        for h in hits:
            assert -1.0 <= h.score <= 1.0 + 1e-5

    def test_save_and_load_roundtrip(self, dense, tmp_path: Path):
        from backend.retrieval.dense import DenseRetriever
        from backend.retrieval.embedder import Embedder
        dense.save(tmp_path)
        embedder = Embedder("BAAI/bge-small-en-v1.5")
        loaded = DenseRetriever.load(tmp_path, embedder)
        assert len(loaded) == len(dense)
        hits_orig = dense.search("BGP flapping", k=3)
        hits_load = loaded.search("BGP flapping", k=3)
        assert [h.doc_id for h in hits_orig] == [h.doc_id for h in hits_load]


# ─── TestReranker ─────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestReranker:
    """Loads bge-reranker-base.  Tests score ordering and confidence mapping."""

    @pytest.fixture(scope="class")
    def reranker(self):
        from backend.retrieval.reranker import CrossEncoderReranker
        return CrossEncoderReranker("BAAI/bge-reranker-base")

    def test_rerank_returns_same_count(self, reranker):
        hits = _make_hits_with_docs(SAMPLE_DOCS[:3])
        result = reranker.rerank("packet loss LTE", hits)
        assert len(result) == 3

    def test_rerank_top_k_truncates(self, reranker):
        hits = _make_hits_with_docs(SAMPLE_DOCS)
        result = reranker.rerank("packet loss LTE", hits, top_k=2)
        assert len(result) == 2

    def test_rerank_scores_descending(self, reranker):
        hits = _make_hits_with_docs(SAMPLE_DOCS)
        result = reranker.rerank("VoLTE one-way audio", hits)
        scores = [h.score for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_changes_order(self, reranker):
        # Feed hits in a shuffled order; reranker should move the most
        # relevant doc to the top regardless of input order.
        shuffled = [
            Hit(doc_id=d["id"], score=0.0, doc=d, rank=i)
            for i, d in enumerate(reversed(SAMPLE_DOCS))
        ]
        result = reranker.rerank("intermittent packet loss on LTE bearer", shuffled)
        assert result[0].doc_id == "kb001"

    def test_ranks_updated_after_reranking(self, reranker):
        hits = _make_hits_with_docs(SAMPLE_DOCS[:3])
        result = reranker.rerank("anything", hits)
        for i, h in enumerate(result):
            assert h.rank == i

    def test_empty_hits_returns_empty(self, reranker):
        assert reranker.rerank("query", []) == []

    def test_confidence_from_logit_range(self, reranker):
        hits = _make_hits_with_docs(SAMPLE_DOCS[:2])
        result = reranker.rerank("packet loss", hits)
        for h in result:
            conf = reranker.confidence_from_logit(h.score)
            assert 0.0 <= conf <= 1.0


def _make_hits_with_docs(docs: list[dict]) -> list[Hit]:
    return [Hit(doc_id=d["id"], score=0.0, doc=d, rank=i) for i, d in enumerate(docs)]


# ─── TestBuildCandidates ──────────────────────────────────────────────────────
# Pure function, hand-built Hit objects — no model loading, no @slow marker.
# This is what feeds TierResult.details["candidates"], which the router
# forwards to the generative tier as grounding context (see
# backend/generation/llm_client.py design decision (1)).


class TestBuildCandidates:
    def test_returns_at_most_3(self):
        from backend.tiers.retrieval_tier import _build_candidates

        hits = [
            Hit(doc_id=d["id"], score=float(5 - i), doc=d, rank=i)
            for i, d in enumerate(SAMPLE_DOCS)  # 5 docs available
        ]
        result = _build_candidates(hits)
        assert len(result) == 3

    def test_fewer_than_3_hits_returns_all(self):
        from backend.tiers.retrieval_tier import _build_candidates

        hits = _make_hits_with_docs(SAMPLE_DOCS[:2])
        result = _build_candidates(hits)
        assert len(result) == 2

    def test_empty_hits_returns_empty(self):
        from backend.tiers.retrieval_tier import _build_candidates

        assert _build_candidates([]) == []

    def test_dict_shape_has_exactly_the_required_keys(self):
        from backend.tiers.retrieval_tier import _build_candidates

        hits = [Hit(doc_id="kb001", score=1.5, doc=SAMPLE_DOCS[0], rank=0)]
        result = _build_candidates(hits)
        assert result[0].keys() == {"doc_id", "question", "answer", "logit"}

    def test_fields_map_from_the_hit_correctly(self):
        from backend.tiers.retrieval_tier import _build_candidates

        hits = [Hit(doc_id="kb001", score=3.7, doc=SAMPLE_DOCS[0], rank=0)]
        result = _build_candidates(hits)[0]
        assert result["doc_id"] == "kb001"
        assert result["question"] == SAMPLE_DOCS[0]["question"]
        assert result["answer"] == SAMPLE_DOCS[0]["answer"]
        assert result["logit"] == 3.7

    def test_preserves_rerank_order(self):
        """reranked is assumed pre-sorted by the caller (CrossEncoderReranker
        always returns descending-score order) — _build_candidates must not
        re-sort or shuffle it."""
        from backend.tiers.retrieval_tier import _build_candidates

        hits = [
            Hit(doc_id="kb003", score=1.0, doc=SAMPLE_DOCS[2], rank=0),
            Hit(doc_id="kb001", score=0.5, doc=SAMPLE_DOCS[0], rank=1),
            Hit(doc_id="kb004", score=0.1, doc=SAMPLE_DOCS[3], rank=2),
        ]
        result = _build_candidates(hits)
        assert [c["doc_id"] for c in result] == ["kb003", "kb001", "kb004"]

    def test_answer_truncated_to_800_chars(self):
        from backend.tiers.retrieval_tier import _build_candidates

        long_doc = {**SAMPLE_DOCS[0], "answer": "A" * 5000}
        hits = [Hit(doc_id="kb001", score=1.0, doc=long_doc, rank=0)]
        result = _build_candidates(hits)
        assert len(result[0]["answer"]) == 800

    def test_short_answer_not_padded_or_altered(self):
        from backend.tiers.retrieval_tier import _build_candidates

        hits = [Hit(doc_id="kb001", score=1.0, doc=SAMPLE_DOCS[0], rank=0)]
        result = _build_candidates(hits)
        assert result[0]["answer"] == SAMPLE_DOCS[0]["answer"]

    def test_missing_question_or_answer_key_does_not_crash(self):
        from backend.tiers.retrieval_tier import _build_candidates

        sparse_doc = {"id": "kb999"}  # no question/answer keys at all
        hits = [Hit(doc_id="kb999", score=1.0, doc=sparse_doc, rank=0)]
        result = _build_candidates(hits)
        assert result[0]["question"] == ""
        assert result[0]["answer"] == ""


# ─── TestBuildSynthesisPrompt ─────────────────────────────────────────────────
# Pure function, hand-built Hit objects — no model loading, no @slow marker.
# This is the RAG prompt tier 2 actually sends to the synthesis LLM.


class TestBuildSynthesisPrompt:
    def test_includes_query_text(self):
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        hits = _make_hits_with_docs(SAMPLE_DOCS[:1])
        prompt = _build_synthesis_prompt("intermittent packet loss on LTE", hits)
        assert "intermittent packet loss on LTE" in prompt

    def test_includes_chunk_text(self):
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        hits = _make_hits_with_docs(SAMPLE_DOCS[:1])
        prompt = _build_synthesis_prompt("query", hits)
        assert SAMPLE_DOCS[0]["answer"] in prompt

    def test_includes_source_spec_label(self):
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        hits = _make_hits_with_docs(SAMPLE_DOCS[:1])
        prompt = _build_synthesis_prompt("query", hits)
        assert SAMPLE_DOCS[0]["source"] in prompt

    def test_includes_chunk_id(self):
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        hits = _make_hits_with_docs(SAMPLE_DOCS[:1])
        prompt = _build_synthesis_prompt("query", hits)
        assert "kb001" in prompt

    def test_multiple_chunks_all_included(self):
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        hits = _make_hits_with_docs(SAMPLE_DOCS[:3])
        prompt = _build_synthesis_prompt("query", hits)
        for doc in SAMPLE_DOCS[:3]:
            assert doc["answer"] in prompt
            assert doc["id"] in prompt

    def test_instructs_context_only_answering(self):
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        hits = _make_hits_with_docs(SAMPLE_DOCS[:1])
        prompt = _build_synthesis_prompt("query", hits)
        assert "ONLY" in prompt

    def test_instructs_partial_coverage_handling(self):
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        hits = _make_hits_with_docs(SAMPLE_DOCS[:1])
        prompt = _build_synthesis_prompt("query", hits)
        assert "missing" in prompt.lower()

    def test_missing_source_key_does_not_crash(self):
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        sparse_doc = {"id": "kb999", "answer": "some text"}
        hits = [Hit(doc_id="kb999", score=1.0, doc=sparse_doc, rank=0)]
        prompt = _build_synthesis_prompt("query", hits)
        assert "some text" in prompt

    def test_does_not_use_excerpt_numbering(self):
        """Regression test: an earlier version labelled each chunk
        "Excerpt N [...]:", and the model echoed that framing straight
        into its answers ("Excerpt 3 states...", walking through excerpts
        in order instead of synthesizing). Chunks are now labelled
        "[spec, chunk id]:" with no ordinal — removing the stimulus that
        invited the bad phrasing, on top of the system prompt's explicit
        prohibition."""
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        hits = _make_hits_with_docs(SAMPLE_DOCS)  # all 5
        prompt = _build_synthesis_prompt("query", hits)
        assert "Excerpt 1" not in prompt
        assert "Excerpt 2" not in prompt
        assert "excerpt 1" not in prompt.lower()

    def test_chunk_chars_none_means_no_truncation(self):
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        hits = _make_hits_with_docs(SAMPLE_DOCS[:1])
        prompt = _build_synthesis_prompt("query", hits, chunk_chars=None)
        assert SAMPLE_DOCS[0]["answer"] in prompt

    def test_chunk_chars_truncates_each_chunk(self):
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        long_doc = {**SAMPLE_DOCS[0], "answer": "A" * 5000}
        hits = [Hit(doc_id="kb001", score=1.0, doc=long_doc, rank=0)]
        prompt = _build_synthesis_prompt("query", hits, chunk_chars=100)
        assert "A" * 100 in prompt
        assert "A" * 101 not in prompt

    def test_chunk_chars_truncates_independently_per_chunk(self):
        """Each chunk is capped at chunk_chars individually — a long
        chunk being truncated must not eat into a later chunk's budget
        (unlike llm_client.MAX_CONTEXT_CHARS, which is a shared total)."""
        from backend.tiers.retrieval_tier import _build_synthesis_prompt

        doc_a = {**SAMPLE_DOCS[0], "answer": "A" * 5000}
        doc_b = {**SAMPLE_DOCS[1], "answer": "B" * 5000}
        hits = [
            Hit(doc_id="kb001", score=1.0, doc=doc_a, rank=0),
            Hit(doc_id="kb002", score=0.5, doc=doc_b, rank=1),
        ]
        prompt = _build_synthesis_prompt("query", hits, chunk_chars=100)
        assert "A" * 100 in prompt
        assert "B" * 100 in prompt


class TestSynthesisSystemPrompt:
    """The system prompt itself — checks the strict-RAG rules are present,
    since these are what actually constrain the model's behaviour."""

    def test_instructs_citing_source_spec(self):
        from backend.tiers.retrieval_tier import SYNTHESIS_SYSTEM_PROMPT

        assert "cite" in SYNTHESIS_SYSTEM_PROMPT.lower()

    def test_forbids_outside_knowledge(self):
        from backend.tiers.retrieval_tier import SYNTHESIS_SYSTEM_PROMPT

        assert "outside" in SYNTHESIS_SYSTEM_PROMPT.lower()

    def test_instructs_admitting_no_relevant_context(self):
        from backend.tiers.retrieval_tier import SYNTHESIS_SYSTEM_PROMPT

        assert "nothing relevant" in SYNTHESIS_SYSTEM_PROMPT.lower()

    def test_requires_root_cause_structure(self):
        from backend.tiers.retrieval_tier import SYNTHESIS_SYSTEM_PROMPT

        assert "root cause" in SYNTHESIS_SYSTEM_PROMPT.lower()

    def test_requires_diagnostic_steps_structure(self):
        from backend.tiers.retrieval_tier import SYNTHESIS_SYSTEM_PROMPT

        assert "diagnostic steps" in SYNTHESIS_SYSTEM_PROMPT.lower()
        assert "max 5" in SYNTHESIS_SYSTEM_PROMPT.lower()

    def test_requires_remediation_structure(self):
        from backend.tiers.retrieval_tier import SYNTHESIS_SYSTEM_PROMPT

        assert "remediation" in SYNTHESIS_SYSTEM_PROMPT.lower()
        assert "max 3" in SYNTHESIS_SYSTEM_PROMPT.lower()

    def test_forbids_excerpt_walking_phrasing(self):
        from backend.tiers.retrieval_tier import SYNTHESIS_SYSTEM_PROMPT

        assert "forbidden" in SYNTHESIS_SYSTEM_PROMPT.lower()
        assert "excerpt 3 states" in SYNTHESIS_SYSTEM_PROMPT.lower()

    def test_forbids_restating_question_as_finding(self):
        from backend.tiers.retrieval_tier import SYNTHESIS_SYSTEM_PROMPT

        assert "restate the question" in SYNTHESIS_SYSTEM_PROMPT.lower()

    def test_instructs_partial_coverage_handling(self):
        from backend.tiers.retrieval_tier import SYNTHESIS_SYSTEM_PROMPT

        assert "partially cover" in SYNTHESIS_SYSTEM_PROMPT.lower()


# ─── TestBuildCitations ────────────────────────────────────────────────────────
# Pure function, hand-built Hit objects — no model loading, no @slow marker.
# This is exactly what TierResult.details["citations"] is populated from.


class TestBuildCitations:
    def test_one_entry_per_chunk(self):
        from backend.tiers.retrieval_tier import _build_citations

        hits = _make_hits_with_docs(SAMPLE_DOCS[:3])
        result = _build_citations(hits)
        assert len(result) == 3

    def test_empty_chunks_returns_empty(self):
        from backend.tiers.retrieval_tier import _build_citations

        assert _build_citations([]) == []

    def test_dict_shape_has_exactly_the_required_keys(self):
        from backend.tiers.retrieval_tier import _build_citations

        hits = [Hit(doc_id="kb001", score=1.0, doc=SAMPLE_DOCS[0], rank=0)]
        result = _build_citations(hits)
        assert result[0].keys() == {"spec_id", "chunk_id", "source"}

    def test_fields_map_from_the_hit_correctly(self):
        from backend.tiers.retrieval_tier import _build_citations

        hits = [Hit(doc_id="kb001", score=1.0, doc=SAMPLE_DOCS[0], rank=0)]
        result = _build_citations(hits)[0]
        assert result["chunk_id"] == "kb001"
        assert result["spec_id"] == SAMPLE_DOCS[0]["spec_id"]
        assert result["source"] == SAMPLE_DOCS[0]["source"]

    def test_not_capped_at_3_unlike_build_candidates(self):
        """citations covers however many chunks synthesis actually used
        (synthesis_top_k, default 5) — unlike _build_candidates, which
        always caps at 3 regardless of input size."""
        from backend.tiers.retrieval_tier import _build_citations

        hits = [
            Hit(doc_id=d["id"], score=float(5 - i), doc=d, rank=i)
            for i, d in enumerate(SAMPLE_DOCS)  # all 5
        ]
        result = _build_citations(hits)
        assert len(result) == 5

    def test_preserves_order(self):
        from backend.tiers.retrieval_tier import _build_citations

        hits = [
            Hit(doc_id="kb003", score=1.0, doc=SAMPLE_DOCS[2], rank=0),
            Hit(doc_id="kb001", score=0.5, doc=SAMPLE_DOCS[0], rank=1),
        ]
        result = _build_citations(hits)
        assert [c["chunk_id"] for c in result] == ["kb003", "kb001"]

    def test_missing_spec_id_or_source_defaults_to_empty_string(self):
        from backend.tiers.retrieval_tier import _build_citations

        sparse_doc = {"id": "kb999"}  # no spec_id/source keys at all
        hits = [Hit(doc_id="kb999", score=1.0, doc=sparse_doc, rank=0)]
        result = _build_citations(hits)
        assert result[0]["spec_id"] == ""
        assert result[0]["source"] == ""


# ─── TestRetrievalTier (integration) ─────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.slow
class TestRetrievalTier:
    """
    End-to-end integration test for the RAG-synthesis rewrite of tier 2.

    Uses a REAL embedder + REAL cross-encoder reranker (hence @slow) over
    a tiny in-memory index built from SAMPLE_DOCS — deliberately NOT the
    globally-configured disk indexes (cfg.paths.faiss_index_dir), so this
    test is independent of whatever corpus config.yaml happens to point
    at (database/real_kb.jsonl, synth_kb.jsonl, or anything else).

    The synthesis LLM client is always mocked — this class makes no real
    API calls. tier.answer() therefore returns whatever canned
    GenerationOutput the test configured, not text derived from
    SAMPLE_DOCS, and the interesting assertions are about the CONTRACT:
    what got passed to the synthesizer, and what ends up in
    details["citations"] — not the literal answer text (which used to be
    doc["answer"] verbatim under the old extractive design; it no longer
    is — see this module's docstring for why).

    RetrievalTier.__init__() itself isn't called (it loads indexes from
    disk via global config, which we're deliberately not using here) —
    instances are built via __new__() + direct attribute assignment,
    matching the pattern already used for the components below.
    """

    @pytest.fixture(scope="class")
    def real_components(self):
        """Expensive, shared across the class: real embedder + reranker,
        real small BM25/dense index built from SAMPLE_DOCS."""
        from backend.retrieval.dense import DenseRetriever
        from backend.retrieval.embedder import Embedder
        from backend.retrieval.reranker import CrossEncoderReranker

        embedder = Embedder("BAAI/bge-small-en-v1.5")
        bm25 = BM25Retriever.build(SAMPLE_DOCS)
        dense = DenseRetriever.build(SAMPLE_DOCS, embedder)
        reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
        return embedder, bm25, dense, reranker

    def _make_tier(self, real_components, **cfg_overrides):
        from backend.config import RetrievalTierConfig
        from backend.generation.llm_client import GenerationOutput
        from backend.tiers.retrieval_tier import RetrievalTier

        embedder, bm25, dense, reranker = real_components
        t = RetrievalTier.__new__(RetrievalTier)
        t._cfg = RetrievalTierConfig(
            min_rerank_score=cfg_overrides.pop("min_rerank_score", 0.20),
            min_confidence=0.50,
            synthesis_top_k=cfg_overrides.pop("synthesis_top_k", 3),
            synthesis_model="llama-3.1-8b-instant",
            **cfg_overrides,
        )
        t.min_confidence = t._cfg.min_confidence
        t._embedder = embedder
        t._bm25 = bm25
        t._dense = dense
        t._reranker = reranker
        t.index_manifest = None
        t._synthesis_client = MagicMock()
        t._synthesis_client.generate.return_value = GenerationOutput(
            text="Synthesized answer grounded in the provided excerpts.",
            tokens_used=42,
            finish_reason="stop",
            model="llama-3.1-8b-instant",
            backend="groq",
        )
        return t

    @pytest.fixture
    def tier(self, real_components):
        """A tier with a default-configured mocked synthesis client,
        function-scoped so each test can freely override its mock
        without affecting others (the expensive real_components stay
        class-scoped and shared)."""
        return self._make_tier(real_components)

    def test_known_query_answers_and_calls_synthesizer(self, tier):
        from backend.types import Query

        result = tier.answer(Query(text="intermittent packet loss on LTE bearer"))
        assert result.answered, "Expected a hit for a query closely matching kb001"
        tier._synthesis_client.generate.assert_called_once()

    def test_answer_is_synthesized_not_doc_verbatim(self, tier):
        """The defining behaviour change: tier 2 no longer returns
        best.doc["answer"] verbatim — it returns whatever the synthesizer
        produced."""
        from backend.types import Query

        result = tier.answer(Query(text="intermittent packet loss on LTE bearer"))
        assert result.answer == "Synthesized answer grounded in the provided excerpts."
        assert result.answer != SAMPLE_DOCS[0]["answer"]

    def test_synthesizer_called_with_relevant_chunk_context(self, tier):
        from backend.types import Query

        tier.answer(Query(text="intermittent packet loss on LTE bearer"))
        call_kwargs = tier._synthesis_client.generate.call_args.kwargs
        user_prompt = call_kwargs["user"]
        # kb001 (the packet-loss doc) should be the top match and appear
        # in the prompt handed to the synthesizer, with its source spec.
        assert "kb001" in user_prompt
        assert "3GPP TS 36.300" in user_prompt
        assert "tcpdump" in user_prompt  # actual chunk text made it into the prompt

    def test_synthesis_system_prompt_used(self, tier):
        from backend.tiers.retrieval_tier import SYNTHESIS_SYSTEM_PROMPT
        from backend.types import Query

        tier.answer(Query(text="intermittent packet loss on LTE bearer"))
        call_kwargs = tier._synthesis_client.generate.call_args.kwargs
        assert call_kwargs["system"] == SYNTHESIS_SYSTEM_PROMPT

    def test_citations_populated_on_synthesis(self, tier):
        from backend.types import Query

        result = tier.answer(Query(text="intermittent packet loss on LTE bearer"))
        assert "citations" in result.details
        citations = result.details["citations"]
        assert len(citations) >= 1
        assert citations[0]["chunk_id"] == "kb001"
        assert citations[0]["spec_id"] == "36.300"
        assert citations[0]["source"] == "3GPP TS 36.300"

    def test_candidates_still_populated_alongside_citations(self, tier):
        from backend.types import Query

        result = tier.answer(Query(text="intermittent packet loss on LTE bearer"))
        assert "candidates" in result.details
        assert len(result.details["candidates"]) >= 1

    def test_result_has_valid_confidence(self, tier):
        from backend.types import Query

        result = tier.answer(Query(text="RSRQ degradation urban interference"))
        assert 0.0 <= result.confidence <= 1.0

    def test_latency_under_budget(self, tier):
        from backend.types import Query

        result = tier.answer(Query(text="BGP flapping on core router"))
        # Generous budget: real embedder + reranker + a MOCKED (near-
        # instant) synthesis call. A real LLM call would blow this, which
        # is exactly why it's mocked here.
        assert result.latency_sec < 3.0, (
            f"Retrieval took {result.latency_sec * 1000:.0f} ms — exceeds budget"
        )

    def test_low_confidence_gate_prevents_synthesis_call(self, real_components):
        """Deterministic version of 'unrelated query delegates': force
        the gate to fail with an unreachably high min_rerank_score,
        rather than hoping a specific query happens to score low with the
        real reranker. Directly verifies requirement: gate BEFORE LLM
        call, no synthesis spend on retrieval we already know is bad."""
        tier = self._make_tier(real_components, min_rerank_score=999.0)

        from backend.types import Query
        result = tier.answer(Query(text="intermittent packet loss on LTE bearer"))

        assert not result.answered
        assert result.details["reason"] == "low_rerank_score"
        tier._synthesis_client.generate.assert_not_called()

    def test_synthesis_failure_delegates_rather_than_hard_errors(self, tier):
        """Unlike tier 3 (the last tier, which must always answer), tier
        2 can safely delegate to tier 3 on a synthesis failure — retrieval
        itself still succeeded, only the LLM call failed."""
        from backend.generation.llm_client import AllBackendsFailedError
        from backend.types import Query

        tier._synthesis_client.generate.side_effect = AllBackendsFailedError("all backends down")

        result = tier.answer(Query(text="intermittent packet loss on LTE bearer"))

        assert not result.answered
        assert result.details["reason"] == "synthesis_failed"

    def test_details_populated(self, tier):
        from backend.types import Query

        result = tier.answer(Query(text="VoLTE one-way audio"))
        if result.answered:
            assert "doc_id" in result.details
            assert "rerank_logit" in result.details

    def test_should_delegate_contract(self, tier):
        from backend.types import Query, TierResult, TierName

        low_conf = TierResult(tier=TierName.RETRIEVAL, answer="x", confidence=0.30)
        assert tier.should_delegate(low_conf) is True

        high_conf = TierResult(tier=TierName.RETRIEVAL, answer="x", confidence=0.90)
        assert tier.should_delegate(high_conf) is False

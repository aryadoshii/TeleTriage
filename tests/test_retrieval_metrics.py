"""
Tests for backend/evaluation/retrieval_metrics.py.

Pure functions, hand-computed expected values, no model loading, no
router, no real KB — every case below is worked out by hand in the
comment above it.

Run:
    uv run pytest tests/test_retrieval_metrics.py -v
"""
from __future__ import annotations

import pytest

from backend.evaluation.retrieval_metrics import mrr, recall_at_k


# ─── TestRecallAtK ─────────────────────────────────────────────────────────────


class TestRecallAtK:
    def test_all_relevant_found_at_k(self):
        # top-3 = {a,b,c}; relevant = {a,b,c} -> 3/3
        assert recall_at_k(["a", "b", "c", "d"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_none_relevant_found(self):
        # top-3 = {a,b,c}; relevant = {x,y} -> 0/2
        assert recall_at_k(["a", "b", "c"], ["x", "y"], k=3) == 0.0

    def test_partial_overlap(self):
        # top-3 = {a,b,c}; relevant = {c,x} -> intersection {c} -> 1/2
        assert recall_at_k(["a", "b", "c", "d", "e"], ["c", "x"], k=3) == pytest.approx(0.5)

    def test_relevant_doc_beyond_k_not_counted(self):
        # top-5 = {a,b,c,d,e}; relevant = {a,f}; f is at position 6, beyond k=5
        # -> intersection {a} -> 1/2
        retrieved = ["a", "b", "c", "d", "e", "f"]
        assert recall_at_k(retrieved, ["a", "f"], k=5) == pytest.approx(0.5)

    def test_relevant_doc_included_when_k_covers_it(self):
        # same list, k=6 now covers f too -> intersection {a,f} -> 2/2
        retrieved = ["a", "b", "c", "d", "e", "f"]
        assert recall_at_k(retrieved, ["a", "f"], k=6) == pytest.approx(1.0)

    def test_recall_at_1_strict(self):
        # top-1 = {b}; relevant = {a,b} -> intersection {b} -> 1/2
        assert recall_at_k(["b", "a", "c"], ["a", "b"], k=1) == pytest.approx(0.5)

    def test_empty_retrieved_returns_zero(self):
        assert recall_at_k([], ["a"], k=5) == 0.0

    def test_empty_relevant_returns_zero(self):
        # 0/0 undefined -> defined as 0.0 (see docstring)
        assert recall_at_k(["a", "b"], [], k=5) == 0.0

    def test_k_zero_returns_zero(self):
        assert recall_at_k(["a", "b", "c"], ["a"], k=0) == 0.0

    def test_negative_k_returns_zero(self):
        # guards against Python's negative-slice footgun (retrieved[:-1]
        # would silently drop the last element instead of meaning "none")
        assert recall_at_k(["a", "b", "c"], ["a"], k=-1) == 0.0

    def test_duplicate_ids_in_retrieved_not_double_counted(self):
        # top-3 (as a set) = {a,b}; relevant = {a} -> 1/1
        assert recall_at_k(["a", "a", "b"], ["a"], k=3) == pytest.approx(1.0)

    def test_duplicate_ids_in_relevant_collapsed(self):
        # relevant treated as a set: {a} not [a,a] -> denominator is 1, not 2
        assert recall_at_k(["a", "b", "c"], ["a", "a"], k=3) == pytest.approx(1.0)

    def test_k_larger_than_retrieved_list_no_crash(self):
        assert recall_at_k(["a"], ["a", "b"], k=100) == pytest.approx(0.5)

    def test_returns_float(self):
        assert isinstance(recall_at_k(["a"], ["a"], k=1), float)

    def test_score_is_bounded(self):
        for retrieved, relevant, k in [
            (["a", "b"], ["c"], 2),
            (["a"], ["a"], 1),
            ([], [], 5),
        ]:
            score = recall_at_k(retrieved, relevant, k)
            assert 0.0 <= score <= 1.0


# ─── TestMRR ────────────────────────────────────────────────────────────────
# (per-query reciprocal rank — see module docstring for why the mean-
# across-queries "MRR" itself is the caller's responsibility)


class TestMRR:
    def test_relevant_doc_at_rank_1(self):
        assert mrr(["a", "b", "c"], ["a"]) == pytest.approx(1.0)

    def test_relevant_doc_at_rank_2(self):
        assert mrr(["a", "b", "c"], ["b"]) == pytest.approx(0.5)

    def test_relevant_doc_at_rank_4(self):
        assert mrr(["x", "y", "z", "a"], ["a"]) == pytest.approx(0.25)

    def test_no_relevant_doc_found_returns_zero(self):
        assert mrr(["a", "b", "c"], ["z"]) == 0.0

    def test_uses_first_relevant_hit_not_best_rank_among_several(self):
        # relevant = {a, c}; "a" (rank 1) is found before "c" (rank 3) is
        # ever considered -> reciprocal rank of the FIRST hit, 1/1, not
        # some other combination
        assert mrr(["a", "b", "c"], ["a", "c"]) == pytest.approx(1.0)

    def test_first_relevant_hit_when_earliest_relevant_is_not_first_overall(self):
        # relevant = {b, c}; "a" at rank 1 isn't relevant, "b" at rank 2 is
        # -> 1/2
        assert mrr(["a", "b", "c"], ["b", "c"]) == pytest.approx(0.5)

    def test_empty_retrieved_returns_zero(self):
        assert mrr([], ["a"]) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert mrr(["a", "b"], []) == 0.0

    def test_both_empty_returns_zero(self):
        assert mrr([], []) == 0.0

    def test_returns_float(self):
        assert isinstance(mrr(["a"], ["a"]), float)

    def test_score_is_bounded(self):
        for retrieved, relevant in [
            (["a", "b"], ["c"]),
            (["a"], ["a"]),
            ([], []),
        ]:
            score = mrr(retrieved, relevant)
            assert 0.0 <= score <= 1.0

    def test_reciprocal_rank_decreases_with_rank(self):
        # sanity: finding the relevant doc earlier must never score lower
        # than finding it later
        early = mrr(["a", "b", "c", "d"], ["a"])
        late = mrr(["a", "b", "c", "d"], ["d"])
        assert early > late

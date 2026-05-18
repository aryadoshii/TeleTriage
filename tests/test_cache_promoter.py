"""
Tests for CachePromoter (Phase 6 — adaptive cache).

Coverage:
  TestGateConfidence    — Gate 1 fires for low-confidence answers
  TestGateLength        — Gate 2 fires for short answers
  TestGateDedup         — Gate 3 fires for near-duplicate questions
  TestEvaluatePassAll   — all gates pass → promoted=True
  TestPromote           — promote() persists to file AND calls cache_tier.add
  TestPromoteFromFile   — batch flow with mixed pass/fail rows
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from backend.cache.promoter import CachePromoter, PromotionReport, PromotionResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

LONG_ANSWER = (
    "When the UE fails to attach to the LTE network, the engineer should first "
    "examine the EMM cause code in the ATTACH REJECT message. Common causes "
    "include IMSI unknown in HSS (cause 2), illegal UE (cause 3), and EPS "
    "services not allowed (cause 7). Cross-check the subscriber profile in the "
    "HSS and verify the PLMN configuration matches the UE's home network."
)  # >50 words

SHORT_ANSWER = "Check the logs."  # <50 words

QUESTION_A = "Why does the UE fail to attach to LTE?"
QUESTION_A_NEAR_DUP = "Why does UE fail to attach LTE?"   # very close edit distance
QUESTION_B = "How is carrier aggregation configured in eNB?"


@pytest.fixture
def additions_file(tmp_path: Path) -> Path:
    return tmp_path / "cache_additions.jsonl"


@pytest.fixture
def promoter(additions_file: Path) -> CachePromoter:
    return CachePromoter(
        cache_tier=None,
        min_confidence=0.60,
        min_words=50,
        near_dup_threshold=0.3,
        additions_file=additions_file,
    )


@pytest.fixture
def mock_cache() -> MagicMock:
    cache = MagicMock()
    cache._cache = {}
    return cache


# ─── Gate 1: confidence ───────────────────────────────────────────────────────


class TestGateConfidence:
    def test_rejects_below_threshold(self, promoter: CachePromoter):
        result = promoter.evaluate(QUESTION_A, LONG_ANSWER, confidence=0.59)
        assert not result.promoted
        assert result.gate_failed == "confidence"

    def test_rejects_zero_confidence(self, promoter: CachePromoter):
        result = promoter.evaluate(QUESTION_A, LONG_ANSWER, confidence=0.0)
        assert not result.promoted
        assert result.gate_failed == "confidence"

    def test_passes_at_threshold(self, promoter: CachePromoter):
        result = promoter.evaluate(QUESTION_A, LONG_ANSWER, confidence=0.60)
        # may still fail length or dedup — just not confidence
        assert result.gate_failed != "confidence"

    def test_passes_above_threshold(self, promoter: CachePromoter):
        result = promoter.evaluate(QUESTION_A, LONG_ANSWER, confidence=0.90)
        assert result.gate_failed != "confidence"


# ─── Gate 2: length ───────────────────────────────────────────────────────────


class TestGateLength:
    def test_rejects_short_answer(self, promoter: CachePromoter):
        result = promoter.evaluate(QUESTION_A, SHORT_ANSWER, confidence=0.90)
        assert not result.promoted
        assert result.gate_failed == "length"

    def test_rejects_empty_answer(self, promoter: CachePromoter):
        result = promoter.evaluate(QUESTION_A, "", confidence=0.90)
        assert not result.promoted
        assert result.gate_failed == "length"

    def test_passes_at_word_boundary(self, promoter: CachePromoter):
        # Exactly 50 words
        fifty_word_answer = " ".join(["word"] * 50)
        result = promoter.evaluate(QUESTION_A, fifty_word_answer, confidence=0.90)
        assert result.gate_failed != "length"

    def test_passes_with_long_answer(self, promoter: CachePromoter):
        result = promoter.evaluate(QUESTION_A, LONG_ANSWER, confidence=0.90)
        assert result.gate_failed != "length"


# ─── Gate 3: dedup ────────────────────────────────────────────────────────────


class TestGateDedup:
    def test_rejects_near_duplicate_question(self, promoter: CachePromoter, additions_file: Path):
        # Promote QUESTION_A first
        promoter.promote(QUESTION_A, LONG_ANSWER, confidence=0.90)
        # Now try a near-duplicate
        result = promoter.evaluate(QUESTION_A_NEAR_DUP, LONG_ANSWER, confidence=0.90)
        assert not result.promoted
        assert result.gate_failed == "dedup"

    def test_rejects_exact_duplicate(self, promoter: CachePromoter, additions_file: Path):
        promoter.promote(QUESTION_A, LONG_ANSWER, confidence=0.90)
        result = promoter.evaluate(QUESTION_A, LONG_ANSWER, confidence=0.90)
        assert not result.promoted
        assert result.gate_failed == "dedup"

    def test_accepts_distinct_question(self, promoter: CachePromoter, additions_file: Path):
        promoter.promote(QUESTION_A, LONG_ANSWER, confidence=0.90)
        # QUESTION_B is clearly different
        result = promoter.evaluate(QUESTION_B, LONG_ANSWER, confidence=0.90)
        assert result.gate_failed != "dedup"

    def test_dedup_seeds_from_existing_cache(self, tmp_path: Path):
        mock_cache = MagicMock()
        mock_cache._cache = {QUESTION_A: {"answer": LONG_ANSWER}}
        promoter = CachePromoter(
            cache_tier=mock_cache,
            additions_file=tmp_path / "adds.jsonl",
        )
        # Near-dup of the pre-existing cache key should be rejected
        result = promoter.evaluate(QUESTION_A_NEAR_DUP, LONG_ANSWER, confidence=0.90)
        assert not result.promoted
        assert result.gate_failed == "dedup"


# ─── All gates pass ───────────────────────────────────────────────────────────


class TestEvaluatePassAll:
    def test_all_gates_pass(self, promoter: CachePromoter):
        result = promoter.evaluate(QUESTION_A, LONG_ANSWER, confidence=0.90)
        assert result.promoted
        assert result.gate_failed == ""
        assert result.passed  # alias property

    def test_reason_string_present(self, promoter: CachePromoter):
        result = promoter.evaluate(QUESTION_A, LONG_ANSWER, confidence=0.90)
        assert result.reason  # non-empty string


# ─── promote() ────────────────────────────────────────────────────────────────


class TestPromote:
    def test_returns_true_on_success(self, promoter: CachePromoter):
        assert promoter.promote(QUESTION_A, LONG_ANSWER, confidence=0.90) is True

    def test_returns_false_on_rejection(self, promoter: CachePromoter):
        assert promoter.promote(QUESTION_A, SHORT_ANSWER, confidence=0.90) is False

    def test_appends_to_jsonl_file(self, promoter: CachePromoter, additions_file: Path):
        promoter.promote(QUESTION_A, LONG_ANSWER, confidence=0.90, source="test")
        rows = [json.loads(l) for l in additions_file.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["question"] == QUESTION_A
        assert rows[0]["answer"] == LONG_ANSWER
        assert rows[0]["source"] == "test"
        assert "promoted_at" in rows[0]

    def test_nothing_written_on_rejection(self, promoter: CachePromoter, additions_file: Path):
        promoter.promote(QUESTION_A, SHORT_ANSWER, confidence=0.90)
        assert not additions_file.exists()

    def test_calls_cache_tier_add(self, tmp_path: Path):
        mock_cache = MagicMock()
        mock_cache._cache = {}
        p = CachePromoter(
            cache_tier=mock_cache,
            additions_file=tmp_path / "adds.jsonl",
        )
        p.promote(QUESTION_A, LONG_ANSWER, confidence=0.90, source="test")
        mock_cache.add.assert_called_once_with(QUESTION_A, LONG_ANSWER, source="test")

    def test_no_cache_tier_add_on_rejection(self, tmp_path: Path):
        mock_cache = MagicMock()
        mock_cache._cache = {}
        p = CachePromoter(
            cache_tier=mock_cache,
            additions_file=tmp_path / "adds.jsonl",
        )
        p.promote(QUESTION_A, SHORT_ANSWER, confidence=0.90)
        mock_cache.add.assert_not_called()

    def test_second_promote_of_same_question_rejected(self, promoter: CachePromoter):
        promoter.promote(QUESTION_A, LONG_ANSWER, confidence=0.90)
        result = promoter.evaluate(QUESTION_A, LONG_ANSWER, confidence=0.90)
        assert not result.promoted
        assert result.gate_failed == "dedup"


# ─── promote_from_file() ──────────────────────────────────────────────────────


class TestPromoteFromFile:
    def _write_candidates(self, tmp_path: Path, rows: list[dict]) -> Path:
        f = tmp_path / "candidates.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return f

    def test_counts_total_and_promoted(self, tmp_path: Path):
        rows = [
            {"question": QUESTION_A, "answer": LONG_ANSWER,
             "grounding_score": 0.90, "answer_in_chunk": True},
            {"question": QUESTION_B, "answer": LONG_ANSWER,
             "grounding_score": 0.90, "answer_in_chunk": True},
        ]
        f = self._write_candidates(tmp_path, rows)
        p = CachePromoter(additions_file=tmp_path / "adds.jsonl")
        report = p.promote_from_file(f)
        assert report.total == 2
        assert report.promoted == 2
        assert report.rejected == 0

    def test_skips_rows_with_answer_in_chunk_false(self, tmp_path: Path):
        rows = [
            {"question": QUESTION_A, "answer": LONG_ANSWER,
             "grounding_score": 0.90, "answer_in_chunk": False},
        ]
        f = self._write_candidates(tmp_path, rows)
        p = CachePromoter(additions_file=tmp_path / "adds.jsonl")
        report = p.promote_from_file(f)
        assert report.total == 0  # skipped entirely, not counted

    def test_rejects_low_confidence(self, tmp_path: Path):
        rows = [
            {"question": QUESTION_A, "answer": LONG_ANSWER,
             "grounding_score": 0.50, "answer_in_chunk": True},
        ]
        f = self._write_candidates(tmp_path, rows)
        p = CachePromoter(additions_file=tmp_path / "adds.jsonl")
        report = p.promote_from_file(f)
        assert report.rejected_confidence == 1
        assert report.promoted == 0

    def test_rejects_short_answers(self, tmp_path: Path):
        rows = [
            {"question": QUESTION_A, "answer": SHORT_ANSWER,
             "grounding_score": 0.90, "answer_in_chunk": True},
        ]
        f = self._write_candidates(tmp_path, rows)
        p = CachePromoter(additions_file=tmp_path / "adds.jsonl")
        report = p.promote_from_file(f)
        assert report.rejected_length == 1

    def test_dedup_across_batch(self, tmp_path: Path):
        # Two near-identical questions: only first should be promoted
        rows = [
            {"question": QUESTION_A, "answer": LONG_ANSWER,
             "grounding_score": 0.90, "answer_in_chunk": True},
            {"question": QUESTION_A_NEAR_DUP, "answer": LONG_ANSWER,
             "grounding_score": 0.90, "answer_in_chunk": True},
        ]
        f = self._write_candidates(tmp_path, rows)
        p = CachePromoter(additions_file=tmp_path / "adds.jsonl")
        report = p.promote_from_file(f)
        assert report.promoted == 1
        assert report.rejected_dedup == 1

    def test_report_summary_string(self, tmp_path: Path):
        f = self._write_candidates(tmp_path, [
            {"question": QUESTION_A, "answer": LONG_ANSWER,
             "grounding_score": 0.90, "answer_in_chunk": True},
        ])
        p = CachePromoter(additions_file=tmp_path / "adds.jsonl")
        report = p.promote_from_file(f)
        summary = report.summary()
        assert "1 promoted" in summary
        assert "Processed 1" in summary

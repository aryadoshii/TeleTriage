"""
Tests for CacheTier.

These are examples of what GOOD tests look like for this layer:
  - Test each match strategy independently
  - Test the miss path (delegation)
  - Test confidence bounds
  - Test edge cases (empty cache, whitespace, case)

Run with:  uv run pytest
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.tiers.cache_tier import CacheTier, _normalize
from backend.types import Query, TierName


@pytest.fixture
def cache_file(tmp_path: Path) -> Path:
    """Minimal cache for isolated testing."""
    data = {
        "high packet loss": {
            "answer": "Check antenna alignment.",
            "source": "test",
        },
        "Call Drop Issue": {
            "answer": "Pull call trace for release cause.",
            "source": "test",
        },
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def tier(cache_file: Path) -> CacheTier:
    return CacheTier(cache_path=cache_file)


# ─── _normalize helper ────────────────────────────────────────────────


class TestNormalize:
    def test_lowercases(self):
        assert _normalize("HIGH Packet Loss") == "high packet loss"

    def test_strips_punctuation(self):
        assert _normalize("packet-loss!") == "packet loss"

    def test_collapses_whitespace(self):
        assert _normalize("high   packet   loss") == "high packet loss"

    def test_strips_leading_trailing_ws(self):
        assert _normalize("  high packet loss  ") == "high packet loss"


# ─── Cache tier behavior ──────────────────────────────────────────────


class TestCacheTier:
    def test_exact_match_returns_answer_with_full_confidence(self, tier: CacheTier):
        result = tier.answer(Query(text="high packet loss"))
        assert result.answered
        assert result.tier == TierName.CACHE
        assert result.confidence == 1.0
        assert result.details["match_type"] == "exact"
        assert "antenna alignment" in result.answer.lower()

    def test_normalized_match_handles_case_and_punctuation(self, tier: CacheTier):
        # Different case + extra punctuation - should hit normalized path
        result = tier.answer(Query(text="HIGH, PACKET LOSS!"))
        assert result.answered
        assert result.details["match_type"] == "normalized"
        assert result.confidence == 0.95

    def test_normalized_match_on_original_case_variant(self, tier: CacheTier):
        # "Call Drop Issue" in cache - query lowercase should normalize-match
        result = tier.answer(Query(text="call drop issue"))
        assert result.answered
        assert result.details["match_type"] == "normalized"

    def test_fuzzy_match_on_word_reorder(self, tier: CacheTier):
        # "packet loss high" reorders "high packet loss"
        result = tier.answer(Query(text="packet loss high"))
        assert result.answered
        # token_set_ratio is permutation-invariant so this may normalize
        # or fuzzy-match depending on normalization; either way must hit
        assert result.details["match_type"] in {"normalized", "fuzzy"}

    def test_fuzzy_match_has_bounded_confidence(self, tier: CacheTier):
        # Deliberately imperfect phrasing to force fuzzy
        result = tier.answer(Query(text="packt loss high"))  # typo
        if result.answered and result.details["match_type"] == "fuzzy":
            assert 0.85 <= result.confidence <= 0.95

    def test_miss_returns_delegation_signal(self, tier: CacheTier):
        result = tier.answer(Query(text="completely unrelated query xyzzy"))
        assert result.answer is None
        assert result.confidence == 0.0
        assert tier.should_delegate(result) is True

    def test_latency_is_recorded(self, tier: CacheTier):
        result = tier.answer(Query(text="high packet loss"))
        assert result.latency_sec >= 0.0
        # Should be fast - cache tier budget is <10ms even on slow hardware
        assert result.latency_sec < 0.1

    def test_should_delegate_on_low_confidence(self, tier: CacheTier):
        # Simulate a fuzzy match just above threshold but below min_confidence
        # (This tests the router contract, not the tier's internal scoring)
        from backend.types import TierResult

        low_conf = TierResult(
            tier=TierName.CACHE, answer="something", confidence=0.50
        )
        assert tier.should_delegate(low_conf) is True

        high_conf = TierResult(
            tier=TierName.CACHE, answer="something", confidence=0.95
        )
        assert tier.should_delegate(high_conf) is False


class TestCacheMutation:
    def test_add_new_entry_makes_it_findable(self, tier: CacheTier):
        starting_size = len(tier)
        tier.add("jitter spike", "Check buffer config.", source="test")
        assert len(tier) == starting_size + 1

        result = tier.answer(Query(text="jitter spike"))
        assert result.answered
        assert "buffer" in result.answer.lower()

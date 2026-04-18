"""
Cache-Augmented Generation (CAG) Tier.

The fast path. Three match strategies, tried in order:

  1. EXACT match       - O(1) dict lookup. confidence = 1.00
  2. NORMALIZED match  - strip punctuation, lowercase, collapse whitespace.
                         O(1) dict lookup on a pre-built normalized index.
                         confidence = 0.95
  3. FUZZY match       - rapidfuzz token_set_ratio, O(n) but very fast
                         (pure C). Configurable threshold.
                         confidence scales with similarity.

Why this design?
- Most cache hits in practice are exact or near-exact repeats from logs,
  so exact/normalized covers >95% of hits at constant time.
- Fuzzy is the safety net for minor typos / word-order differences.
- We do NOT use semantic (embedding) matching here - that's the RAG tier's
  job. Keeping CAG purely lexical keeps it fast and the two tiers
  conceptually distinct.

Calibration note:
  The default fuzzy_threshold=90 is strict. If you find CAG missing
  obvious rephrases, lower to 85. If it's matching too loosely,
  raise to 92+. Measure on a validation set before deciding.
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from teletriage.config import get_config
from teletriage.tiers.base import BaseTier
from teletriage.types import Query, TierName, TierResult

# Precompile once at import time
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


class CacheTier(BaseTier):
    tier_name = TierName.CACHE

    def __init__(self, cache_path: Path | None = None):
        cfg = get_config()
        self.min_confidence = cfg.cache_tier.min_confidence
        self.fuzzy_threshold = cfg.cache_tier.fuzzy_threshold

        path = cache_path or cfg.resolve_path(cfg.paths.cache_file)
        self._cache: dict[str, dict[str, Any]] = self._load(path)

        # Pre-compute normalized index: normalized_question -> original_key
        # We keep original keys so we can report which one matched.
        self._normalized_index: dict[str, str] = {
            _normalize(k): k for k in self._cache
        }

        # For fuzzy: rapidfuzz wants a list of strings to search against
        self._keys_normalized: list[str] = list(self._normalized_index.keys())

    # ─── Loading ──────────────────────────────────────────────────────

    @staticmethod
    def _load(path: Path) -> dict[str, dict[str, Any]]:
        """
        Cache file format (sample_cache.json):
          {
            "high packet loss": {
                "answer": "Check antenna alignment...",
                "source": "3GPP TS 36.300",
                "added": "2026-04-01"
            },
            ...
          }
        """
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")
        with path.open() as f:
            return json.load(f)

    # ─── Public API ───────────────────────────────────────────────────

    def answer(self, query: Query) -> TierResult:
        start = self._now()
        text = query.text.strip()

        # 1. Exact match
        if text in self._cache:
            return self._hit(text, text, "exact", confidence=1.0, start=start)

        # 2. Normalized match
        normalized_query = _normalize(text)
        if normalized_query in self._normalized_index:
            original_key = self._normalized_index[normalized_query]
            return self._hit(
                text, original_key, "normalized", confidence=0.95, start=start
            )

        # 3. Fuzzy match (only if we have any keys)
        if self._keys_normalized:
            # token_set_ratio handles word reordering well
            # e.g. "packet loss high" vs "high packet loss" -> 100
            match = process.extractOne(
                normalized_query,
                self._keys_normalized,
                scorer=fuzz.token_set_ratio,
                score_cutoff=self.fuzzy_threshold,
            )
            if match is not None:
                matched_norm, score, _ = match
                original_key = self._normalized_index[matched_norm]
                # Map fuzzy score [threshold, 100] to confidence [0.85, 0.95]
                # - threshold score -> 0.85
                # - perfect 100     -> 0.95
                confidence = 0.85 + 0.10 * (
                    (score - self.fuzzy_threshold)
                    / max(1, 100 - self.fuzzy_threshold)
                )
                return self._hit(
                    text,
                    original_key,
                    "fuzzy",
                    confidence=round(confidence, 3),
                    start=start,
                    extra={"fuzzy_score": score},
                )

        # No match - delegate
        return self._empty_result(self._now() - start)

    # ─── Internals ────────────────────────────────────────────────────

    def _hit(
        self,
        query_text: str,
        matched_key: str,
        match_type: str,
        *,
        confidence: float,
        start: float,
        extra: dict[str, Any] | None = None,
    ) -> TierResult:
        entry = self._cache[matched_key]
        details: dict[str, Any] = {
            "match_type": match_type,
            "matched_key": matched_key,
            "source": entry.get("source"),
        }
        if extra:
            details.update(extra)

        return TierResult(
            tier=self.tier_name,
            answer=entry["answer"],
            confidence=confidence,
            latency_sec=self._now() - start,
            details=details,
        )

    # ─── Cache management (for Phase 6: adaptive cache) ───────────────

    def add(self, question: str, answer: str, **metadata: Any) -> None:
        """Add a new Q&A pair to the cache (in-memory only).
        Persistence to disk will come in Phase 6."""
        self._cache[question] = {"answer": answer, **metadata}
        self._normalized_index[_normalize(question)] = question
        self._keys_normalized = list(self._normalized_index.keys())

    def __len__(self) -> int:
        return len(self._cache)

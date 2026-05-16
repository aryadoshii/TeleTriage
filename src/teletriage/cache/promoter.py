"""
Adaptive cache promoter — Phase 6.

Decides whether a Q&A pair is good enough to enter the live cache, then
appends accepted pairs to data/cache_additions.jsonl and immediately
injects them into a running CacheTier instance.

━━━ QUALITY GATES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three gates are evaluated in order (fail-fast):

  1. CONFIDENCE  — caller-supplied score must be >= min_confidence (0.60).
                   Rejects answers retrieved or generated with low certainty.

  2. LENGTH      — answer must be >= min_words (50) significant words.
                   Rejects one-liners that give no actionable detail.

  3. DEDUP       — normalized edit distance to every existing cache key
                   must exceed near_dup_threshold (0.3).
                   Prevents the cache bloating with paraphrases that add
                   no new retrieval surface.  Uses RapidFuzz Levenshtein
                   normalized_distance (already a project dependency).

A pair that passes all three is "promotable".

━━━ PERSISTENCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

promote() appends to data/cache_additions.jsonl (JSONL, one row per pair)
so that promoted entries survive process restarts.  The CacheTier's
on-disk file (sample_cache.json) is intentionally NOT mutated — it remains
a curated, human-reviewed fixture.  On the next startup, operators can
review cache_additions.jsonl and merge chosen entries into sample_cache.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from rapidfuzz.distance import Levenshtein

if TYPE_CHECKING:
    from teletriage.tiers.cache_tier import CacheTier

_DEFAULT_ADDITIONS_FILE = Path("data/cache_additions.jsonl")


# ─── Result types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of evaluating a single Q&A pair against the quality gates."""

    promoted: bool
    reason: str          # human-readable explanation
    gate_failed: str     # "confidence" | "length" | "dedup" | "" (all passed)

    @property
    def passed(self) -> bool:
        return self.promoted


@dataclass
class PromotionReport:
    """Aggregate results from promote_from_file()."""

    total: int = 0
    promoted: int = 0
    rejected_confidence: int = 0
    rejected_length: int = 0
    rejected_dedup: int = 0

    @property
    def rejected(self) -> int:
        return self.rejected_confidence + self.rejected_length + self.rejected_dedup

    def summary(self) -> str:
        pct = 100 * self.promoted / max(self.total, 1)
        return (
            f"Processed {self.total} candidates: "
            f"{self.promoted} promoted ({pct:.0f}%), "
            f"{self.rejected} rejected "
            f"(confidence={self.rejected_confidence}, "
            f"length={self.rejected_length}, "
            f"dedup={self.rejected_dedup})"
        )


# ─── CachePromoter ────────────────────────────────────────────────────────────


class CachePromoter:
    """
    Evaluates Q&A pairs against three quality gates and promotes passing
    pairs into the live cache.

    Parameters
    ----------
    cache_tier:
        A live CacheTier instance.  promote() calls cache_tier.add() for
        immediate effect.  If None, pairs are only appended to the file.
    min_confidence:
        Minimum confidence score (Gate 1).  Default 0.60.
    min_words:
        Minimum word count in the answer (Gate 2).  Default 50.
    near_dup_threshold:
        Maximum allowed normalized edit distance to an existing cache key
        before a pair is rejected as a near-duplicate (Gate 3).
        Range [0, 1]; lower = stricter.  Default 0.3 (reject if distance
        is ≤ 0.3, i.e. ≥ 70% similar).
    additions_file:
        Where to append promoted pairs for persistence.
    """

    def __init__(
        self,
        cache_tier: "CacheTier | None" = None,
        *,
        min_confidence: float = 0.60,
        min_words: int = 50,
        near_dup_threshold: float = 0.3,
        additions_file: Path = _DEFAULT_ADDITIONS_FILE,
    ) -> None:
        self._cache_tier = cache_tier
        self.min_confidence = min_confidence
        self.min_words = min_words
        self.near_dup_threshold = near_dup_threshold
        self._additions_file = Path(additions_file)

        # Seed the dedup index with existing cache keys so we don't re-promote
        # variations of questions already in the cache.
        self._promoted_questions: list[str] = list(
            cache_tier._cache.keys() if cache_tier is not None else []
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def evaluate(self, question: str, answer: str, confidence: float) -> PromotionResult:
        """Run all three quality gates and return the result."""

        # Gate 1: confidence
        if confidence < self.min_confidence:
            return PromotionResult(
                promoted=False,
                reason=f"Confidence {confidence:.3f} < minimum {self.min_confidence}",
                gate_failed="confidence",
            )

        # Gate 2: answer length
        word_count = len(answer.split())
        if word_count < self.min_words:
            return PromotionResult(
                promoted=False,
                reason=f"Answer has {word_count} words, minimum is {self.min_words}",
                gate_failed="length",
            )

        # Gate 3: near-duplicate check
        dup_key = self._nearest_existing(question)
        if dup_key is not None:
            dist = Levenshtein.normalized_distance(question.lower(), dup_key.lower())
            return PromotionResult(
                promoted=False,
                reason=(
                    f"Near-duplicate of existing key (distance={dist:.3f} "
                    f"<= threshold={self.near_dup_threshold}): '{dup_key[:60]}'"
                ),
                gate_failed="dedup",
            )

        return PromotionResult(promoted=True, reason="All gates passed", gate_failed="")

    def promote(self, question: str, answer: str, confidence: float, **metadata: object) -> bool:
        """
        Evaluate and, if passing, persist + inject into the live cache.

        Returns True if the pair was promoted.
        """
        result = self.evaluate(question, answer, confidence)
        if not result.promoted:
            return False

        self._persist(question, answer, confidence, **metadata)

        if self._cache_tier is not None:
            self._cache_tier.add(question, answer, **metadata)

        # Track in local dedup index so subsequent calls within this session
        # don't promote near-duplicates of what we just added.
        self._promoted_questions.append(question)
        return True

    def promote_from_file(self, path: Path) -> PromotionReport:
        """
        Batch-promote from a JSONL file.

        Expected row schema (compatible with real_cache_candidates.jsonl):
            {
                "question": str,
                "answer": str,
                "grounding_score": float,   # used as confidence
                "source": str,              # optional metadata
                ...
            }

        Only rows where answer_in_chunk is true (or absent) are processed.
        """
        report = PromotionReport()
        path = Path(path)

        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                # Skip explicitly failed grounding rows
                if not row.get("answer_in_chunk", True):
                    continue

                report.total += 1
                question = row.get("question", "").strip()
                answer = row.get("answer", "").strip()
                confidence = float(row.get("grounding_score", 0.0))
                source = row.get("source", "")

                result = self.evaluate(question, answer, confidence)

                if result.promoted:
                    self._persist(question, answer, confidence, source=source)
                    if self._cache_tier is not None:
                        self._cache_tier.add(question, answer, source=source)
                    self._promoted_questions.append(question)
                    report.promoted += 1
                else:
                    if result.gate_failed == "confidence":
                        report.rejected_confidence += 1
                    elif result.gate_failed == "length":
                        report.rejected_length += 1
                    else:
                        report.rejected_dedup += 1

        return report

    # ─── Internals ────────────────────────────────────────────────────────────

    def _nearest_existing(self, question: str) -> str | None:
        """
        Return the existing question most similar to `question`, if that
        similarity exceeds the near_dup_threshold.  Returns None if the
        question is sufficiently novel.

        Uses normalized Levenshtein distance (chars).  For long questions
        this is O(n·m) per existing key, but cache sizes are typically
        hundreds to low thousands — acceptable.
        """
        q_lower = question.lower()
        for existing in self._promoted_questions:
            dist = Levenshtein.normalized_distance(q_lower, existing.lower())
            if dist <= self.near_dup_threshold:
                return existing
        return None

    def _persist(
        self, question: str, answer: str, confidence: float, **metadata: object
    ) -> None:
        """Append one promoted pair to the additions JSONL file."""
        self._additions_file.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        with self._additions_file.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

"""
BaseTier: the abstract interface every tier implements.

Design principle: each tier is a SELF-CONTAINED unit with ONE responsibility:
  Given a query, either produce an answer with a confidence score,
  or delegate by returning a low-confidence (or None-answer) TierResult.

The router handles the cascade logic - tiers know nothing about each other.
This makes them individually testable and swappable.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

from backend.types import Query, TierName, TierResult


class BaseTier(ABC):
    """Abstract base for CAG, RAG, and Generative tiers."""

    #: Every subclass declares which tier it is (used in TierResult & logging)
    tier_name: TierName

    #: Confidence threshold below which the router should delegate
    #: to the next tier. Set from config in __init__.
    min_confidence: float

    @abstractmethod
    def answer(self, query: Query) -> TierResult:
        """
        Attempt to answer the query.

        Contract:
          - MUST return a TierResult (never raise for "I don't know")
          - `answer=None` means "I can't answer, please delegate"
          - `confidence < self.min_confidence` also means "please delegate"
          - MUST set `latency_sec` honestly (use `_timed()` helper below)
        """
        ...

    def should_delegate(self, result: TierResult) -> bool:
        """Router asks: should I try the next tier?"""
        if result.answer is None:
            return True
        return result.confidence < self.min_confidence

    # ─── Helpers for subclasses ───────────────────────────────────────

    @staticmethod
    def _now() -> float:
        """Monotonic clock for reliable latency measurement."""
        return time.perf_counter()

    def _empty_result(self, elapsed: float) -> TierResult:
        """Standard 'I can't help' result - passes to next tier."""
        return TierResult(
            tier=self.tier_name,
            answer=None,
            confidence=0.0,
            latency_sec=elapsed,
            details={"reason": "no_match"},
        )

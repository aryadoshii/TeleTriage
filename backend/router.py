"""
Router: the cascade orchestrator.

Responsibilities:
  - Hold references to all enabled tiers in order
  - For each query: try tier 1. If it says "delegate", try tier 2. etc.
  - Accumulate tier trace (for debugging, eval, observability)
  - Return the final Response

Why a separate Router class instead of chaining tiers?
  - Tiers stay ignorant of each other (easier to test, swap, add new ones)
  - Central place to add cross-cutting concerns: logging, timing, metrics,
    caching the final result, request tracing, etc.
  - Easy to A/B test configurations (e.g. "router with vs without RAG")

Thread-safety: this class is NOT thread-safe. In production you'd
create one Router per worker, or add locks around tier state mutation
(e.g. adaptive cache updates in Phase 6).
"""
from __future__ import annotations

import logging
import time

from backend.config import get_config
from backend.tiers.base import BaseTier
from backend.tiers.cache_tier import CacheTier
from backend.tiers.generative_tier import GenerativeTier
from backend.tiers.retrieval_tier import RetrievalTier
from backend.types import Query, Response, TierName, TierResult

log = logging.getLogger(__name__)


class Router:
    def __init__(self, tiers: list[BaseTier] | None = None):
        """
        Args:
            tiers: ordered list of tiers to try. Pass None to auto-build
                   from config (standard CAG -> RAG -> Generative order).
        """
        if tiers is None:
            tiers = self._build_default_tiers()
        if not tiers:
            raise ValueError("Router needs at least one tier")
        self.tiers = tiers

    @staticmethod
    def _build_default_tiers() -> list[BaseTier]:
        cfg = get_config()
        tiers: list[BaseTier] = []

        if cfg.cache_tier.enabled:
            tiers.append(CacheTier())
        if cfg.retrieval_tier.enabled:
            tiers.append(RetrievalTier())
        if cfg.generative_tier.enabled:
            tiers.append(GenerativeTier())

        return tiers

    # ─── Main entry point ─────────────────────────────────────────────

    def route(self, query: Query | str) -> Response:
        """Run the cascade and return a Response."""
        if isinstance(query, str):
            query = Query(text=query)

        # `query` (the parameter) is never reassigned below — every
        # Response(query=...) call reports what the caller actually
        # submitted. `current_query` is the one actually handed to each
        # tier, and MAY be a different object than `query` once a tier
        # contributes candidates (see below).
        current_query = query

        start = time.perf_counter()
        trace: list[TierResult] = []
        # Accumulated across tiers as the cascade proceeds. Currently only
        # the retrieval tier ever populates TierResult.details["candidates"]
        # (see backend/tiers/retrieval_tier.py's _build_candidates), but
        # this loop is tier-agnostic by design — any tier that delegates
        # and leaves "candidates" in its details gets folded in here.
        candidates: list[dict] = []

        for tier in self.tiers:
            result = tier.answer(current_query)
            trace.append(result)

            log.debug(
                "tier=%s answered=%s conf=%.3f latency=%.3fs",
                result.tier.value,
                result.answered,
                result.confidence,
                result.latency_sec,
            )

            if not tier.should_delegate(result):
                # This tier is confident enough - we're done
                return Response(
                    query=query,
                    answer=result.answer or "",
                    answered_by=result.tier,
                    confidence=result.confidence,
                    total_latency_sec=time.perf_counter() - start,
                    tier_trace=trace,
                )

            # This tier delegated. If it surfaced candidates, carry them
            # forward as grounding context for whichever tier answers
            # next — see backend/generation/llm_client.py design
            # decision (1). Query is a frozen Pydantic model, so
            # "carrying forward" means building a NEW Query with enriched
            # metadata rather than mutating current_query in place.
            new_candidates = result.details.get("candidates")
            if new_candidates:
                candidates = candidates + list(new_candidates)
                current_query = Query(
                    text=current_query.text,
                    metadata={**current_query.metadata, "retrieved_context": candidates},
                    created_at=current_query.created_at,
                )

        # Every tier delegated. This shouldn't happen if the generative
        # tier is enabled (it always answers). If it does, surface a
        # helpful error rather than silently returning junk.
        elapsed = time.perf_counter() - start
        return Response(
            query=query,
            answer=(
                "No tier was able to answer this query. "
                "Check that the generative tier is enabled and configured."
            ),
            answered_by=TierName.NONE,
            confidence=0.0,
            total_latency_sec=elapsed,
            tier_trace=trace,
        )

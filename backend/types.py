"""
Core data types shared across tiers.

These are the CONTRACTS between components. Any tier takes a Query and
returns a TierResult. The router assembles the final Response from
whichever tier answered.

Keeping these as frozen Pydantic models (immutable) makes them safe to
pass around and trivial to serialize (for logging, caching, evaluation).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TierName(str, Enum):
    CACHE = "cache"
    RETRIEVAL = "retrieval"
    GENERATIVE = "generative"
    NONE = "none"  # nothing answered (should never happen in practice)


class Query(BaseModel):
    """An incoming user query."""

    model_config = ConfigDict(frozen=True)

    text: str
    # Optional context: network type, vendor, severity etc.
    # The RAG tier can use these as metadata filters later.
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TierResult(BaseModel):
    """What a single tier returns."""

    model_config = ConfigDict(frozen=True)

    tier: TierName
    # None = tier could not answer with enough confidence (delegate downstream)
    answer: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    # Latency in SECONDS (floats, not ints, so we can report sub-ms)
    latency_sec: float = 0.0
    # Free-form per-tier debug info:
    #   - cache tier: match_type (exact/normalized/fuzzy), matched_key
    #   - retrieval tier: retrieved_doc_ids, bm25_score, dense_score, rerank_score
    #   - generative tier: model_name, tokens_used, finish_reason
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.answer is not None


class Response(BaseModel):
    """Final response returned to the caller."""

    model_config = ConfigDict(frozen=True)

    query: Query
    answer: str
    answered_by: TierName
    confidence: float
    total_latency_sec: float
    # Every tier that was tried, in order. Useful for debugging and eval.
    tier_trace: list[TierResult]

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"[{self.answered_by.value}] "
            f"conf={self.confidence:.2f} "
            f"latency={self.total_latency_sec * 1000:.1f}ms"
        )

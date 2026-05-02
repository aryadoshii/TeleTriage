"""
Internal data types for the retrieval package.

The Hit dataclass is the currency that flows between every retrieval stage:
  BM25 search  →  [Hit, ...]
  Dense search →  [Hit, ...]
  RRF fusion   →  [Hit, ...]  (score field now holds RRF score)
  Reranker     →  [Hit, ...]  (score field now holds cross-encoder logit)

Keeping it as a plain dataclass (not Pydantic) keeps it lightweight —
these are created and discarded in hot paths, not persisted or validated
at system boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Hit:
    """
    A single retrieval result.

    The meaning of `score` changes at each pipeline stage:

      After BM25:     raw BM25 score (unbounded above 0; IDF-weighted term freq sum)
      After Dense:    cosine similarity in [-1, 1] (practically [0.3, 1] for relevant docs)
      After RRF:      fused rank score = Σ 1/(k + rank_i) across retrievers
      After Reranker: raw cross-encoder logit (roughly [-10, +10])

    The `doc` field carries the full KB document dict so downstream stages
    can access the answer text and metadata without additional lookups.
    """

    doc_id: str
    score: float
    doc: dict = field(default_factory=dict)
    rank: int = 0

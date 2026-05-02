"""
Reciprocal Rank Fusion (RRF) — score-free fusion of multiple ranked lists.

━━━ THE MATH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: we have two ranked lists (BM25 and dense) with incompatible scores.
  BM25 score:  an unbounded positive number (IDF-weighted term-frequency sum).
  Dense score: cosine similarity in [−1, 1] (practically [0.3, 1] for results).

Naïve score combination: normalise both to [0, 1], then α·bm25 + (1−α)·dense.
  Pitfalls:
    · Min/max normalisation is corpus-dependent — the "max" BM25 score changes
      with every new document and every new query.
    · What weight α is right?  It changes per domain, per query length.
    · Outliers in BM25 (a document that perfectly matches every query token)
      will dominate the normalised range.

Reciprocal Rank Fusion sidesteps all of this.  It operates only on RANKS:

  RRF_score(d) = Σᵢ  1 / (k + rankᵢ(d))

  where:
    rankᵢ(d) = position of document d in ranked list i  (1-indexed)
    k         = smoothing constant  (canonical: 60, from Cormack et al. 2009)

  Documents absent from a ranked list contribute 0 from that retriever.

WHY k = 60?
  The constant dampens the advantage of being ranked 1st vs 2nd:
    rank 1:   1/(60+1)  = 0.01639
    rank 2:   1/(60+2)  = 0.01613   ← only 1.6% less than rank 1
    rank 60:  1/(60+60) = 0.00833   ← 49% less than rank 1

  With k = 5 (too low):
    rank 1:  1/6  = 0.167
    rank 2:  1/7  = 0.143           ← 14% less → rank 1 dominates
    Consequence: a document that BM25 ranks 1st but dense ranks 60th will
    likely win even if a better document is consistently ranked 3rd by both.

  With k = 200 (too high):
    rank 1:  1/201 = 0.00498
    rank 60: 1/260 = 0.00385        ← only 23% less → ranks almost equal
    Consequence: we lose the signal in the top-10 ranks.

  k=60 is the empirical sweet spot from the original paper.  Robust across
  diverse TREC and CLEF test collections.

WORKED EXAMPLE with k=60, 2 retrievers, 3 docs:
  BM25 ranking:   [doc_A (1), doc_B (2), doc_C (3)]
  Dense ranking:  [doc_B (1), doc_A (2), doc_C (3)]

  RRF(doc_A) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
  RRF(doc_B) = 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252  ← tie
  RRF(doc_C) = 1/63 + 1/63 = 0.01587 + 0.01587 = 0.03175

  doc_A and doc_B are tied — the reranker breaks the tie using joint
  (query, doc) encoding, which is far more accurate.

Reference:  Cormack, Clarke, Buettcher (2009). "Reciprocal Rank Fusion
            Outperforms Condorcet and Individual Rank Learning Methods."
            ACM SIGIR 2009.
"""
from __future__ import annotations

from teletriage.retrieval._types import Hit


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hit]],
    k: int = 60,
    top_k: int | None = None,
) -> list[Hit]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists:  Each inner list must be sorted by descending score
                       (index 0 = best result, i.e. rank 1).
        k:             RRF smoothing constant.  Cormack et al. recommend 60.
        top_k:         If provided, return only the top-k fused results.

    Returns:
        A new list of Hit objects sorted by descending RRF score.
        The ``score`` field contains the RRF score (sum of 1/(k+rank)).
        The ``rank``  field reflects the new fused position (0-indexed).
        The ``doc``   field is carried through unchanged from the input hit.

    Notes:
        Documents that appear in only one retriever's list still get scored —
        they just receive 0 contribution from the other retriever(s).
        This is correct: absence from a list signals lower confidence, not
        total irrelevance.
    """
    # Accumulate: doc_id → cumulative RRF score, doc_id → doc dict
    rrf_scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank_idx, hit in enumerate(ranked_list):
            # rank_idx is 0-based; RRF rank is 1-based → add 1
            contribution = 1.0 / (k + rank_idx + 1)
            rrf_scores[hit.doc_id] = rrf_scores.get(hit.doc_id, 0.0) + contribution
            if hit.doc_id not in docs:
                docs[hit.doc_id] = hit.doc

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)

    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    return [
        Hit(doc_id=doc_id, score=rrf_scores[doc_id], doc=docs[doc_id], rank=rank)
        for rank, doc_id in enumerate(sorted_ids)
    ]

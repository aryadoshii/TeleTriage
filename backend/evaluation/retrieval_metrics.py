"""
Retrieval-native evaluation metrics — recall@k and MRR.

━━━ WHY THESE, SEPARATE FROM ROUGE-L / BERTSCORE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

backend/evaluation/metrics.py measures whether the FINAL ANSWER text is
close to a reference answer.  It says nothing about whether the retrieval
tier found the right SOURCE DOCUMENT in the first place — a confidently
wrong document can still produce a fluent, plausible-sounding answer that
scores reasonably on ROUGE/BERTScore while being substantively incorrect.

Concretely, the case that motivated this file: a query about EMM cause 11
retrieved a chunk about HSS timestamp validation — a different NAS
procedure that happens to share surface vocabulary ("MME", "reject",
"TAU") — with rerank confidence 0.69, comfortably above the 0.50
delegation threshold.  Answer-quality metrics alone can't distinguish
that from a genuinely correct retrieval; they were never given the
answer's SOURCE to check against anything.

recall_at_k() and mrr() measure retrieval quality directly, against a set
of human-labelled relevant document ids per query (see
backend/evaluation/relevance.py) — independent of what the reranker or
the generative tier eventually do with the retrieved documents.

━━━ THE MATH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recall@k
  Fraction of the KNOWN relevant documents that appear anywhere in the
  top-k retrieved results, for a single query:

    recall@k = |relevant ∩ retrieved[:k]| / |relevant|

  Recall@1 asks "is the single best-ranked document actually relevant?"
  — the question that matters most for this system, since only
  retrieved[0] is ever surfaced as an answer (see
  backend/tiers/retrieval_tier.py — best = reranked[0]).  Recall@5 asks a
  looser question: "is a relevant document anywhere in what the reranker
  considered?" — useful for telling a RANKING problem (relevant doc
  present but not ranked first) apart from a genuine RETRIEVAL problem
  (relevant doc never surfaced at all, at any k).

Reciprocal rank / MRR
  Reciprocal rank for one query is 1/rank of the FIRST relevant document
  in the retrieved list (rank is 1-indexed — the top result is rank 1),
  or 0 if no relevant document appears anywhere in the retrieved list:

    RR(q) = 1 / rank_of_first_relevant_hit,   or 0 if none found

  Mean Reciprocal Rank (MRR) — the metric conventionally meant by "MRR"
  in IR literature — is the MEAN of RR(q) across a set of queries.  That
  averaging is the CALLER's job (see scripts/eval_retrieval.py), not this
  function's: mrr() below computes the per-query reciprocal rank only.
  Call it once per query and average the results to get MRR proper.

  RR rewards ranking the right answer FIRST much more than merely
  including it somewhere in the top-k: a relevant doc at rank 1 scores
  1.0, at rank 2 scores 0.5, at rank 10 scores 0.1.  That steep falloff
  is the right shape for a system where — same as recall@1 — only the
  top-ranked result is ever actually used as the answer.
"""
from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Fraction of relevant_ids that appear within the first k retrieved_ids.

    Order of retrieved_ids matters (only the first k entries are
    considered — this is a RANKED list, not a set). Order of
    relevant_ids does not (it's treated as a set of ids).

    Returns 0.0 if:
      - relevant_ids is empty (no gold judgement to satisfy — a
        degenerate input; a query with a real judgement of "nothing in
        the KB is relevant" should still be represented some other way
        by the caller, not by an empty relevant_ids list, since 0/0 is
        undefined and 0.0 is the honest "found none of the (zero)
        targets" answer either way), or
      - k <= 0 (no candidates considered).
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    hits = len(top_k & relevant)
    return hits / len(relevant)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Reciprocal rank of the first relevant document in retrieved_ids, for
    ONE query.  See the module docstring for why "MRR" — the mean across
    a query set — is computed by the caller, not here.

    Returns 0.0 if retrieved_ids or relevant_ids is empty, or if no id in
    retrieved_ids is present in relevant_ids at all.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

"""
Human-labelled relevance judgements for retrieval evaluation.

RELEVANCE_JUDGEMENTS maps an eval query id (see
backend/evaluation/test_set.py — e.g. "eval_008") to the list of KB
document ids a human reviewer has confirmed are genuinely relevant to
that query.  This is the ground truth backend/evaluation/retrieval_metrics.py's
recall_at_k() and mrr() are scored against.

Starts EMPTY, on purpose.  Retrieval quality can only be measured against
real human judgement — a fabricated or guessed label here would make
scripts/eval_retrieval.py's numbers look authoritative while measuring
nothing real.  Populate it by running scripts/label_relevance.py, which
shows a reviewer the top-10 retrieved candidates per query and records
y/n judgements to a JSON working file (outputs/relevance_judgements.json
by default — see that script's docstring for the exact schema).  Once
labelling is done, either point scripts/eval_retrieval.py directly at
that JSON file (the default), or copy its "relevant" lists into
RELEVANCE_JUDGEMENTS below to commit them as the versioned, canonical set
— mirroring how backend/evaluation/test_set.py's EVAL_SET is hand-
maintained Python, not loaded from an external file.

A query can have zero relevant doc ids labelled — the reviewer looked at
the top-10 and found none of them actually relevant, which is a real,
informative outcome (the KB genuinely doesn't cover this query yet), not
missing data — or several, if multiple KB chunks legitimately answer it.
"""
from __future__ import annotations

RELEVANCE_JUDGEMENTS: dict[str, list[str]] = {}


def get_relevance_judgements() -> dict[str, list[str]]:
    """Return the full set of committed relevance judgements."""
    return dict(RELEVANCE_JUDGEMENTS)

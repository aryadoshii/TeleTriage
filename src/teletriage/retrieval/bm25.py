"""
BM25 sparse retriever backed by rank_bm25.

━━━ THE MATH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Start from TF-IDF, the grandfather of term-based scoring:

  TF-IDF(t, d) = TF(t, d) · IDF(t)

Two fundamental problems with raw TF-IDF:

  1. TF is unbounded.  A doc with "TEID" 100× scores 10× higher than one
     with "TEID" 10×.  But relevance doesn't scale that way — both docs
     are about TEIDs, the one mentioning it more isn't 10× more useful.

  2. Long documents get a free boost.  More tokens → more chances to
     accumulate TF, even if the document is only marginally relevant.

BM25 (Best Match 25) fixes both:

  BM25(t, d) = IDF(t) · tf · (k₁ + 1)
                        ──────────────────────────────────────────
                        tf + k₁ · (1 − b + b · |d| / avgdl)

  where:
    tf    = raw term frequency of t in d
    |d|   = number of tokens in d
    avgdl = average document length across the corpus
    k₁    = term-frequency saturation parameter  (default: 1.5)
             As tf → ∞, the TF contribution → (k₁ + 1).
             Small k₁: quick saturation (rare terms matter more).
             Large k₁: slower saturation (raw TF carries more weight).
             Empirically 1.2–2.0 works well across corpora.
    b     = length normalisation strength  (default: 0.75)
             b = 1 → full normalisation (long docs heavily penalised)
             b = 0 → no normalisation (same as classic TF-IDF on length)

  BM25 IDF (slightly different from classic log(N/df)):
    IDF(t) = log( (N − df + 0.5) / (df + 0.5) + 1 )
    The +0.5 is a Laplace-like smoothing to avoid zero IDF on common terms
    and prevent negative IDF for terms that appear in > half the corpus.

WHY BM25 STILL BEATS DENSE RETRIEVAL ON SOME QUERIES:
  Dense embedding of "TEID mismatch" might conflate it with other ID-related
  concepts in latent space.  BM25 assigns sky-high IDF to a rare term like
  "TEID" (appears in maybe 2 of 30 docs) and exact-matches the token.
  For rare, domain-specific terms, BM25 is often the correct answer.
  Hybrid retrieval (BM25 + dense + RRF) captures the best of both worlds.

━━━ IMPLEMENTATION NOTES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tokeniser:  lowercase → strip punctuation → split on whitespace.
  Deliberately simple — no stemming, no stopwords.  Stemming can hurt on
  technical corpora ("HARQ" != "HARQed"), and stopword lists from generic
  NLP libraries may remove important telecom acronyms.

Serialisation: pickle the entire (BM25Okapi, docs) state.
  rank_bm25 stores term frequencies and IDF weights in the BM25Okapi object.
  Pickling is fast and lossless.  Not cross-version safe, but rebuilding the
  index is cheap (< 1 s for our KB size) so that's acceptable.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from teletriage.retrieval._types import Hit


def _tokenize(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, split on whitespace.

    Module-level (not a method) so the exact same tokeniser is used at
    index-build time AND at query time — essential for consistent vocabulary.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # replace punctuation with space
    return text.split()


@dataclass
class _BM25State:
    """What we pickle to disk — the fitted BM25 model and the ordered doc list."""
    bm25: BM25Okapi
    docs: list[dict]


class BM25Retriever:
    """
    BM25 sparse retriever.

    Build from a doc list, then save/load the serialised state to avoid
    re-fitting on every startup.

    Usage::

        retriever = BM25Retriever.build(docs)
        retriever.save(path / "bm25.pkl")

        # later …
        retriever = BM25Retriever.load(path / "bm25.pkl")
        hits = retriever.search("packet loss LTE", k=10)
    """

    def __init__(self, bm25: BM25Okapi, docs: list[dict]) -> None:
        self._bm25 = bm25
        self._docs = docs

    @classmethod
    def build(cls, docs: list[dict]) -> "BM25Retriever":
        """
        Fit a BM25 index from a list of KB documents.

        Indexed text = question + " " + answer.  Concatenating both
        maximises keyword recall: a user might phrase their query like
        either the question OR a fragment of the answer.

        An empty corpus is allowed — search() will always return [].
        (BM25Okapi raises ZeroDivisionError on empty input, so we skip it.)
        """
        if not docs:
            return cls(None, [])  # type: ignore[arg-type]
        corpus_texts = [f"{d['question']} {d['answer']}" for d in docs]
        tokenized_corpus = [_tokenize(t) for t in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(bm25, docs)

    def search(self, query: str, k: int = 20) -> list[Hit]:
        """
        Return the top-k BM25 hits for a query string.

        Documents with score ≤ 0 are not returned — a zero BM25 score means
        the query and document share no vocabulary overlap at all.
        """
        if not self._docs or self._bm25 is None:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)  # ndarray of length N

        # argsort ascending → reverse for descending; slice top-k
        top_indices = scores.argsort()[::-1][:k]

        hits: list[Hit] = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0.0:
                break  # remaining docs have zero overlap — not useful candidates
            hits.append(Hit(
                doc_id=self._docs[idx]["id"],
                score=score,
                doc=self._docs[idx],
                rank=rank,
            ))
        return hits

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(_BM25State(bm25=self._bm25, docs=self._docs), f)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        with Path(path).open("rb") as f:
            state: _BM25State = pickle.load(f)
        return cls(state.bm25, state.docs)

    def __len__(self) -> int:
        return len(self._docs)

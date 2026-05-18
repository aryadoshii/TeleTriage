"""
Cross-encoder reranker — high-accuracy joint scoring of (query, doc) pairs.

━━━ THE MATH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BI-ENCODER (embedder.py) vs CROSS-ENCODER (this module):

  Bi-encoder: encodes query and document INDEPENDENTLY.
    query  →  encoder  →  q̂ ∈ ℝᵈ
    doc    →  encoder  →  v̂ ∈ ℝᵈ
    score  =  q̂ · v̂  (inner product = cosine for unit vectors)

    Advantage:  document vectors pre-computed offline → O(1) query time.
    Limitation: query and document tokens never "see each other" inside the
                transformer.  The model must compress all meaning into a single
                vector before comparison.  This loses fine-grained interaction.

  Cross-encoder: encodes the PAIR jointly.
    input:   [CLS] query tokens [SEP] document tokens [SEP]
    output:  a scalar logit from the final classification head

    The transformer's self-attention mechanism can now model interactions
    between every query token and every document token at every layer.
    Example: "LTE bearer" in the query attending to "S1-U" in the document
    creates a much richer matching signal than the bi-encoder equivalent.

    Cost: can't pre-compute — O(N) scoring time for N candidates.
    At 50 ms/pair on CPU, scoring 100 candidates = 5 s → too slow for retrieval.
    At 10 candidates = 0.5 s → acceptable for Phase 2.

STANDARD TWO-STAGE PIPELINE:
  Stage 1 (retrieval):  bi-encoder + BM25 retrieve K=20–100 candidates fast.
  Stage 2 (reranking):  cross-encoder rescores top-K → pick top-1.
  This is the same pattern used by state-of-the-art commercial search engines.

━━━ OUTPUT INTERPRETATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bge-reranker-base outputs a raw LOGIT (not a probability).
  logit  =  linear(last_hidden_state[CLS])   # single float, no activation

  Typical range: −10 to +10.
    logit > 0  → model thinks the pair is relevant
    logit < 0  → model thinks it is not relevant
    logit ≈ 0  → borderline

  Map to probability via sigmoid:
    p = σ(logit) = 1 / (1 + e^{−logit})

  Sigmoid values:
    σ(−10) ≈ 0.000045  → very low confidence
    σ(−2)  ≈ 0.12      → model leans "not relevant"
    σ(0)   = 0.5        → borderline
    σ(2)   ≈ 0.88      → model leans "relevant"
    σ(10)  ≈ 0.99995   → very high confidence

  We store the RAW LOGIT in Hit.score (to allow thresholding on the logit
  scale, which is linear and easier to reason about than probabilities near
  0 or 1).  We expose confidence_from_logit() for converting to [0, 1].

NUMERICALLY STABLE SIGMOID:
  For large positive x, e^{−x} ≈ 0 and 1/(1+0) = 1.0.
  For large negative x, e^{−x} is huge → overflow in float64.
  Safe implementation: if x ≥ 0, compute 1/(1+e^{−x}); else compute
  e^x / (1+e^x) which uses e^x < 1 so no overflow.
"""
from __future__ import annotations

import math

from sentence_transformers import CrossEncoder

from backend.retrieval._types import Hit


def _sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid: σ(x) = 1 / (1 + e^{−x}).

    Avoids float overflow for large |x| by choosing the numerically stable
    branch depending on the sign of x.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # x < 0: compute e^x/(1+e^x) — e^x is small (< 1), no overflow
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class CrossEncoderReranker:
    """
    Reranks candidate hits using a cross-encoder model.

    Builds (query, passage) pairs, scores them jointly, and re-sorts
    the hit list by descending logit.  Returns raw logits in Hit.score
    so the caller can apply their own threshold.

    Usage::

        reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
        reranked = reranker.rerank(query_text, fused_hits, top_k=5)
        best = reranked[0]
        conf = reranker.confidence_from_logit(best.score)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        top_k: int | None = None,
    ) -> list[Hit]:
        """
        Re-score and re-sort hits using joint (query, doc) encoding.

        Args:
            query:   The user query string.
            hits:    Candidate hits — typically the top-K from RRF fusion.
                     Order doesn't matter; cross-encoder rescores all of them.
            top_k:   If set, return only the top-K after reranking.

        Returns:
            Hits sorted by descending cross-encoder logit, with:
            - score: raw logit (threshold against config.min_rerank_score)
            - rank:  new 0-indexed position in the reranked list
            - doc:   unchanged from input

        The passage fed to the cross-encoder is question + " " + answer —
        same representation as the BM25 and dense corpus texts.
        """
        if not hits:
            return []

        # Build (query, passage) pairs for joint encoding
        pairs = [
            (query, f"{h.doc['question']} {h.doc['answer']}")
            for h in hits
        ]

        # predict() returns a numpy array of raw logits, shape (N,)
        logits = self._model.predict(pairs, convert_to_numpy=True)

        # Attach logits to hits, sort descending, update ranks
        scored = sorted(
            zip(hits, logits),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        result = [
            Hit(doc_id=h.doc_id, score=float(logit), doc=h.doc, rank=i)
            for i, (h, logit) in enumerate(scored)
        ]

        if top_k is not None:
            result = result[:top_k]

        return result

    def confidence_from_logit(self, logit: float) -> float:
        """
        Map a raw cross-encoder logit to a [0, 1] confidence score.

        Uses sigmoid so that:
          logit = 0   → confidence = 0.5  (borderline)
          logit > 2   → confidence > 0.88 (fairly confident)
          logit < -2  → confidence < 0.12 (low confidence)
        """
        return max(0.0, min(1.0, _sigmoid(logit)))

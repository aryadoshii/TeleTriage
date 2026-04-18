"""
Retrieval-Augmented Generation (RAG) Tier — PHASE 2 STUB.

This is a structured skeleton. Fill it in as Phase 2 of the rebuild.

ARCHITECTURE (to be implemented):
┌─────────────────────────────────────────────────────────────────┐
│                         Query                                   │
└─────────┬──────────────────────┬────────────────────────────────┘
          │                      │
   ┌──────▼──────┐        ┌──────▼──────┐
   │    BM25     │        │    Dense    │
   │   (sparse)  │        │ (BGE-small) │
   │  top_k=20   │        │  top_k=20   │
   └──────┬──────┘        └──────┬──────┘
          │                      │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │  Reciprocal Rank     │   RRF: combines rankings without
          │  Fusion (RRF)        │   needing to normalize scores.
          │  top_k=10            │   score = sum(1 / (k + rank_i))
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │  Cross-encoder       │   bi-encoders retrieve, cross-
          │  reranker            │   encoders RE-SCORE. Much more
          │  (bge-reranker-base) │   accurate, much slower per pair.
          │  final_top_k=1       │
          └──────────┬───────────┘
                     ▼
                 TierResult

CONCEPTS TO LEARN WHILE BUILDING THIS:
  - TF-IDF → BM25 math (why k1 and b matter)
  - Bi-encoder vs cross-encoder (why we need both)
  - Reciprocal Rank Fusion (Cormack et al. 2009)
  - FAISS index types: Flat vs IVF vs HNSW tradeoffs
  - Normalized embeddings + inner product = cosine similarity
  - Chunking strategies for long documents (fixed, recursive, semantic)

IMPLEMENTATION ORDER (suggested):
  1. retrieval/embedder.py     - wrap sentence-transformers
  2. retrieval/bm25.py         - wrap rank_bm25
  3. retrieval/dense.py        - FAISS index build + search
  4. retrieval/hybrid.py       - RRF fusion
  5. retrieval/reranker.py     - cross-encoder wrapper
  6. Wire them all up here in RetrievalTier.answer()
"""
from __future__ import annotations

from teletriage.config import get_config
from teletriage.tiers.base import BaseTier
from teletriage.types import Query, TierName, TierResult


class RetrievalTier(BaseTier):
    tier_name = TierName.RETRIEVAL

    def __init__(self) -> None:
        cfg = get_config()
        self.min_confidence = cfg.retrieval_tier.min_confidence
        self._cfg = cfg.retrieval_tier
        # TODO Phase 2: initialize these
        # self._embedder = Embedder(self._cfg.embedder_model)
        # self._bm25 = BM25Retriever.load(cfg.paths.knowledge_base)
        # self._dense = DenseRetriever.load(cfg.paths.faiss_index_dir)
        # self._reranker = CrossEncoderReranker(self._cfg.reranker_model)

    def answer(self, query: Query) -> TierResult:
        start = self._now()

        # TODO Phase 2: implement this properly. Pseudocode:
        #
        # 1. bm25_hits = self._bm25.search(query.text, k=self._cfg.bm25_top_k)
        # 2. dense_hits = self._dense.search(query.text, k=self._cfg.dense_top_k)
        # 3. fused = reciprocal_rank_fusion(
        #        [bm25_hits, dense_hits],
        #        k=self._cfg.rrf_k,
        #    )[: self._cfg.rerank_top_k]
        # 4. reranked = self._reranker.rerank(query.text, fused)
        # 5. best = reranked[0]
        # 6. if best.score < self._cfg.min_rerank_score: delegate
        #    else: return TierResult with best.doc.answer
        #
        # For confidence: map reranker logit through sigmoid, clamp to [0, 1].

        return self._empty_result(self._now() - start)

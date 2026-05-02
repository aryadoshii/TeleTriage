"""Retrieval components (Phase 2): BM25, dense (FAISS), hybrid RRF, cross-encoder reranker."""
from teletriage.retrieval._types import Hit
from teletriage.retrieval.bm25 import BM25Retriever
from teletriage.retrieval.dense import DenseRetriever
from teletriage.retrieval.embedder import Embedder
from teletriage.retrieval.hybrid import reciprocal_rank_fusion
from teletriage.retrieval.reranker import CrossEncoderReranker

__all__ = [
    "Hit",
    "Embedder",
    "BM25Retriever",
    "DenseRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
]

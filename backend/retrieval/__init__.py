"""Retrieval components (Phase 2): BM25, dense (FAISS), hybrid RRF, cross-encoder reranker."""
from backend.retrieval._types import Hit
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.embedder import Embedder
from backend.retrieval.hybrid import reciprocal_rank_fusion
from backend.retrieval.reranker import CrossEncoderReranker

__all__ = [
    "Hit",
    "Embedder",
    "BM25Retriever",
    "DenseRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
]

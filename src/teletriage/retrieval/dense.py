"""
Dense retriever: FAISS exact nearest-neighbour search over BGE embeddings.

━━━ THE MATH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dense retrieval encodes every document into a fixed-length vector offline,
then finds the "nearest" vectors to a query at runtime:

  Offline:  doc_text  →  encoder  →  d-dim vector v (stored in index)
  Online:   query     →  encoder  →  d-dim vector q
            nearest-neighbour search: find the k vectors v most similar to q

  "Similar" = cosine similarity = inner product after L2 normalisation.
  (See embedder.py for the normalisation proof.)

━━━ FAISS INDEX TYPES (and when to use each) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IndexFlatIP  ← what we use
  Stores raw vectors.  Brute-force scans all N vectors per query.
  Time:   O(N · d)  per query
  Space:  O(N · d · 4 bytes)  (float32)
  Recall: 100% — exact, no approximation.
  Limit:  Gets slow above ~100k docs on CPU.  Our KB: 30 docs → trivial.

IndexIVFFlat  (scale tier for 100k–10M docs)
  k-means clusters the corpus into `nlist` Voronoi cells at build time.
  At query time, only searches the `nprobe` closest cluster centroids.
  Time:   O(nprobe · N/nlist · d)  — configurable speed/recall tradeoff.
  Recall: Drops if nprobe is too small (true neighbours in un-searched cells).
  Gotcha: Needs at least 39 × nlist training vectors to fit well.

IndexHNSW  (Hierarchical Navigable Small World)
  Graph-based approximate NN.  Builds a layered proximity graph at index time.
  Time:   O(log N)  per query — very fast.
  Space:  Higher than Flat (stores graph edges: ~2× the vector storage).
  Recall: Tunable via `ef_search`; typically > 0.95 at default settings.
  Best for: Large interactive workloads where sub-50ms latency is critical.

For TeleTriage Phase 2 (30 docs, CPU, < 300ms budget): IndexFlatIP is correct.
Upgrade path: swap to IndexHNSW when KB grows past ~5k docs.

━━━ SERIALISATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FAISS has its own binary format (write_index / read_index).  We store doc
metadata alongside in faiss_docs.json — FAISS only stores float32 vectors,
so we maintain a positional map: vector at position i  ↔  docs[i].
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from teletriage.retrieval._types import Hit
from teletriage.retrieval.embedder import Embedder

_INDEX_FILE = "faiss.index"
_DOCS_FILE = "faiss_docs.json"


class DenseRetriever:
    """
    Dense retriever backed by FAISS IndexFlatIP.

    Encodes documents with an Embedder (offline), stores L2-normalised float32
    vectors.  At query time, encodes the query (with BGE prefix) and returns
    the top-k by inner product, which equals cosine similarity for unit vectors.

    Usage::

        embedder = Embedder("BAAI/bge-small-en-v1.5")
        retriever = DenseRetriever.build(docs, embedder)
        retriever.save(index_dir)

        # later …
        retriever = DenseRetriever.load(index_dir, embedder)
        hits = retriever.search("packet loss LTE", k=10)
    """

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        docs: list[dict],
        embedder: Embedder,
    ) -> None:
        self._index = index
        self._docs = docs
        self._embedder = embedder

    @classmethod
    def build(cls, docs: list[dict], embedder: Embedder) -> "DenseRetriever":
        """
        Encode all documents and build a FAISS IndexFlatIP.

        Indexed text = question + " " + answer — same corpus text as BM25
        for consistent coverage across both retrievers.
        """
        corpus_texts = [f"{d['question']} {d['answer']}" for d in docs]
        embeddings = embedder.encode_corpus(corpus_texts)  # (N, dim) float32

        index = faiss.IndexFlatIP(embedder.dim)
        index.add(embeddings)
        return cls(index, docs, embedder)

    def search(self, query: str, k: int = 20) -> list[Hit]:
        """
        Return top-k hits by cosine similarity (inner product of unit vectors).

        FAISS returns indices as int64; an index of -1 means an empty slot
        (can happen if k > index.ntotal) — we skip those.
        """
        if self._index.ntotal == 0:
            return []

        query_vec = self._embedder.encode_query(query)       # (1, dim)
        k_actual = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k_actual)
        # scores shape: (1, k_actual), indices shape: (1, k_actual)

        hits: list[Hit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue  # FAISS returns -1 for empty slots
            hits.append(Hit(
                doc_id=self._docs[idx]["id"],
                score=float(score),
                doc=self._docs[idx],
                rank=rank,
            ))
        return hits

    def save(self, index_dir: Path) -> None:
        """Persist FAISS binary index + doc metadata to directory."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_dir / _INDEX_FILE))
        with (index_dir / _DOCS_FILE).open("w") as f:
            json.dump(self._docs, f)

    @classmethod
    def load(cls, index_dir: Path, embedder: Embedder) -> "DenseRetriever":
        """Load a previously saved DenseRetriever from disk."""
        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / _INDEX_FILE))
        with (index_dir / _DOCS_FILE).open() as f:
            docs: list[dict] = json.load(f)
        return cls(index, docs, embedder)

    def __len__(self) -> int:
        return self._index.ntotal

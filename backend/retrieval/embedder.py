"""
Embedder: wraps sentence-transformers for dense (vector) retrieval.

━━━ THE MATH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bi-encoder pipeline (what this module implements):

  Input text  →  [CLS] tok₁ tok₂ ... tokN [SEP]
                        ↓  transformer layers  ↓
                 token-level hidden states h₀ h₁ ... hN
                        ↓  mean-pooling  ↓
                 sentence vector v = mean(h₁...hN)   # exclude CLS
                        ↓  L2 normalisation  ↓
                 v̂ = v / ‖v‖₂

WHY MEAN-POOLING instead of the CLS token?
  BERT trained CLS as a classification aggregate. Contrastively fine-tuned
  models (like BGE) pool ALL non-padding token states and average them.
  This gives the entire sequence equal representation rather than funnelling
  everything through one special token. Empirically it outperforms CLS on
  retrieval benchmarks.

WHY L2 NORMALISE?
  After normalisation ‖v̂‖ = 1.  For two unit vectors:
      v̂ · û = ‖v̂‖ ‖û‖ cos θ = cos θ

  So inner_product(v̂, û) == cosine_similarity(v̂, û).
  This lets FAISS IndexFlatIP (inner product search) act as exact cosine
  search, with similarity values in [−1, 1] — immediately interpretable.

BGE QUERY PREFIX:
  BGE models were contrastively trained with an asymmetric corpus:
    - queries had the prefix "Represent this sentence for searching relevant passages: "
    - documents were encoded WITHOUT any prefix
  At inference we must reproduce this asymmetry.  The model learned to
  project prefixed queries near the relevant unprefixed document embeddings.
  Removing the prefix at query time shifts the query embedding to the
  "document side" of the space, degrading retrieval accuracy.

━━━ DESIGN CHOICES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: BAAI/bge-small-en-v1.5
  - 33 M parameters, 384-dim embeddings, ~22 ms / batch on CPU.
  - Ranks highly on MTEB for its size class (better quality/latency tradeoff
    than larger models for our interactive latency budget of 300 ms).
  - v1.5 fine-tuned on more diverse data than v1; better OOD generalisation.
  - Upgrade path: bge-base-en-v1.5 (109 M, 768-dim) if quality is lacking.

Batched encoding:
  Always call encode() on a list, not one string at a time.
  sentence-transformers handles padding and truncation per-batch, and the
  GPU/CPU utilisation is dramatically better at batch size ≥ 16.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

# BGE retrieval models expect this prefix on queries (not on documents).
# Source: BAAI/bge-small-en-v1.5 model card on HuggingFace.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """
    Thin wrapper around SentenceTransformer for retrieval embeddings.

    Encodes queries with the BGE asymmetric prefix; corpus documents as-is.
    All returned vectors are L2-normalised float32 so inner product ≡ cosine.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model = SentenceTransformer(model_name)
        # Cache the embedding dimension so FAISS index can be sized correctly.
        self.dim: int = self._model.get_sentence_embedding_dimension()

    def encode_corpus(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> NDArray[np.float32]:
        """
        Encode a list of document texts for offline indexing.

        No query prefix — BGE corpus documents are plain text.
        Returns float32 array of shape (N, dim), each row L2-normalised.
        """
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # L2 norm so IP = cosine
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """
        Encode a single query string with the BGE asymmetric prefix.

        Returns float32 array of shape (1, dim), L2-normalised.
        Wrapping in a list produces a 2-D array which FAISS search() expects.
        """
        prefixed = _BGE_QUERY_PREFIX + query
        vec = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

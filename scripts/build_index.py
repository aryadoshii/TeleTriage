"""
Build retrieval indexes from the knowledge base — PHASE 2 SCRIPT.

Run once after adding/updating data/sample_kb.jsonl to rebuild:
  - BM25 index (pickled sparse representation)
  - FAISS dense index (embedded with BGE-small)

Usage (after Phase 2 implementation):
    uv run python scripts/build_index.py

PSEUDOCODE for when you implement this:

    from teletriage.config import get_config
    from teletriage.retrieval.embedder import Embedder
    from teletriage.retrieval.bm25 import BM25Retriever
    from teletriage.retrieval.dense import DenseRetriever

    cfg = get_config()

    # 1. Load KB
    kb_docs = load_jsonl(cfg.paths.knowledge_base)

    # 2. Build BM25
    bm25 = BM25Retriever.build(kb_docs)
    bm25.save(cfg.paths.faiss_index_dir / "bm25.pkl")

    # 3. Build dense index
    embedder = Embedder(cfg.retrieval_tier.embedder_model)
    dense = DenseRetriever.build(kb_docs, embedder)
    dense.save(cfg.paths.faiss_index_dir / "faiss.index")

    print(f"Indexed {len(kb_docs)} docs")
"""
from __future__ import annotations

import sys


def main() -> int:
    print(
        "Phase 2 not yet implemented.\n"
        "Next steps:\n"
        "  1. Create src/teletriage/retrieval/embedder.py\n"
        "  2. Create src/teletriage/retrieval/bm25.py\n"
        "  3. Create src/teletriage/retrieval/dense.py\n"
        "  4. Wire up this script.\n"
        "Use Claude Code in VS Code to scaffold these - the interfaces are\n"
        "already sketched in src/teletriage/tiers/retrieval_tier.py."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

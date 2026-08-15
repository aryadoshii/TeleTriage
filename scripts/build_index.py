"""
Build retrieval indexes from the knowledge base.

Run once (or whenever data/sample_kb.jsonl changes):

    uv run python scripts/build_index.py

What this does:
  1. Loads every document from data/sample_kb.jsonl
  2. Builds a BM25 index (pickled BM25Okapi) → data/indexes/bm25.pkl
  3. Encodes all documents with BGE-small → builds a FAISS IndexFlatIP
     → data/indexes/faiss.index  +  data/indexes/faiss_docs.json

After this, `uv run teletriage ask "..."` will use the retrieval tier.

Flags:
  --kb PATH        Override the knowledge base path (default: from config)
  --out DIR        Override the output index directory  (default: from config)
  --quiet          Suppress progress output
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

from backend.config import get_config
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.embedder import Embedder

app = typer.Typer(add_completion=False)
console = Console()

MANIFEST_HASH_BYTES = 1_000_000  # hash only the first 1 MB — cheap change-detector


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def _sha256_prefix(path: Path, n_bytes: int = MANIFEST_HASH_BYTES) -> str:
    """SHA-256 over just the first n_bytes of a file.

    Hashing all 7,181 KB entries on every build is unnecessary for what
    this is used for (detecting "did someone swap the KB file under the
    index's feet") — the first megabyte changes if the file's content
    changes for any of the ways this repo actually swaps KBs (different
    file entirely, regenerated pipeline output, edited seed doc).
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


@app.command()
def main(
    kb: Path = typer.Option(None, help="Knowledge base JSONL path (overrides config)"),
    out: Path = typer.Option(None, help="Output index directory (overrides config)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
) -> None:
    cfg = get_config()

    kb_path = kb or cfg.resolve_path(cfg.paths.knowledge_base)
    index_dir = out or cfg.resolve_path(cfg.paths.faiss_index_dir)

    if not quiet:
        console.print(f"[bold]TeleTriage index builder[/bold]")
        console.print(f"  KB:        {kb_path}")
        console.print(f"  Index dir: {index_dir}")
        console.print()

    # ── 1. Load documents ─────────────────────────────────────────────────
    if not quiet:
        console.print("Loading documents…")
    docs = load_jsonl(kb_path)
    if not quiet:
        console.print(f"  Loaded [bold]{len(docs)}[/bold] documents.\n")

    # ── 2. Build BM25 index ───────────────────────────────────────────────
    if not quiet:
        console.print("Building BM25 index…")
    t0 = time.perf_counter()
    bm25 = BM25Retriever.build(docs)
    bm25_path = index_dir / "bm25.pkl"
    bm25.save(bm25_path)
    bm25_ms = (time.perf_counter() - t0) * 1000
    if not quiet:
        console.print(f"  Saved → {bm25_path}  ({bm25_ms:.0f} ms)\n")

    # ── 3. Build dense (FAISS) index ──────────────────────────────────────
    embedder_model = cfg.retrieval_tier.embedder_model
    if not quiet:
        console.print(f"Loading embedder [dim]{embedder_model}[/dim]…")
        console.print("  (first run downloads ~30 MB of model weights)")
    t0 = time.perf_counter()
    embedder = Embedder(embedder_model)
    embed_load_ms = (time.perf_counter() - t0) * 1000
    if not quiet:
        console.print(f"  Loaded in {embed_load_ms:.0f} ms.  Dim = {embedder.dim}\n")

    if not quiet:
        console.print("Encoding documents and building FAISS index…")
    t0 = time.perf_counter()
    dense = DenseRetriever.build(docs, embedder)
    dense.save(index_dir)
    dense_ms = (time.perf_counter() - t0) * 1000
    if not quiet:
        console.print(f"  Saved → {index_dir}/faiss.index  ({dense_ms:.0f} ms)\n")

    # ── 4. Write manifest ─────────────────────────────────────────────────
    # Records which KB file these indexes actually came from, so anything
    # loading the index later (RetrievalTier, `teletriage info`, the
    # dashboard) can detect a stale/mismatched build instead of silently
    # searching the wrong corpus.
    if not quiet:
        console.print("Writing manifest…")
    manifest = {
        "kb_path": str(kb_path.resolve()),
        "kb_filename": kb_path.name,
        "n_docs": len(docs),
        "embedder_model": embedder_model,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "kb_sha256": _sha256_prefix(kb_path),
    }
    manifest_path = index_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    if not quiet:
        console.print(f"  Saved → {manifest_path}\n")

    # ── 5. Summary ────────────────────────────────────────────────────────
    if not quiet:
        console.print("[green]✓ Index build complete.[/green]")
        console.print(f"  {len(docs)} documents indexed from {kb_path.name}.")
        console.print(f"  BM25:     {bm25_path}")
        console.print(f"  FAISS:    {index_dir / 'faiss.index'}")
        console.print(f"  Docs:     {index_dir / 'faiss_docs.json'}")
        console.print(f"  Manifest: {manifest_path}")
        console.print()
        console.print("Run queries with:")
        console.print("  [bold]uv run teletriage ask \"intermittent packet loss on LTE\"[/bold]")


if __name__ == "__main__":
    app()
    
"""
Retrieval ablation: BM25-only vs dense-only vs RRF-fused vs RRF+reranked,
scored against human relevance judgements.

backend/evaluation/metrics.py (ROUGE-L/BERTScore) measures END-TO-END
answer quality. It cannot tell you whether a specific retriever stage
found the right document, or whether the reranker's confidence is
actually earning its keep — those questions need a document-identity
ground truth, not a reference-answer-text one. That's what
backend/evaluation/relevance.py's human judgements + recall@k/MRR (see
backend/evaluation/retrieval_metrics.py) are for.

For every query with a relevance judgement on file, this script runs all
four retrieval configurations that exist in the pipeline (see
backend/tiers/retrieval_tier.py) side by side against the SAME indexes,
and reports recall@1, recall@5, and MRR for each — so you can see, e.g.,
whether RRF fusion is actually beating BM25/dense alone, and whether the
reranker is actually improving on raw RRF ranking rather than just
reordering noise.

Usage:
    uv run python scripts/eval_retrieval.py
    uv run python scripts/eval_retrieval.py --judgements outputs/relevance_judgements.json
    uv run python scripts/eval_retrieval.py --top-k 10

Prerequisites:
    1. Build retrieval indexes: uv run python scripts/build_index.py
    2. Label relevance:         uv run python scripts/label_relevance.py

This script does not fabricate relevance judgements. If none are found,
it exits with an error rather than running the ablation against nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

from backend.config import get_config
from backend.evaluation.relevance import RELEVANCE_JUDGEMENTS
from backend.evaluation.retrieval_metrics import mrr, recall_at_k
from backend.evaluation.test_set import get_eval_set
from backend.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    Embedder,
    reciprocal_rank_fusion,
)
from scripts.label_relevance import DEFAULT_JUDGEMENTS_FILE, _load_judgements

app = typer.Typer(add_completion=False)
console = Console()

CONFIGS = ["bm25", "dense", "rrf", "rrf+rerank"]


def _load_labelled_queries(judgements_path: Path) -> dict[str, list[str]]:
    """
    Merge relevance judgements from the JSON working file (produced by
    scripts/label_relevance.py) with anything already committed to
    backend/evaluation/relevance.py's RELEVANCE_JUDGEMENTS. The file
    takes precedence per query id when both have an entry — it's the
    more recently updated source during active labelling.
    """
    merged: dict[str, list[str]] = dict(RELEVANCE_JUDGEMENTS)
    for qid, data in _load_judgements(judgements_path).items():
        merged[qid] = data.get("relevant", [])
    return merged


@app.command()
def main(
    judgements: Path = typer.Option(
        DEFAULT_JUDGEMENTS_FILE, "--judgements", "-j",
        help="Relevance judgements JSON (scripts/label_relevance.py's output)",
    ),
    top_k: int = typer.Option(
        10, "--top-k", help="Candidates retrieved per stage (recall@5 needs >= 5)",
    ),
) -> None:
    """Run the BM25/dense/RRF/reranked retrieval ablation against labelled queries."""
    cfg = get_config()
    labelled = _load_labelled_queries(judgements)

    if not labelled:
        console.print(
            f"[red]No relevance judgements found at {judgements} "
            f"(and none committed in backend/evaluation/relevance.py).[/red]\n"
            f"Run [bold]uv run python scripts/label_relevance.py[/bold] first — "
            f"this script does not fabricate judgements."
        )
        raise typer.Exit(1)

    eval_items = {item.id: item for item in get_eval_set()}
    queries = [(qid, eval_items[qid].query) for qid in labelled if qid in eval_items]
    skipped = sorted(set(labelled) - set(eval_items))
    if skipped:
        console.print(
            f"[yellow]  {len(skipped)} labelled query id(s) not found in the current "
            f"eval set, skipped: {skipped}[/yellow]"
        )

    if not queries:
        console.print("[red]No labelled query ids matched the current eval set.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]TeleTriage — Retrieval Ablation[/bold]")
    console.print(f"  Labelled queries: {len(queries)}")
    console.print(f"  Judgements file:  {judgements}")
    console.print(f"  Candidates/stage: {top_k}\n")

    console.print("Loading retrieval components (embedder + reranker)…")
    try:
        embedder = Embedder(cfg.retrieval_tier.embedder_model)
        index_dir = cfg.resolve_path(cfg.paths.faiss_index_dir)
        bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
        dense = DenseRetriever.load(index_dir, embedder)
        reranker = CrossEncoderReranker(cfg.retrieval_tier.reranker_model)
    except FileNotFoundError as exc:
        console.print(f"[red]Could not load retrieval indexes: {exc}[/red]")
        console.print("Run [bold]uv run python scripts/build_index.py[/bold] first.")
        raise typer.Exit(1)

    # config -> metric name -> list of per-query scores
    scores: dict[str, dict[str, list[float]]] = {
        c: {"recall@1": [], "recall@5": [], "rr": []} for c in CONFIGS
    }

    for qid, query_text in queries:
        relevant_ids = labelled[qid]
        if not relevant_ids:
            # A real, recorded judgement of "nothing in the top-10 was
            # relevant" — recall_at_k/mrr both correctly score 0.0 for
            # every config against an empty relevant_ids list, so this
            # query still contributes honest (if maximally harsh) signal.
            console.print(f"[dim]  {qid}: 0 relevant docs labelled (informative, not skipped)[/dim]")

        bm25_hits = bm25.search(query_text, k=cfg.retrieval_tier.bm25_top_k)
        dense_hits = dense.search(query_text, k=cfg.retrieval_tier.dense_top_k)
        fused = reciprocal_rank_fusion(
            [bm25_hits, dense_hits], k=cfg.retrieval_tier.rrf_k, top_k=top_k,
        )
        reranked = reranker.rerank(query_text, fused, top_k=top_k) if fused else []

        per_config_hits = {
            "bm25": bm25_hits,
            "dense": dense_hits,
            "rrf": fused,
            "rrf+rerank": reranked,
        }

        for cname, hits in per_config_hits.items():
            ids = [h.doc_id for h in hits]
            scores[cname]["recall@1"].append(recall_at_k(ids, relevant_ids, 1))
            scores[cname]["recall@5"].append(recall_at_k(ids, relevant_ids, 5))
            scores[cname]["rr"].append(mrr(ids, relevant_ids))

    # ─── Report ──────────────────────────────────────────────────────────
    table = Table(title="Retrieval Ablation", show_header=True, header_style="bold")
    table.add_column("Config", style="cyan")
    table.add_column("Recall@1", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("n", justify="right")

    for cname in CONFIGS:
        n = len(scores[cname]["rr"])
        r1 = sum(scores[cname]["recall@1"]) / n if n else 0.0
        r5 = sum(scores[cname]["recall@5"]) / n if n else 0.0
        m = sum(scores[cname]["rr"]) / n if n else 0.0
        table.add_row(cname, f"{r1:.3f}", f"{r5:.3f}", f"{m:.3f}", str(n))

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    app()

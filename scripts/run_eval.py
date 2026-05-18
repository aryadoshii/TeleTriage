"""
Run the TeleTriage evaluation harness.

Usage:
    uv run python scripts/run_eval.py                        # full eval, saves to logs/
    uv run python scripts/run_eval.py --out logs/my_run.json
    uv run python scripts/run_eval.py --subset retrieval     # only retrieval-expected queries
    uv run python scripts/run_eval.py --no-generative        # skip generative tier (faster)
    uv run python scripts/run_eval.py --bertscore-model roberta-large  # higher quality BERTScore

Prerequisites:
    1. Build retrieval indexes:  uv run python scripts/build_index.py
    2. Set API key in .env:      GROQ_API_KEY=gsk_...
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

from backend.config import get_config
from backend.evaluation.evaluator import Evaluator
from backend.evaluation.test_set import get_eval_set, get_eval_set_by_expected_tier
from backend.router import Router
from backend.tiers.cache_tier import CacheTier
from backend.tiers.generative_tier import GenerativeTier
from backend.tiers.retrieval_tier import RetrievalTier

app = typer.Typer(add_completion=False)
console = Console()


def _build_router(include_generative: bool, no_retrieval: bool = False) -> Router:
    """Build a router, gracefully skipping tiers whose prerequisites are missing."""
    from backend.config import get_config
    cfg = get_config()

    tiers = []

    # Cache tier — always available
    tiers.append(CacheTier())

    # Retrieval tier — needs pre-built indexes
    if not no_retrieval:
        index_dir = cfg.resolve_path(cfg.paths.faiss_index_dir)
        if not (index_dir / "bm25.pkl").exists():
            console.print(
                "[yellow]Warning:[/yellow] Retrieval indexes not found. "
                "Run [bold]uv run python scripts/build_index.py[/bold] first. "
                "Retrieval tier will be skipped."
            )
        else:
            tiers.append(RetrievalTier())

    # Generative tier — optional (slow, needs API key)
    if include_generative:
        tiers.append(GenerativeTier())

    return Router(tiers=tiers)


@app.command()
def main(
    out: Path = typer.Option(
        None, "--out", "-o",
        help="Path for JSON report output (default: logs/eval_TIMESTAMP.json)",
    ),
    subset: str = typer.Option(
        None, "--subset",
        help="Filter eval set by expected tier: 'retrieval' or 'generative'",
    ),
    no_generative: bool = typer.Option(
        False, "--no-generative",
        help="Skip the generative tier (faster; tests cache + retrieval only)",
    ),
    bertscore_model: str = typer.Option(
        "distilbert-base-uncased", "--bertscore-model",
        help="Transformer model for BERTScore (distilbert-base-uncased or roberta-large)",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    cfg = get_config()

    # ── Load eval set ──────────────────────────────────────────────────
    if subset:
        items = get_eval_set_by_expected_tier(subset)
        if not items:
            console.print(f"[red]No eval items found for tier '{subset}'.[/red]")
            raise typer.Exit(1)
    else:
        items = get_eval_set()

    if not quiet:
        console.print(f"\n[bold]TeleTriage Evaluation Harness[/bold]")
        console.print(f"  Queries:       {len(items)}")
        console.print(f"  BERTScore:     {bertscore_model}")
        console.print(f"  Generative:    {'disabled' if no_generative else 'enabled'}")
        console.print()

    # ── Build router ───────────────────────────────────────────────────
    try:
        router = _build_router(include_generative=not no_generative)
    except Exception as exc:
        console.print(f"[red]Failed to initialise router: {exc}[/red]")
        raise typer.Exit(1)

    # ── Run evaluation ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    evaluator = Evaluator(bertscore_model=bertscore_model, verbose=not quiet)

    try:
        report = evaluator.run(router, items)
    except Exception as exc:
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        raise typer.Exit(1)

    wall_sec = time.perf_counter() - t0

    # ── Save JSON report first (before model cleanup can crash) ────────
    if out is None:
        ts = report.timestamp.replace(":", "-").replace("+", "").split(".")[0]
        log_dir = cfg.resolve_path(cfg.paths.log_dir)
        out = log_dir / f"eval_{ts}.json"

    report.save(out)

    # ── Print summary ──────────────────────────────────────────────────
    print(evaluator.summary_table(report))

    if not quiet:
        console.print(f"\n  Wall-clock time: {wall_sec:.1f}s")
        console.print(f"  Report saved → [bold]{out}[/bold]\n")


if __name__ == "__main__":
    app()

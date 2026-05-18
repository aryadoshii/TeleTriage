"""
Batch-promote Q&A candidates into the live cache.

Usage:
    uv run python scripts/promote_cache.py
    uv run python scripts/promote_cache.py --input data/real_cache_candidates.jsonl
    uv run python scripts/promote_cache.py --min-confidence 0.70 --min-words 60
    uv run python scripts/promote_cache.py --dry-run   # evaluate gates, no writes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

from backend.cache.promoter import CachePromoter
from backend.tiers.cache_tier import CacheTier

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    input: Path = typer.Option(
        Path("database/real_cache_candidates.jsonl"),
        "--input", "-i",
        help="JSONL file to promote from (real_cache_candidates.jsonl schema)",
    ),
    min_confidence: float = typer.Option(0.60, "--min-confidence"),
    min_words: int = typer.Option(50, "--min-words"),
    near_dup_threshold: float = typer.Option(
        0.3, "--near-dup-threshold",
        help="Normalized edit distance threshold; pairs closer than this are rejected",
    ),
    additions_file: Path = typer.Option(
        Path("database/cache_additions.jsonl"), "--additions-file",
        help="Where promoted pairs are appended",
    ),
    report_file: Path = typer.Option(
        Path("outputs/promotion_report.json"), "--report-file",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Run gates and print report without writing any files",
    ),
) -> None:
    """Promote Q&A candidates to the cache after passing quality gates."""
    if not input.exists():
        console.print(f"[red]Input file not found: {input}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]TeleTriage — Cache Promoter[/bold]")
    console.print(f"  Input:             {input}")
    console.print(f"  Min confidence:    {min_confidence}")
    console.print(f"  Min words:         {min_words}")
    console.print(f"  Near-dup threshold:{near_dup_threshold}")
    console.print(f"  Dry run:           {dry_run}\n")

    # Load the live cache tier so promoter can call .add() immediately
    try:
        cache_tier: CacheTier | None = CacheTier()
    except FileNotFoundError as exc:
        console.print(f"[yellow]  Warning: could not load CacheTier ({exc}). "
                      f"Pairs will still be written to file.[/yellow]")
        cache_tier = None

    effective_additions = Path("/dev/null") if dry_run else additions_file

    promoter = CachePromoter(
        cache_tier=cache_tier,
        min_confidence=min_confidence,
        min_words=min_words,
        near_dup_threshold=near_dup_threshold,
        additions_file=effective_additions,
    )

    report = promoter.promote_from_file(input)

    # ─── Print table ──────────────────────────────────────────────────────────
    table = Table(title="Promotion Report", show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Count", justify="right")
    table.add_column("Pct", justify="right")

    def pct(n: int) -> str:
        return f"{100 * n / max(report.total, 1):.0f}%"

    table.add_row("Total candidates",      str(report.total),                "100%")
    table.add_row("[green]Promoted[/green]", f"[green]{report.promoted}[/green]", f"[green]{pct(report.promoted)}[/green]")
    table.add_row("Rejected — confidence", str(report.rejected_confidence),  pct(report.rejected_confidence))
    table.add_row("Rejected — length",     str(report.rejected_length),       pct(report.rejected_length))
    table.add_row("Rejected — dedup",      str(report.rejected_dedup),        pct(report.rejected_dedup))

    console.print(table)

    if dry_run:
        console.print("[yellow]  Dry run — no files written.[/yellow]\n")
        return

    # ─── Save JSON report ─────────────────────────────────────────────────────
    report_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": str(input),
        "total": report.total,
        "promoted": report.promoted,
        "rejected_confidence": report.rejected_confidence,
        "rejected_length": report.rejected_length,
        "rejected_dedup": report.rejected_dedup,
        "additions_file": str(additions_file),
        "config": {
            "min_confidence": min_confidence,
            "min_words": min_words,
            "near_dup_threshold": near_dup_threshold,
        },
    }
    with report_file.open("w") as fh:
        json.dump(payload, fh, indent=2)

    console.print(f"  Promoted pairs appended → [bold]{additions_file}[/bold]")
    console.print(f"  Report saved          → [bold]{report_file}[/bold]\n")


if __name__ == "__main__":
    app()

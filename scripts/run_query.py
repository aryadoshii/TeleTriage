"""
CLI demo.

Usage:
    uv run python scripts/run_query.py "high packet loss on sector 4"
    uv run python scripts/run_query.py "my calls keep dropping" --verbose

The --verbose flag prints the full tier trace so you can see exactly
which tier answered and with what confidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when running as a script (before uv install)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.observability import setup_logging
from backend.router import Router
from backend.types import Query

app = typer.Typer(help="TeleTriage CLI", add_completion=False)
console = Console()


@app.command()
def query(
    text: str = typer.Argument(..., help="The fault description to triage"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tier trace"),
) -> None:
    """Run a single query through the tier cascade."""
    setup_logging()
    router = Router()

    response = router.route(Query(text=text))

    # Pretty-print the answer
    console.print()
    console.print(
        Panel(
            response.answer,
            title=f"[bold green]Answer[/] (from {response.answered_by.value} tier)",
            border_style="green",
        )
    )

    # Quick stats line
    console.print(
        f"[dim]Confidence: {response.confidence:.2f}  "
        f"Latency: {response.total_latency_sec * 1000:.1f}ms[/]"
    )

    if verbose:
        _print_trace(response.tier_trace)


def _print_trace(trace: list) -> None:
    console.print()
    table = Table(title="Tier Trace", show_lines=False)
    table.add_column("Tier", style="cyan")
    table.add_column("Answered", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Details")

    for r in trace:
        table.add_row(
            r.tier.value,
            "✓" if r.answered else "→ delegate",
            f"{r.confidence:.2f}",
            f"{r.latency_sec * 1000:.1f}",
            str(r.details),
        )
    console.print(table)


if __name__ == "__main__":
    app()

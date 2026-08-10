"""
Installed entrypoint: `teletriage` command (defined in pyproject.toml scripts).

Usage after `uv sync`:
    teletriage query "high packet loss"
    teletriage query "dropped calls" --verbose
    teletriage info
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from backend.config import get_config
from backend.observability import setup_logging
from backend.router import Router
from backend.tiers.retrieval_tier import load_index_manifest
from backend.types import Query

app = typer.Typer(help="TeleTriage: hierarchical telecom fault triage", add_completion=False)
console = Console()


@app.command()
def query(
    text: str = typer.Argument(..., help="The fault description to triage"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a query through the tier cascade."""
    setup_logging()
    router = Router()
    response = router.route(Query(text=text))

    console.rule(f"[bold]{response.answered_by.value.upper()} tier[/]")
    console.print(response.answer)
    console.rule()
    console.print(
        f"[dim]confidence={response.confidence:.2f}  "
        f"latency={response.total_latency_sec * 1000:.1f}ms[/]"
    )

    if verbose:
        table = Table(title="Tier trace")
        table.add_column("Tier")
        table.add_column("Answered")
        table.add_column("Conf")
        table.add_column("Latency (ms)")
        for r in response.tier_trace:
            table.add_row(
                r.tier.value,
                "✓" if r.answered else "→",
                f"{r.confidence:.2f}",
                f"{r.latency_sec * 1000:.1f}",
            )
        console.print(table)


@app.command()
def info() -> None:
    """Show current config."""
    cfg = get_config()
    console.print(f"[bold]App:[/] {cfg.app.name} v{cfg.app.version}")
    console.print(f"[bold]Cache enabled:[/] {cfg.cache_tier.enabled} "
                  f"(threshold={cfg.cache_tier.fuzzy_threshold}, "
                  f"min_conf={cfg.cache_tier.min_confidence})")
    console.print(f"[bold]Retrieval enabled:[/] {cfg.retrieval_tier.enabled} "
                  f"(embedder={cfg.retrieval_tier.embedder_model})")

    manifest = load_index_manifest(cfg.resolve_path(cfg.paths.faiss_index_dir))
    configured_kb = cfg.resolve_path(cfg.paths.knowledge_base)
    if manifest is None:
        console.print(
            "[yellow]  Indexes:[/] no manifest.json found — either indexes "
            "aren't built yet, or they predate this feature. Run: "
            "[bold]uv run python scripts/build_index.py[/bold]"
        )
    else:
        console.print(
            f"[bold]  Indexes built from:[/] {manifest.get('kb_filename', '?')} "
            f"({manifest.get('n_docs', '?')} docs) "
            f"at {manifest.get('built_at', '?')}"
        )
        if manifest.get("kb_filename") != configured_kb.name:
            console.print(
                f"[yellow]  ⚠ Configured KB is '{configured_kb.name}' but indexes "
                f"were built from '{manifest.get('kb_filename')}' — stale. Run: "
                f"[bold]uv run python scripts/build_index.py[/bold][/yellow]"
            )

    console.print(f"[bold]Generative enabled:[/] {cfg.generative_tier.enabled} "
                  f"(backend={cfg.generative_tier.backend}, "
                  f"model={cfg.generative_tier.groq_model})")
    console.print(f"[bold]Groq key set:[/] {'yes' if cfg.secrets.groq_api_key else 'no'}")
    console.print(f"[bold]Gemini key set:[/] {'yes' if cfg.secrets.google_api_key else 'no'}")


if __name__ == "__main__":
    app()

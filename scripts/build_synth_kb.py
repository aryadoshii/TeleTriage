"""
Build database/synth_kb.jsonl from database/real_cache_candidates.jsonl —
the missing link between scripts/synth_qa.py's output and
scripts/build_index.py's expected KB schema.

scripts/synth_qa.py writes candidate rows shaped for human review and
auditing:
    {question, answer, answer_in_chunk, grounding_score, utility_score,
     reject_reasons, source_chunk_id, source, ...}

scripts/build_index.py expects KB rows shaped for indexing (see
backend/retrieval/bm25.py and dense.py, both index `question + " " +
answer` and use `id` as doc_id):
    {id, question, answer, tags, source}

This script filters candidates through both quality gates —
grounding_score and utility_score, see scripts/synth_qa.py's
_grounding_score() / _utility_score() — plus the answer_in_chunk flag set
at generation time, reshapes survivors into the KB schema, and assigns
sequential synth_NNNNN ids.

Usage:
    uv run python scripts/build_synth_kb.py
    uv run python scripts/build_synth_kb.py --min-grounding 0.70 --min-utility 0.70
    uv run python scripts/build_synth_kb.py \
        --input database/real_cache_candidates.jsonl --out database/synth_kb.jsonl

Then build indexes from the result:
    uv run python scripts/build_index.py --kb database/synth_kb.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

app = typer.Typer(add_completion=False)
console = Console()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_spec(source: str) -> str:
    """'3GPP TS 36.300' -> '36.300'. Falls back to the raw string for
    anything that doesn't match the expected '<org> TS <N.NNN>' shape,
    so an odd/missing source never crashes the summary, just shows as-is."""
    prefix = "3GPP TS "
    if source.startswith(prefix):
        return source[len(prefix):]
    return source or "(unknown)"


@app.command()
def main(
    input: Path = typer.Option(
        Path("database/real_cache_candidates.jsonl"), "--input", "-i",
        help="Candidates JSONL written by scripts/synth_qa.py",
    ),
    out: Path = typer.Option(
        Path("database/synth_kb.jsonl"), "--out", "-o",
        help="KB JSONL in the schema scripts/build_index.py expects",
    ),
    min_grounding: float = typer.Option(
        0.60, "--min-grounding",
        help="Minimum grounding_score to keep a candidate",
    ),
    min_utility: float = typer.Option(
        0.60, "--min-utility",
        help="Minimum utility_score to keep a candidate (the independent second gate)",
    ),
) -> None:
    """Filter synth_qa.py candidates through both quality gates and write a KB build_index.py can index."""
    if not input.exists():
        console.print(f"[red]Input file not found: {input}[/red]")
        console.print("Run [bold]uv run python scripts/synth_qa.py[/bold] first.")
        raise typer.Exit(1)

    console.print(f"\n[bold]TeleTriage — Build Synthetic KB[/bold]")
    console.print(f"  Input:            {input}")
    console.print(f"  Output:           {out}")
    console.print(f"  Min grounding:    {min_grounding}")
    console.print(f"  Min utility:      {min_utility}\n")

    candidates = _load_jsonl(input)

    n_total = len(candidates)
    # Fail-fast bucketing, same convention as CachePromoter.promote_from_file:
    # each rejected row counts against exactly the FIRST gate it fails, so
    # the three counts below sum to (n_total - n_kept) with no double-counting.
    n_rejected_ungrounded = 0   # answer_in_chunk is false
    n_rejected_grounding = 0    # grounding_score below --min-grounding
    n_rejected_utility = 0      # utility_score below --min-utility
    kept: list[dict] = []
    spec_counts: dict[str, int] = {}

    for row in candidates:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        source = row.get("source", "")
        grounding_score = float(row.get("grounding_score", 0.0))
        utility_score = float(row.get("utility_score", 0.0))

        # Default False (not True): unlike CachePromoter's cache-additions
        # use case, this script feeds the retrieval KB — an entry with no
        # recorded grounding verdict should not silently enter the corpus.
        if not row.get("answer_in_chunk", False):
            n_rejected_ungrounded += 1
            continue
        if grounding_score < min_grounding:
            n_rejected_grounding += 1
            continue
        if utility_score < min_utility:
            n_rejected_utility += 1
            continue

        kept.append({
            "id": f"synth_{len(kept) + 1:05d}",
            "question": question,
            "answer": answer,
            "tags": [],
            "source": source,
            # Extra fields beyond build_index.py's required schema — only
            # id/question/answer are read at index-build time, so these
            # ride along harmlessly and let a KB entry be traced back to
            # the chunk and scores that produced it.
            "source_chunk_id": row.get("source_chunk_id", ""),
            "grounding_score": grounding_score,
            "utility_score": utility_score,
        })
        spec = _extract_spec(source)
        spec_counts[spec] = spec_counts.get(spec, 0) + 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ─── Summary table ──────────────────────────────────────────────────────
    n_kept = len(kept)

    def pct(n: int) -> str:
        return f"{100 * n / max(n_total, 1):.0f}%"

    table = Table(title="Build Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Count", justify="right")
    table.add_column("Pct", justify="right")
    table.add_row("Candidates read", str(n_total), "100%")
    table.add_row(
        "[green]Kept[/green]", f"[green]{n_kept}[/green]", f"[green]{pct(n_kept)}[/green]"
    )
    table.add_row(
        "Rejected — not grounded (answer_in_chunk)",
        str(n_rejected_ungrounded), pct(n_rejected_ungrounded),
    )
    table.add_row(
        f"Rejected — grounding_score < {min_grounding}",
        str(n_rejected_grounding), pct(n_rejected_grounding),
    )
    table.add_row(
        f"Rejected — utility_score < {min_utility}",
        str(n_rejected_utility), pct(n_rejected_utility),
    )
    console.print(table)

    # ─── Spec distribution (of KEPT rows) ────────────────────────────────────
    if spec_counts:
        spec_table = Table(
            title="Kept — Distribution by Source Spec", show_header=True, header_style="bold"
        )
        spec_table.add_column("Spec", style="cyan")
        spec_table.add_column("Count", justify="right")
        spec_table.add_column("Pct of kept", justify="right")
        for spec, count in sorted(spec_counts.items(), key=lambda kv: -kv[1]):
            spec_table.add_row(spec, str(count), f"{100 * count / n_kept:.0f}%")
        console.print(spec_table)
    else:
        console.print("[yellow]  No candidates survived both gates — synth_kb.jsonl is empty.[/yellow]")

    console.print(f"\n  Saved {n_kept} KB entries → [bold]{out}[/bold]")
    console.print(
        f"  [dim]Build indexes from this KB:[/dim]  "
        f"uv run python scripts/build_index.py --kb {out}\n"
    )


if __name__ == "__main__":
    app()

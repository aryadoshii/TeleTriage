"""
Interactive CLI for building human relevance judgements.

For each query in backend/evaluation/test_set.py's EVAL_SET, runs the
current retrieval tier's full BM25 -> dense -> RRF -> rerank pipeline
(via RetrievalTier.debug_search(), which returns top-k reranked hits
without answer()'s confidence gate or its hardcoded top_k=3 cap), shows
a human reviewer the top-10 candidates with their question/answer text,
and asks a y/n relevance judgement for each. Judgements are written to a
JSON file after EVERY query (not just at the end), so an interrupted
session (Ctrl-C, closed terminal) never loses more than the current
query's in-progress judgements.

Resumable: re-running this script skips candidates already judged in a
previous session for the same query (matched by doc_id) — pass --relabel
to re-ask everything from scratch instead.

This script ASKS; it never guesses or fabricates. No judgement is
written unless a human reviewer explicitly answered y or n for that
specific candidate.

Usage:
    uv run python scripts/label_relevance.py
    uv run python scripts/label_relevance.py --query-id eval_008
    uv run python scripts/label_relevance.py --top-k 5
    uv run python scripts/label_relevance.py --relabel
    uv run python scripts/label_relevance.py --out outputs/my_judgements.json

Prerequisites:
    Build retrieval indexes: uv run python scripts/build_index.py

Output format (outputs/relevance_judgements.json by default):
    {
      "eval_008": {
        "relevant": ["synth_00050"],
        "judged":   ["synth_00050", "synth_00049", "synth_00136", ...]
      },
      ...
    }

"relevant" is exactly the {query_id: [doc_ids]} shape
backend/evaluation/relevance.py's RELEVANCE_JUDGEMENTS expects.
"judged" additionally tracks every candidate id a reviewer has already
answered y/n for (relevant or not) — that's what makes resuming
possible: a "no" is real signal too and must not be re-asked on the next
run. Once labelling is done, either point scripts/eval_retrieval.py
directly at this file (the default), or copy the "relevant" lists into
RELEVANCE_JUDGEMENTS to commit them as the versioned canonical set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

from backend.evaluation.test_set import get_eval_set
from backend.tiers.retrieval_tier import RetrievalTier

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_JUDGEMENTS_FILE = Path("outputs/relevance_judgements.json")


def _load_judgements(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load the judgements JSON, or {} if it doesn't exist yet."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_judgements(judgements: dict[str, dict[str, list[str]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(judgements, indent=2, sort_keys=True))


@app.command()
def main(
    out: Path = typer.Option(
        DEFAULT_JUDGEMENTS_FILE, "--out", "-o",
        help="JSON file to load existing judgements from and save new ones to",
    ),
    query_id: str = typer.Option(
        None, "--query-id",
        help="Label only this eval query id (e.g. eval_008). Default: all of them.",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Candidates shown per query"),
    relabel: bool = typer.Option(
        False, "--relabel",
        help="Re-ask every candidate even if already judged in a previous session",
    ),
    answer_chars: int = typer.Option(
        400, "--answer-chars", help="Characters of each candidate's answer to display",
    ),
) -> None:
    """Interactively label which retrieved candidates are actually relevant."""
    items = get_eval_set()
    if query_id:
        items = [i for i in items if i.id == query_id]
        if not items:
            console.print(f"[red]No eval item with id '{query_id}'.[/red]")
            raise typer.Exit(1)

    judgements = _load_judgements(out)

    console.print(f"\n[bold]TeleTriage — Relevance Labelling[/bold]")
    console.print(f"  Queries to review: {len(items)}")
    console.print(f"  Candidates/query:  {top_k}")
    console.print(f"  Judgements file:   {out}")
    console.print(f"  Existing progress: {len(judgements)} query(ies) with saved judgements\n")

    console.print("Loading retrieval tier (embedder + reranker)…")
    try:
        tier = RetrievalTier()
    except FileNotFoundError as exc:
        console.print(f"[red]Could not load retrieval indexes: {exc}[/red]")
        console.print("Run [bold]uv run python scripts/build_index.py[/bold] first.")
        raise typer.Exit(1)

    n_queries_touched = 0
    n_candidates_judged = 0

    try:
        for item in items:
            console.rule(f"[bold]{item.id}[/bold]")
            console.print(f"[dim]Query:[/dim] {item.query}\n")

            hits = tier.debug_search(item.query, top_k=top_k)
            if not hits:
                console.print("[yellow]  No candidates retrieved for this query — skipping.[/yellow]\n")
                continue

            q_judgements = judgements.setdefault(item.id, {"relevant": [], "judged": []})
            already_judged = set(q_judgements["judged"])

            any_shown = False
            for rank, hit in enumerate(hits, start=1):
                if hit.doc_id in already_judged and not relabel:
                    continue
                any_shown = True

                console.print(
                    f"[bold]#{rank}[/bold]  doc_id=[cyan]{hit.doc_id}[/cyan]  "
                    f"logit=[magenta]{hit.score:.3f}[/magenta]"
                )
                console.print(f"  Q: {hit.doc.get('question', '')}")
                answer = hit.doc.get("answer", "")
                if len(answer) > answer_chars:
                    answer = answer[:answer_chars] + "…"
                console.print(f"  A: {answer}")

                is_relevant = Confirm.ask("  Relevant?", default=False)

                if is_relevant and hit.doc_id not in q_judgements["relevant"]:
                    q_judgements["relevant"].append(hit.doc_id)
                elif not is_relevant and hit.doc_id in q_judgements["relevant"]:
                    q_judgements["relevant"].remove(hit.doc_id)  # relabelled y -> n
                if hit.doc_id not in q_judgements["judged"]:
                    q_judgements["judged"].append(hit.doc_id)

                n_candidates_judged += 1
                console.print()

            if any_shown:
                n_queries_touched += 1
                _save_judgements(judgements, out)  # save after every query, not just at exit
                console.print(
                    f"[dim]Saved. {len(q_judgements['relevant'])} relevant / "
                    f"{len(q_judgements['judged'])} judged so far for {item.id}.[/dim]\n"
                )
            else:
                console.print(f"[dim]All {len(hits)} candidates already judged — nothing new.[/dim]\n")

    except KeyboardInterrupt:
        _save_judgements(judgements, out)
        console.print("\n[yellow]Interrupted — progress saved.[/yellow]")
        raise typer.Exit(0)

    console.print(
        f"\n[bold]Done.[/bold] {n_queries_touched} query(ies) touched this session, "
        f"{n_candidates_judged} new candidate judgement(s)."
    )
    console.print(f"Saved → [bold]{out}[/bold]")
    console.print(
        f"\n[dim]Run the ablation:[/dim]  "
        f"uv run python scripts/eval_retrieval.py --judgements {out}\n"
    )


if __name__ == "__main__":
    app()

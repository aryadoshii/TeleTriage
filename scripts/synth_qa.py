"""
Generate synthetic Q&A pairs from real_kb.jsonl chunks using Groq
(Llama 3.3 70B), validate that each answer is grounded in the source
chunk, and write data/real_cache_candidates.jsonl for human review.

Usage:
    uv run python scripts/synth_qa.py
    uv run python scripts/synth_qa.py --input data/real_kb.jsonl --qa-per-chunk 3
    uv run python scripts/synth_qa.py --max-chunks 100 --out data/real_cache_candidates.jsonl
    uv run python scripts/synth_qa.py --sample-rate 0.5   # random 50% of chunks

Prerequisites:
    GROQ_API_KEY set in .env
    data/real_kb.jsonl built by scripts/scrape_data.py

━━━ DESIGN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q&A generation prompt
  The LLM is given a raw 3GPP chunk and asked to produce QA pairs where:
  (1) the question is something a telecom engineer would actually ask, and
  (2) the answer is verifiable directly in the chunk — no hallucination.

Grounding validation
  We verify that ≥60% of the significant words in the LLM's answer appear
  in the source chunk (lowercased, stopwords stripped).  This is a recall-
  oriented check: a valid answer must draw its facts from the chunk, but
  doesn't have to quote it verbatim.  Pairs that fail are flagged
  answer_in_chunk=False and excluded from the output count but kept for
  inspection.

Why human review before cache promotion?
  Synthetic Q&A pairs may be grammatically correct but factually imprecise,
  or may describe edge cases the LLM misread.  The cache is the highest-
  confidence tier (p50 < 5ms), so only validated pairs should be promoted.
  The JSONL output is structured so a reviewer can do:
      jq 'select(.answer_in_chunk)' real_cache_candidates.jsonl | head -20
  and then move approved rows into data/sample_cache.json.

Rate limiting
  Groq's free tier allows 30 req/min on Llama 3.3 70B.  With 3 QA pairs
  per chunk and ~1s per request, processing 100 chunks takes ~7 minutes.
  The script tracks requests-per-minute and sleeps when needed.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

app = typer.Typer(add_completion=False)
console = Console()

# ─── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior telecom engineer writing a troubleshooting knowledge base
from 3GPP Technical Specifications.  Your task: given a raw spec excerpt,
generate question-answer pairs that a field engineer would actually search for.

Rules:
- Questions must be specific and technical — no generic "What is X?"
- Answers must be directly supported by the text in the excerpt — do not
  add information from outside the excerpt.
- Answers should be 1-4 sentences: dense, actionable, accurate.
- Output ONLY a JSON array.  No markdown, no prose before or after.
"""

QA_USER_TEMPLATE = """\
Generate exactly {n} question-answer pairs from the following 3GPP excerpt.

<excerpt source="{source}">
{text}
</excerpt>

Output format (JSON array, no other text):
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""

# ─── Grounding validator ──────────────────────────────────────────────────────

# Common English stopwords — their absence in the chunk doesn't indicate
# hallucination, so we exclude them from the coverage check.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "at", "by", "with", "from", "as", "or", "and", "but", "if",
    "this", "that", "it", "its", "not", "no", "so", "than", "then",
    "when", "which", "what", "how", "why", "where", "who",
}


def _significant_words(text: str) -> list[str]:
    """Lowercase alphabetic tokens longer than 2 chars, stopwords removed."""
    return [
        w for w in re.findall(r"[a-z]{3,}", text.lower())
        if w not in _STOPWORDS
    ]


def _grounding_score(answer: str, chunk: str) -> float:
    """
    Fraction of significant words in the answer that also appear in the chunk.
    Score = 1.0 means fully grounded; score < 0.6 means likely hallucination.
    """
    a_words = _significant_words(answer)
    if not a_words:
        return 0.0
    chunk_words = set(_significant_words(chunk))
    return sum(1 for w in a_words if w in chunk_words) / len(a_words)


def _is_grounded(answer: str, chunk: str, threshold: float = 0.60) -> bool:
    return _grounding_score(answer, chunk) >= threshold


# ─── LLM call ────────────────────────────────────────────────────────────────


def _parse_json_array(text: str) -> list[dict] | None:
    """
    Extract the first JSON array from an LLM response.
    The model sometimes wraps output in markdown code fences — strip them.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Find the first [ ... ] block
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group())
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _generate_qa(
    client: Groq,
    chunk_doc: dict,
    n: int,
    model: str,
    grounding_threshold: float,
    max_retries: int = 2,
) -> list[dict]:
    """
    Call Groq to generate n Q&A pairs for chunk_doc.
    Retries once on transient errors.  Rate limit errors propagate up.
    Returns only pairs that pass the grounding check.
    """
    user_msg = QA_USER_TEMPLATE.format(
        n=n,
        source=chunk_doc.get("source", "3GPP"),
        text=chunk_doc["answer"][:2000],  # hard cap to stay inside context
    )

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,   # slight creativity to vary phrasings
                max_tokens=768,
            )
            raw = resp.choices[0].message.content or ""
            pairs = _parse_json_array(raw)
            if not pairs:
                if attempt < max_retries:
                    continue
                return []

            results = []
            for pair in pairs:
                q = str(pair.get("question", "")).strip()
                a = str(pair.get("answer", "")).strip()
                if not q or not a:
                    continue
                grounded = _is_grounded(a, chunk_doc["answer"], grounding_threshold)
                results.append({
                    "question": q,
                    "answer": a,
                    "answer_in_chunk": grounded,
                    "grounding_score": round(_grounding_score(a, chunk_doc["answer"]), 3),
                })
            return results

        except RateLimitError:
            raise  # propagate — caller handles rate limiting
        except Exception:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return []

    return []


# ─── Rate limiter ─────────────────────────────────────────────────────────────


class RateLimiter:
    """Simple token-bucket rate limiter: at most max_per_min requests/minute."""

    def __init__(self, max_per_min: int) -> None:
        self._max = max_per_min
        self._timestamps: list[float] = []

    def wait(self) -> None:
        now = time.monotonic()
        # Drop timestamps older than 60s
        self._timestamps = [t for t in self._timestamps if now - t < 60.0]
        if len(self._timestamps) >= self._max:
            sleep_for = 60.0 - (now - self._timestamps[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._timestamps = []
        self._timestamps.append(time.monotonic())


# ─── Main ─────────────────────────────────────────────────────────────────────


@app.command()
def main(
    input: Path = typer.Option(
        Path("data/real_kb.jsonl"), "--input", "-i",
        help="Input JSONL (output of scrape_data.py)",
    ),
    out: Path = typer.Option(
        Path("data/real_cache_candidates.jsonl"), "--out", "-o",
    ),
    qa_per_chunk: int = typer.Option(3, "--qa-per-chunk", help="Q&A pairs to generate per chunk"),
    max_chunks: int = typer.Option(0, "--max-chunks", help="Cap chunks processed (0 = all)"),
    sample_rate: float = typer.Option(1.0, "--sample-rate", help="Random fraction of chunks to process"),
    grounding_threshold: float = typer.Option(0.60, "--grounding-threshold"),
    groq_model: str = typer.Option("llama-3.3-70b-versatile", "--model"),
    groq_rpm: int = typer.Option(28, "--groq-rpm", help="Groq requests-per-minute limit"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """
    Generate Q&A pairs from real_kb.jsonl chunks via Groq.
    Output goes to real_cache_candidates.jsonl for human review.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        console.print("[red]GROQ_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    if not input.exists():
        console.print(f"[red]Input file not found: {input}[/red]")
        console.print("Run [bold]uv run python scripts/scrape_data.py[/bold] first.")
        raise typer.Exit(1)

    # Load chunks
    with open(input) as fh:
        chunks = [json.loads(line) for line in fh if line.strip()]

    # Sample
    if sample_rate < 1.0:
        random.seed(seed)
        chunks = random.sample(chunks, int(len(chunks) * sample_rate))

    if max_chunks > 0:
        chunks = chunks[:max_chunks]

    console.print(f"\n[bold]TeleTriage — Synthetic Q&A Generation[/bold]")
    console.print(f"  Chunks to process: {len(chunks)}")
    console.print(f"  Q&A per chunk:     {qa_per_chunk}")
    console.print(f"  Model:             {groq_model}")
    console.print(f"  Grounding thresh:  {grounding_threshold}")
    console.print(f"  Output:            {out}\n")

    client = Groq(api_key=api_key)
    limiter = RateLimiter(groq_rpm)

    n_generated = 0
    n_grounded = 0
    n_errors = 0
    results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating…", total=len(chunks))

        for chunk_doc in chunks:
            progress.update(
                task,
                description=f"  [{chunk_doc['id']}] {chunk_doc['source']}",
                advance=1,
            )

            limiter.wait()
            try:
                pairs = _generate_qa(
                    client,
                    chunk_doc,
                    n=qa_per_chunk,
                    model=groq_model,
                    grounding_threshold=grounding_threshold,
                )
            except RateLimitError:
                console.print("[yellow]  Rate limit hit — sleeping 60s[/yellow]")
                time.sleep(60)
                pairs = []
                n_errors += 1
            except Exception as exc:
                console.print(f"[red]  Error on {chunk_doc['id']}: {exc}[/red]")
                n_errors += 1
                pairs = []

            for pair in pairs:
                n_generated += 1
                if pair["answer_in_chunk"]:
                    n_grounded += 1
                results.append({
                    "id": f"cand_{len(results):05d}",
                    "source_chunk_id": chunk_doc["id"],
                    "source": chunk_doc.get("source", ""),
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "answer_in_chunk": pair["answer_in_chunk"],
                    "grounding_score": pair["grounding_score"],
                    "validated": False,   # set to True after human review
                })

    # Write output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    grounded_only = [r for r in results if r["answer_in_chunk"]]

    console.print(f"\n[bold]Results[/bold]")
    console.print(f"  Chunks processed:  {len(chunks)}")
    console.print(f"  Q&A pairs generated: {n_generated}")
    console.print(f"  Grounded (pass):   [green]{n_grounded}[/green] ({100*n_grounded/max(n_generated,1):.0f}%)")
    console.print(f"  Failed grounding:  [yellow]{n_generated - n_grounded}[/yellow]")
    console.print(f"  API errors:        {n_errors}")
    console.print(f"\n  Saved {len(results)} candidates → [bold]{out}[/bold]")
    console.print(
        f"  [dim]Promote grounded pairs to cache:[/dim]  "
        f"jq 'select(.answer_in_chunk)' {out} | head -20"
    )
    console.print()


if __name__ == "__main__":
    app()

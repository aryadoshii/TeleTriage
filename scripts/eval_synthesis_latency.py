"""
Measure tier 2 (RAG synthesis) latency across synthesis_top_k x
synthesis_chunk_chars configurations. Does NOT decide which is best.

Problem: tier 2 was observed live at ~1930ms — SLOWER than tier 3's
~1600ms, inverting the cascade's cost argument (tier 2 is supposed to be
the cheap path). Prime suspect: context size — synthesis_top_k=5 chunks
at ~2048 chars each is ~10K chars in every synthesis prompt, on every
retrieval hit (the common case, not the exception).

FIRST VERSION OF THIS SCRIPT WAS ITSELF BROKEN: it ran 27 back-to-back
calls with 1s pacing and measured mostly rate-limit noise, not prompt-
size effects. Root-caused with direct HTTP debug logging against Groq:
llama-3.1-8b-instant's free-tier budget on this account is
x-ratelimit-limit-tokens=6000 (6000 TOKENS PER MINUTE) — confirmed via
Groq's own response headers. Each synthesis call is ~700-2700 tokens, so
only ~2-3 fit per rolling 60s window before throttling. Worse: the groq
SDK's own HTTP transport defaults to max_retries=2 and silently retries
429s using the server's Retry-After header — invisible to this project's
own retry logic, inflating latency 20-50x with zero signal anywhere that
a retry happened. That SDK-level retry is now disabled (see
backend/generation/llm_client.py's GroqClient — "CONFIRMED BUG"), so any
retry that happens now is OUR retry loop: logged, and counted in
GenerationOutput.retry_count / TierResult.details["synthesis_retry_count"].

This version fixes the MEASUREMENT, not just the pacing number:
  1. --pacing-sec is a CLI flag (default 5.0 — conservative starting
     point; increase if the report below shows throttling still
     dominates at this value. Do NOT assume 5.0 is sufficient: 6000 TPM /
     ~2000 tokens-per-call average is only ~3 calls per 60s window, i.e.
     one call every ~20s in the worst case (synthesis_top_k=5, no
     truncation) — 5s pacing is expected to still throttle for the
     larger configs. This script will show you, not guess for you.)
  2. Every result row logs raw latency_ms AND retry_count directly from
     GenerationOutput — a retried call is now IDENTIFIABLE, not inferred
     from suspicious timing.
  3. Call order is randomized across ALL (query, top_k, chunk_chars)
     combinations (not nested loops always running top_k=5 first) so a
     systematic "later configs benefit from an elapsed rate-limit window"
     bias can't masquerade as a "smaller context is faster" effect.
  4. A warm-up call (discarded from the table) runs first, so cold-start
     connection/model overhead doesn't get conflated with throttling.

SECOND VERSION WAS ALSO BROKEN, differently: it used tier._synthesis_client
(the full FallbackClient, Groq -> Gemini -> local Qwen). At 5s pacing, Groq
throttled by call 5 exactly as predicted above (confirmed: "Limit 6000,
Used 3735, Requested 2582" from the live error) and, because the SDK-level
retry fix above makes that fail FAST now instead of hanging, FallbackClient
correctly moved to Gemini immediately — which returned a 404: the
configured model "gemini-2.0-flash" no longer exists ("no longer
available... use a newer model"). That's a real, separate bug (see
config/config.yaml's generative_tier.gemini_model / retrieval_tier's
gemini_model reuse) independent of this investigation, but it meant every
throttled cell cascaded through a dead Gemini straight to local Qwen —
slow (CPU inference, first-token latency alone can be 10s+) and, at least
once, slow enough to exceed the harness's own subprocess timeout and get
killed mid-run with zero results saved. This script now talks to GroqClient
DIRECTLY, bypassing the fallback chain entirely: the point here is
measuring Groq's OWN latency behaviour, and a fast, clean, logged failure
is more useful for that than an accidental tour through a broken fallback
chain. (The Gemini bug is reported separately — it needs its own fix, not
a workaround here.)

Usage:
    uv run python scripts/eval_synthesis_latency.py
    uv run python scripts/eval_synthesis_latency.py --pacing-sec 20
    uv run python scripts/eval_synthesis_latency.py --out outputs/synthesis_latency.json

Prerequisites:
    Build retrieval indexes: uv run python scripts/build_index.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

from backend.config import get_config
from backend.generation.llm_client import GroqClient
from backend.tiers.retrieval_tier import (
    SYNTHESIS_SYSTEM_PROMPT,
    RetrievalTier,
    _build_synthesis_prompt,
)

app = typer.Typer(add_completion=False)
console = Console()

# Same params tier 2 actually uses in production — see
# backend/tiers/retrieval_tier.py's _SYNTHESIS_MAX_TOKENS/_SYNTHESIS_TEMPERATURE.
_MAX_TOKENS = 512
_TEMPERATURE = 0.2

QUERIES = [
    "UE fails RRC re-establishment after handover",
    "MME rejects TAU request with EMM cause 11",
    "eNB not generating handover decision despite UE sending A3 measurement reports",
]

TOP_K_VALUES = [5, 3, 2]
CHUNK_CHARS_VALUES = [None, 1500, 1000]

_SEED = 42  # reproducible shuffle order


@app.command()
def main(
    out: Path = typer.Option(
        Path("outputs/synthesis_latency.json"), "--out", "-o",
        help="Where to save full results (latency + complete answers) as JSON",
    ),
    pacing_sec: float = typer.Option(
        5.0, "--pacing-sec",
        help="Seconds between real API calls. Conservative starting point — "
             "llama-3.1-8b-instant's measured free-tier budget is 6000 tokens/min, "
             "so larger configs (~2000+ tokens/call) may still throttle at 5s. "
             "The report below will show you if that happened; it will NOT "
             "silently retry with a longer pacing on its own.",
    ),
) -> None:
    """Measure synthesis latency across top_k x chunk_chars configs, in randomized order. Reports a table; picks nothing."""
    total_calls = len(QUERIES) * len(TOP_K_VALUES) * len(CHUNK_CHARS_VALUES)
    console.print("\n[bold]TeleTriage — Synthesis Latency Measurement[/bold]")
    console.print(f"  Queries:      {len(QUERIES)}")
    console.print(f"  top_k values: {TOP_K_VALUES}")
    console.print(f"  chunk_chars:  {[c if c is not None else 'none' for c in CHUNK_CHARS_VALUES]}")
    console.print(f"  Pacing:       {pacing_sec}s between calls")
    console.print(f"  Total calls:  {total_calls} + 1 warm-up (discarded), randomized order\n")

    console.print("Loading retrieval tier (embedder + reranker + synthesis client)…")
    tier = RetrievalTier()

    # Talk to Groq DIRECTLY — not tier._synthesis_client (the full
    # FallbackClient: Groq -> Gemini -> local Qwen). See module docstring:
    # the fallback chain's Gemini leg is currently broken (404, deprecated
    # model), so at this pacing every throttled call would otherwise
    # cascade through a guaranteed-dead Gemini into a slow local model.
    # We're measuring Groq's OWN latency profile here; a fast, logged
    # RateLimitError is a clean, informative result for that purpose —
    # a multi-second detour through a broken fallback chain is not.
    cfg = get_config()
    groq_client = GroqClient(
        api_key=cfg.secrets.groq_api_key,
        model=cfg.retrieval_tier.synthesis_model,
    )

    # ─── Retrieval runs ONCE per query up front — fixed cost, not what's ──
    # under test. max(TOP_K_VALUES) so there are enough reranked chunks to
    # slice down to every top_k value below.
    console.print("Running retrieval for each query (once, not repeated per config)…")
    chunks_by_query: dict[str, list] = {}
    for query_text in QUERIES:
        reranked = tier.debug_search(query_text, top_k=max(TOP_K_VALUES))
        if reranked:
            chunks_by_query[query_text] = reranked
        else:
            console.print(f"[yellow]  No candidates retrieved for: {query_text!r} — skipping.[/yellow]")

    # ─── Build the full work-list, then shuffle it ─────────────────────────
    # Randomizing across ALL combinations (not nested top_k-then-chunk_chars
    # loops) decorrelates "config size" from "elapsed time since last
    # throttle" — see module docstring point 3.
    work_items = [
        (query_text, top_k, chunk_chars)
        for query_text in chunks_by_query
        for top_k in TOP_K_VALUES
        for chunk_chars in CHUNK_CHARS_VALUES
    ]
    rng = random.Random(_SEED)
    rng.shuffle(work_items)

    # ─── Warm-up: one throwaway call, discarded from all results/table ────
    console.print("\nWarm-up call (cold-start cost, discarded from results)…")
    warmup_query, warmup_top_k, warmup_chunk_chars = work_items[0]
    warmup_prompt = _build_synthesis_prompt(
        warmup_query, chunks_by_query[warmup_query][:warmup_top_k], chunk_chars=warmup_chunk_chars,
    )
    try:
        t0 = time.perf_counter()
        warmup_output = groq_client.generate(
            system=SYNTHESIS_SYSTEM_PROMPT, user=warmup_prompt,
            max_tokens=_MAX_TOKENS, temperature=_TEMPERATURE,
        )
        console.print(
            f"  warm-up: {(time.perf_counter() - t0) * 1000:.0f}ms, "
            f"backend={warmup_output.backend}, retry_count={warmup_output.retry_count} (discarded)\n"
        )
    except Exception as exc:
        console.print(f"[yellow]  warm-up call failed (continuing anyway): {exc}[/yellow]\n")
    time.sleep(pacing_sec)

    # ─── Real measurements, in shuffled order ──────────────────────────────
    # groq_client is used directly (no fallback chain), so a rate-limited
    # call raises BackendUnavailableError immediately (see GroqClient.generate:
    # RateLimitError is never retried, by design, so FallbackClient could
    # switch backends in production — here there's nothing to switch to, so
    # it's just recorded as a clean, fast "error" row below) instead of
    # succeeding via a slow, undesired fallback.
    results: list[dict] = []
    n_throttled = 0
    n_retried = 0

    for i, (query_text, top_k, chunk_chars) in enumerate(work_items, start=1):
        chunks = chunks_by_query[query_text][:top_k]
        user_prompt = _build_synthesis_prompt(query_text, chunks, chunk_chars=chunk_chars)
        prompt_chars = len(user_prompt)
        chunk_chars_label = chunk_chars if chunk_chars is not None else "none"

        try:
            t0 = time.perf_counter()
            output = groq_client.generate(
                system=SYNTHESIS_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            n_throttled += 1
            console.print(
                f"[red]  [{i}/{total_calls}] top_k={top_k} chunk_chars={chunk_chars_label}: "
                f"call failed outright: {exc}[/red]"
            )
            results.append({
                "query": query_text, "top_k": top_k,
                "chunk_chars": chunk_chars_label, "prompt_chars": prompt_chars,
                "latency_ms": None, "backend": None, "model": None,
                "tokens_used": None, "retry_count": None, "answer": None, "error": str(exc),
            })
            time.sleep(pacing_sec)
            continue

        retried = output.retry_count > 0
        n_retried += retried

        row = {
            "query": query_text,
            "top_k": top_k,
            "chunk_chars": chunk_chars_label,
            "prompt_chars": prompt_chars,
            "latency_ms": round(latency_ms, 1),
            "backend": output.backend,
            "model": output.model,
            "tokens_used": output.tokens_used,
            "retry_count": output.retry_count,
            "answer": output.text,
            "error": None,
        }
        results.append(row)

        flag_str = f"  [yellow]retried x{output.retry_count}[/yellow]" if retried else ""

        console.print(
            f"  [{i}/{total_calls}] top_k={top_k} chunk_chars={chunk_chars_label!s:>4}  "
            f"prompt_chars={prompt_chars:>6}  "
            f"latency={latency_ms:>7.1f}ms  "
            f"backend={output.backend}/{output.model}{flag_str}"
        )

        time.sleep(pacing_sec)

    # ─── Data-quality summary — read this before the latency table ────────
    console.print()
    if n_throttled:
        console.print(
            f"[red bold]WARNING: {n_throttled}/{len(results)} call(s) were rate-limited "
            f"outright at {pacing_sec}s pacing (recorded as error rows, excluded from the "
            f"latency table below). If this is a large fraction, throttling still "
            f"dominates at this pacing — re-run with a larger --pacing-sec.[/red bold]"
        )
    if n_retried:
        console.print(
            f"[yellow bold]NOTE: {n_retried}/{len(results)} call(s) needed at least one retry "
            f"(visible now via retry_count — previously silent). Their latency includes "
            f"real backoff wait time, not pure inference time. Still included below, "
            f"but flagged per-row above.[/yellow bold]"
        )
    if n_throttled == 0 and n_retried == 0:
        console.print(
            f"[green]All {len(results)} calls completed on the first attempt via Groq — "
            f"no throttling observed at {pacing_sec}s pacing.[/green]"
        )

    # ─── Summary table: avg latency + avg prompt size per config ──────────
    ok_results = [r for r in results if r["latency_ms"] is not None and r["backend"] == "groq"]
    table = Table(
        title=f"Synthesis Latency by Config (Groq calls only, pacing={pacing_sec}s)",
        show_header=True, header_style="bold",
    )
    table.add_column("top_k", justify="right")
    table.add_column("chunk_chars", justify="right")
    table.add_column("avg prompt chars", justify="right")
    table.add_column("avg latency (ms)", justify="right")
    table.add_column("retried", justify="right")
    table.add_column("n", justify="right")

    for top_k in TOP_K_VALUES:
        for chunk_chars in CHUNK_CHARS_VALUES:
            label = chunk_chars if chunk_chars is not None else "none"
            matching = [r for r in ok_results if r["top_k"] == top_k and r["chunk_chars"] == label]
            if not matching:
                continue
            avg_prompt = sum(r["prompt_chars"] for r in matching) / len(matching)
            avg_latency = sum(r["latency_ms"] for r in matching) / len(matching)
            n_retried_cell = sum(1 for r in matching if r["retry_count"] > 0)
            table.add_row(
                str(top_k), str(label), f"{avg_prompt:.0f}", f"{avg_latency:.1f}",
                f"{n_retried_cell}/{len(matching)}", str(len(matching)),
            )

    console.print()
    console.print(table)

    # ─── Full per-query answers, so quality is visible alongside speed ────
    console.print("\n[bold]Full per-query answers[/bold] (complete text in --out JSON)\n")
    for r in results:
        if r["answer"] is None:
            continue
        retry_note = f" retry_count={r['retry_count']}" if r["retry_count"] else ""
        console.print(
            f"[dim]{r['query'][:55]}  top_k={r['top_k']} chunk_chars={r['chunk_chars']}  "
            f"{r['latency_ms']:.0f}ms  backend={r['backend']}/{r['model']}{retry_note}[/dim]"
        )
        preview = r["answer"][:350] + ("…" if len(r["answer"]) > 350 else "")
        console.print(f"  {preview}\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    console.print(f"Full results (latency + retry_count + complete answer text) saved → [bold]{out}[/bold]\n")
    console.print(
        "[yellow]This script measures — it does not recommend a config. If throttling "
        "or fallback dominated this run, re-run with a larger --pacing-sec rather than "
        "trusting this table, and report back if it STILL doesn't clear up — that would "
        "be a viability finding about llama-3.1-8b-instant on free-tier Groq, not a "
        "pacing problem.[/yellow]\n"
    )


if __name__ == "__main__":
    app()

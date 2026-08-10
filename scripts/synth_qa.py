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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

app = typer.Typer(add_completion=False)
console = Console()


@dataclass
class CompletionResult:
    text: str
    tokens_used: int


class LLMBackend(Protocol):
    def complete(self, system: str, user: str) -> CompletionResult: ...


class GroqBackend:
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> CompletionResult:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text = resp.choices[0].message.content or ""
        # OpenAI-compatible usage object: resp.usage.total_tokens
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) or 0
        return CompletionResult(text=text, tokens_used=tokens)


class GeminiBackend:
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> CompletionResult:
        from google.genai import types as gtypes
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=gtypes.GenerateContentConfig(
                system_instruction=system,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            ),
        )
        text = response.text or ""
        # google-genai usage object: response.usage_metadata.total_token_count
        usage = getattr(response, "usage_metadata", None)
        tokens = getattr(usage, "total_token_count", 0) or 0
        return CompletionResult(text=text, tokens_used=tokens)

# ─── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior telecom engineer writing a troubleshooting knowledge base
from 3GPP Technical Specifications.  Your task: given a raw spec excerpt,
generate question-answer pairs that a field engineer would actually search for.

Rules:
- Questions must be specific and technical — no generic "What is X?"
- Frame every question as something an engineer debugging a LIVE NETWORK
  FAULT would type into a search box — a symptom, an error, a behaviour
  they don't understand. NOT a question about the document itself.
- NEVER ask about document metadata: release numbers, version numbers,
  section/clause numbering, where specifications are published or
  obtained, or who maintains the spec.
- NEVER produce an answer whose substance is a pointer to another
  section. "This is described in section 4.9.1" is not an answer — it
  doesn't say what actually happens. If the excerpt only cross-references
  other clauses without stating the actual behaviour, generate FEWER
  pairs for it. Returning 0 pairs for a chunk that has nothing answerable
  in it is a correct and expected outcome, not a failure — do not force
  pairs out of unusable material.
- Prefer questions about: failure conditions, timers and their expiry
  behaviour, cause codes and what they mean, state transitions, procedure
  preconditions, and what happens when something is absent, invalid, or
  times out.
- Answers must be directly supported by the text in the excerpt — do not
  add information from outside the excerpt.
- Answers should be 1-4 sentences: dense, actionable, accurate.
- Output ONLY a JSON array.  No markdown, no prose before or after.
"""

QA_USER_TEMPLATE = """\
Generate exactly {n} question-answer pairs from the following 3GPP excerpt.

Two examples (not from this excerpt) showing the difference between a
usable pair and one that looks plausible but isn't:

GOOD:
  {{"question": "MME rejects TAU request with EMM cause 11 for a subset of subscribers",
    "answer": "EMM cause 11 is 'PLMN not allowed' — the MME rejects the TAU because the UE's home PLMN is not present in the MME's allowed-PLMN list, or the HSS subscription does not permit service for that PLMN."}}
  Why it's good: a real symptom an engineer would search for, and the answer states the actual cause, not a section pointer.

BAD:
  {{"question": "What does EMM cause 11 mean?",
    "answer": "EMM cause values and their meanings are defined in clause 9.9.3.9 of this specification."}}
  Why it's bad: the question is about the document's own numbering scheme, and the answer just points at another clause instead of stating the actual meaning. Do not produce pairs like this.

<excerpt source="{source}">
{text}
</excerpt>

Output format (JSON array, no other text — if nothing in the excerpt is
answerable per the rules above, output an empty array []):
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


# ─── Utility validator (independent of grounding) ──────────────────────────────
#
# grounding_score answers "did the model copy real words from the source
# chunk" — it says NOTHING about whether the resulting pair is useful. A
# probe run proved this concretely: the two most useless pairs it produced
# — "What is the release number of the TSG RAN document?" and "Where can
# specifications for the 3GPP system be obtained?" — scored 0.90 and 1.00
# grounding, the HIGHEST in the batch, because their boilerplate answers
# were copied near-verbatim from the chunk. Grounding and usefulness are
# orthogonal. This is a SECOND, independent validator — deliberately not
# folded into grounding_score, so the (expected to be near-zero)
# correlation between the two is visible rather than averaged away.

_CROSSREF_RE = re.compile(
    r"(?:described|specified|defined|found)\s+in\s+(?:sub)?(?:clause|section)s?\s+\d+(?:\.\d+)*",
    re.IGNORECASE,
)
# NOTE: the brief specified r"...(?:sub)?clause|section\s+\d+\.\d+" — as
# written that '|' splits the WHOLE pattern in two (verb+in+clause, OR
# any bare "section N.N"), and neither branch handles the plural
# ("sections 4.6.3.1") that's actually in the corpus. Verified against
# the two real bad-answer examples in this bug report ("...defined in
# sections 4.6.3.1 and 4.6.3.2, respectively.", "...described in
# sections 4.9.1, 4.9.2, 4.9.3...") — the literal brief regex does NOT
# match either due to the plural. Fixed the grouping and added an
# optional trailing "s" so it actually catches the examples it's meant to
# catch; kept the same verbs/keywords.

_DOC_METADATA_RE = re.compile(
    r"\b(section|clause|release|version|specification|document|annex)\b",
    re.IGNORECASE,
)

# Small curated set of corpus vocabulary — presence of ANY of these in an
# answer (or question) is evidence the pair is actually about the network,
# not the document. Deliberately small and telecom-specific per the brief.
#
# Expanded after a real run showed a correctly-grounded pair
# ("UE receives UL grant on PCell in response to random access preamble
# on SCell of MCG, which cell is the grant for?" -> "The UL grant is for
# the SCell...") fail this check: PCell/SCell weren't in the original
# list. Added the explicitly requested LTE/EPC terms below, plus "utran"
# — the only term clearing a >5-answers document-frequency bar across
# database/synth_probe.jsonl + database/synth_probe2.jsonl (76 answers
# total) that wasn't already covered. (For reference, the next-most
# frequent uncovered candidate was "rat" at exactly 5 answers — doesn't
# clear ">5", correctly excluded.)
#
# "ca", "ra", "tb", "ri", "sr" were requested but are deliberately NOT
# included: verified they collide with ordinary English as bare whole-word
# tokens ("ca" = circa, "ra" = the deity Ra, "tb" = tuberculosis,
# "ri" = Rhode Island, "sr" = Senior — e.g. "Smith Sr wrote..." tokenizes
# to the exact word "sr"). This isn't a tokenizer bug — _has_domain_term's
# findall-based extraction already does whole-word matching (verified
# separately: "categorical", "subtracted", "self-organizing" etc. do NOT
# false-positive on "ca"/"ra" as substrings) — the ambiguity is real and
# whole-word regardless. Genuine technical usage is consistently ALL-CAPS
# ("SR", "CA") while every English collision found was title-case or
# lowercase ("Sr", "ca"), so a case-sensitive carve-out for just these 5
# terms would recover them — not implemented here, kept simple per the
# "drop the ones that collide" instruction. "ue" and "uu" were also
# checked and have no equivalent real-word collision, so both stayed in.
_DOMAIN_TERMS = {
    # original set
    "enb", "ue", "mme", "rrc", "s1ap", "nas", "eps", "bearer", "handover",
    "pdcp", "rlc", "mac", "tau", "attach", "timer", "cause", "harq", "qci",
    "teid", "pdn", "sgw", "pgw", "measurement", "reconfiguration",
    # explicitly requested (minus ca/ra/tb/ri/sr — see note above)
    "pcell", "scell", "pcc", "scc", "drb", "srb", "erab", "gtp",
    "x2", "s1", "uu", "rlf", "prach", "cqi", "pmi", "harq-ack",
    # frequency analysis: >5/76 answers in synth_probe.jsonl + synth_probe2.jsonl
    "utran",
}


def _crossref_char_fraction(answer: str) -> float:
    """Fraction of `answer`'s characters covered by a cross-reference span
    ("described in sections 4.9.1", "specified in clause 5.3", ...)."""
    if not answer:
        return 0.0
    covered = sum(m.end() - m.start() for m in _CROSSREF_RE.finditer(answer))
    return covered / len(answer)


def _has_domain_term(text: str) -> bool:
    # [a-z]+ alone would silently break several requested terms: digits
    # aren't letters, so "X2" -> "x2" tokenizes to just "x" (the "2" is
    # dropped, not merely separated), and "S1" -> just "s". Hyphenated
    # compounds like "HARQ-ACK" split into "harq"+"ack" under either
    # regex, so "harq-ack" as a literal set entry would never match on
    # its own (it's covered here by "harq" already being in the set).
    # [a-z0-9]+(?:-[a-z0-9]+)* keeps digit-suffixed tokens (x2, s1) and
    # hyphenated compounds (harq-ack) intact as single tokens.
    words = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()))
    return bool(words & _DOMAIN_TERMS)


@dataclass
class UtilityResult:
    score: float
    reject_reasons: list[str]


def _utility_score(question: str, answer: str) -> UtilityResult:
    """
    Independent second validator: does this pair look like something an
    engineer debugging a live fault would actually want, as opposed to
    grounded-but-useless boilerplate (document metadata, cross-reference
    pointers)? Deliberately does NOT look at the source chunk at all —
    grounding already checks "is this text really from the chunk"; this
    checks "is the text worth anything even if it is."

    Four independent checks, each a real failure mode observed in probe
    output. ANY single failure is enough to drop the pair below a 0.6
    pass bar (score <= 0.5) — "reject or heavily penalise" per spec, not
    "dock a quarter point." More failures push the score down further so
    results stay rankable/inspectable rather than all clamping to 0.
    """
    reasons: list[str] = []

    crossref_frac = _crossref_char_fraction(answer)
    if crossref_frac > 0.30:
        reasons.append(f"crossref_dominated ({crossref_frac:.0%} of answer)")

    if not _has_domain_term(answer):
        reasons.append("no_domain_term_in_answer")

    word_count = len(answer.split())
    if word_count < 15:
        reasons.append(f"answer_too_short ({word_count} words)")

    if _DOC_METADATA_RE.search(question) and not _has_domain_term(question):
        reasons.append("question_asks_about_document")

    if not reasons:
        score = 1.0
    else:
        score = max(0.0, 0.5 - 0.15 * (len(reasons) - 1))

    return UtilityResult(score=round(score, 3), reject_reasons=reasons)


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


@dataclass
class GenerateQAResult:
    pairs: list[dict]
    tokens_used: int


def _generate_qa(
    backend: LLMBackend,
    chunk_doc: dict,
    n: int,
    grounding_threshold: float,
    max_retries: int = 2,
) -> GenerateQAResult:
    """
    Generate n Q&A pairs for chunk_doc via the given backend.
    Retries once on transient errors.  Rate limit errors propagate up.
    Returns every parsed pair annotated with grounding_score AND the
    independent utility_score — callers decide what to do with low
    scores; nothing is silently dropped here.

    tokens_used accumulates across ALL attempts for this chunk (including
    ones that were retried because the response didn't parse) — a retried
    call still cost real quota, so it counts toward the caller's budget
    tracking regardless of whether its output was usable. NOTE: if an
    attempt raises a quota/rate-limit error (see the `raise` below), that
    specific attempt's tokens are NOT recoverable — the exception carries
    no usage data, and the request may not have completed server-side
    anyway. This is a rare edge case (quota exhaustion is monotonic, so
    it's very unlikely a chunk's first attempt succeeds and a *later*
    retry for the same chunk is the one that hits the wall) and is not
    worth the complexity of attaching partial counts to the exception.
    """
    user_msg = QA_USER_TEMPLATE.format(
        n=n,
        source=chunk_doc.get("source", "3GPP"),
        text=chunk_doc["answer"][:2000],  # hard cap to stay inside context
    )

    tokens_used = 0
    for attempt in range(max_retries + 1):
        try:
            completion = backend.complete(SYSTEM_PROMPT, user_msg)
            tokens_used += completion.tokens_used
            pairs = _parse_json_array(completion.text)
            if pairs is None:
                # Model output wasn't parseable as a JSON array at all —
                # worth a retry. An empty array ([]) is different: per
                # the new prompt rules that's the model correctly saying
                # "nothing usable in this excerpt," and must NOT be
                # retried — otherwise every legitimately-empty chunk
                # burns max_retries calls for no reason.
                if attempt < max_retries:
                    continue
                return GenerateQAResult(pairs=[], tokens_used=tokens_used)

            results = []
            for pair in pairs:
                q = str(pair.get("question", "")).strip()
                a = str(pair.get("answer", "")).strip()
                if not q or not a:
                    continue
                grounded = _is_grounded(a, chunk_doc["answer"], grounding_threshold)
                utility = _utility_score(q, a)
                results.append({
                    "question": q,
                    "answer": a,
                    "answer_in_chunk": grounded,
                    "grounding_score": round(_grounding_score(a, chunk_doc["answer"]), 3),
                    "utility_score": utility.score,
                    "reject_reasons": utility.reject_reasons,
                })
            return GenerateQAResult(pairs=results, tokens_used=tokens_used)

        except Exception as exc:
            # Re-raise rate limit / quota errors so caller can handle them
            exc_str = str(exc).lower()
            exc_name = type(exc).__name__
            if (
                "RateLimit" in exc_name
                or "rate_limit" in exc_str
                or "429" in exc_str
                or "resource_exhausted" in exc_str
                or "quota" in exc_str
            ):
                raise
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return GenerateQAResult(pairs=[], tokens_used=tokens_used)

    return GenerateQAResult(pairs=[], tokens_used=tokens_used)


# ─── Resume support ─────────────────────────────────────────────────────────────


def _read_existing_output_rows(path: Path) -> tuple[list[dict], int]:
    """
    Read every valid JSON row from an existing output file.

    Returns (rows, n_malformed_skipped). Tolerates a malformed line (e.g.
    a trailing partial line from a process killed mid-write) instead of
    crashing — such a line is skipped and counted, not repaired or
    truncated from the file. Blank lines are silently ignored (not
    counted as malformed).
    """
    if not path.exists():
        return [], 0
    rows: list[dict] = []
    n_bad = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    return rows, n_bad


def _filter_chunks_for_resume(
    chunks: list[dict], out: Path, resume: bool
) -> tuple[list[dict], int, int]:
    """
    If resume is True and `out` already contains rows, drop any chunk
    whose id already appears as a source_chunk_id among them — that chunk
    was already processed by a prior run, possibly days ago against a
    different backend, and reprocessing it would burn quota for nothing.

    When resume is False, `chunks` is returned unchanged (0, 0) — the
    non-resume path keeps its existing overwrite-from-scratch behaviour.

    Returns (remaining_chunks, n_skipped, n_malformed_lines_in_existing_output).
    """
    if not resume:
        return chunks, 0, 0
    existing_rows, n_bad = _read_existing_output_rows(out)
    processed_ids = {
        r["source_chunk_id"] for r in existing_rows if "source_chunk_id" in r
    }
    remaining = [c for c in chunks if c["id"] not in processed_ids]
    n_skipped = len(chunks) - len(remaining)
    return remaining, n_skipped, n_bad


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
        Path("database/real_kb.jsonl"), "--input", "-i",
        help="Input JSONL (output of scrape_data.py)",
    ),
    out: Path = typer.Option(
        Path("database/real_cache_candidates.jsonl"), "--out", "-o",
    ),
    qa_per_chunk: int = typer.Option(3, "--qa-per-chunk", help="Q&A pairs to generate per chunk"),
    max_chunks: int = typer.Option(0, "--max-chunks", help="Cap chunks processed (0 = all)"),
    sample_rate: float = typer.Option(1.0, "--sample-rate", help="Random fraction of chunks to process"),
    grounding_threshold: float = typer.Option(0.60, "--grounding-threshold"),
    backend: str = typer.Option("groq", "--backend", help="LLM backend: groq or gemini"),
    groq_model: str = typer.Option("llama-3.3-70b-versatile", "--groq-model"),
    gemini_model: str = typer.Option("gemini-2.0-flash", "--gemini-model"),
    rpm: int = typer.Option(0, "--rpm", help="Requests-per-minute limit (0 = backend default: groq=28, gemini=15)"),
    seed: int = typer.Option(42, "--seed"),
    resume: bool = typer.Option(
        False, "--resume",
        help="Skip chunks whose id already appears as source_chunk_id in --out; append instead of overwrite.",
    ),
    token_budget: int = typer.Option(
        0, "--token-budget",
        help="Stop before a request that would likely push this run's token usage over this many. 0 = unlimited.",
    ),
) -> None:
    """
    Generate Q&A pairs from real_kb.jsonl chunks via Groq or Gemini.
    Output goes to real_cache_candidates.jsonl for human review.

    Resumable across multiple days/backends: pass --resume to skip chunks
    already present in --out (matched by source_chunk_id) and append
    rather than overwrite. Combine with --token-budget to stop cleanly
    before a request would likely exceed a per-run token cap, instead of
    running into a 429 mid-request.
    """
    load_dotenv()

    if backend not in ("groq", "gemini"):
        console.print(f"[red]Unknown backend '{backend}'. Choose groq or gemini.[/red]")
        raise typer.Exit(1)

    if backend == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            console.print("[red]GROQ_API_KEY not set in .env[/red]")
            raise typer.Exit(1)
        model_name = groq_model
        default_rpm = 28
        llm: LLMBackend = GroqBackend(api_key, groq_model, temperature=0.4, max_tokens=768)
    else:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            console.print("[red]GOOGLE_API_KEY not set in .env[/red]")
            raise typer.Exit(1)
        model_name = gemini_model
        default_rpm = 15
        llm = GeminiBackend(api_key, gemini_model, temperature=0.4, max_tokens=768)

    effective_rpm = rpm if rpm > 0 else default_rpm

    if not input.exists():
        console.print(f"[red]Input file not found: {input}[/red]")
        console.print("Run [bold]uv run python scripts/scrape_data.py[/bold] first.")
        raise typer.Exit(1)

    # Load chunks
    with open(input) as fh:
        chunks = [json.loads(line) for line in fh if line.strip()]

    # Sample. Seeded once here (not per-branch) so --seed reproducibly
    # controls both the rate-based and count-based sampling below.
    random.seed(seed)

    if sample_rate < 1.0:
        chunks = random.sample(chunks, int(len(chunks) * sample_rate))

    if max_chunks > 0 and max_chunks < len(chunks):
        # Shuffle-then-prefix, NOT random.sample(chunks, max_chunks).
        # random.sample(pop, k) does not guarantee sample(pop, 10) is a
        # subset of sample(pop, 15) under the same seed — its algorithm
        # can differ by k/n ratio. A resumable, multi-day workflow with a
        # growing --max-chunks (start at 10, bump to 1200 over many runs)
        # depends on that nesting: shuffling once and taking a prefix
        # guarantees it, since prefix(10) of a fixed permutation is always
        # exactly the first 10 elements of prefix(15) of that same
        # permutation. Also RANDOM, not a slice off the front:
        # real_kb.jsonl is in document order, so chunks[:N] is guaranteed
        # to be front matter (title page, ToC, Scope, References,
        # Definitions) for any N smaller than that front-matter run — a
        # prefix of a structured document is never representative.
        shuffled = chunks[:]
        random.shuffle(shuffled)
        chunks = shuffled[:max_chunks]

    chunks, n_skipped_resume, n_bad_existing = _filter_chunks_for_resume(chunks, out, resume)

    console.print(f"\n[bold]TeleTriage — Synthetic Q&A Generation[/bold]")
    console.print(f"  Backend:           {backend}")
    if resume:
        console.print(
            f"  Resume:            [green]on[/green] — skipped {n_skipped_resume} "
            f"chunk(s) already in {out}"
        )
        if n_bad_existing:
            console.print(
                f"  [yellow]  ({n_bad_existing} malformed line(s) in existing "
                f"output ignored)[/yellow]"
            )
    console.print(f"  Chunks to process: {len(chunks)}")
    console.print(f"  Q&A per chunk:     {qa_per_chunk}")
    console.print(f"  Model:             {model_name}")
    console.print(f"  RPM limit:         {effective_rpm}")
    console.print(f"  Grounding thresh:  {grounding_threshold}")
    if token_budget > 0:
        console.print(f"  Token budget:      {token_budget:,}")
    console.print(f"  Output:            {out}\n")

    limiter = RateLimiter(effective_rpm)

    n_generated = 0
    n_grounded = 0
    n_errors = 0
    tokens_used_this_run = 0
    chunks_completed_this_run = 0
    results: list[dict] = []
    # Tags every row written by this process invocation, so a later run
    # can report "N chunks, M pairs (across K runs)" by counting distinct
    # values of this field across the whole output file.
    run_started_at = datetime.now(timezone.utc).isoformat()

    out.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if resume else "w"

    with open(out, file_mode) as out_fh, Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating…", total=len(chunks))

        for chunk_doc in chunks:
            # Token-budget check happens BEFORE issuing a request, using
            # the running average tokens/chunk from THIS run, so the run
            # ends on a clean chunk boundary with a full row written
            # rather than mid-request on a 429. With zero samples yet we
            # can't estimate, so the very first chunk always proceeds.
            if token_budget > 0 and chunks_completed_this_run > 0:
                avg_tokens_per_chunk = tokens_used_this_run / chunks_completed_this_run
                if tokens_used_this_run + avg_tokens_per_chunk > token_budget:
                    # escape(): chunk_doc["id"] is wrapped in literal "[" "]"
                    # below — Rich's console.print() treats [...] as markup,
                    # and an unrecognized tag like [real_00042] is silently
                    # DROPPED, not printed literally. escape() makes Rich
                    # render the brackets as plain text instead of eating them.
                    console.print(
                        f"\n[yellow]  Token budget ({token_budget:,}) would likely "
                        f"be exceeded by the next chunk — stopping cleanly before "
                        f"{escape('[' + chunk_doc['id'] + ']')}.[/yellow]"
                    )
                    break

            progress.update(
                task,
                description=f"  {escape('[' + chunk_doc['id'] + ']')} {escape(chunk_doc['source'])}",
                advance=1,
            )

            limiter.wait()
            try:
                qa_result = _generate_qa(
                    llm,
                    chunk_doc,
                    n=qa_per_chunk,
                    grounding_threshold=grounding_threshold,
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                exc_name = type(exc).__name__
                is_quota = (
                    "RateLimit" in exc_name
                    or "rate_limit" in exc_str
                    or "429" in exc_str
                    or "resource_exhausted" in exc_str
                    or "quota" in exc_str
                )
                if is_quota:
                    # Hard quota exhausted — stop immediately rather than waste time.
                    # Everything written so far is already flushed to `out_fh`
                    # (each row is written+flushed as it's produced, below), so
                    # a --resume run tomorrow picks up exactly where this left off.
                    if "limit: 0" in str(exc) or "per_day" in exc_str or "per day" in exc_str:
                        console.print(f"[red]  Daily quota exhausted for {backend} — stopping.[/red]")
                        console.print(f"[red]  {escape(str(exc))}[/red]")
                        break
                    console.print("[yellow]  Rate limit hit — sleeping 60s[/yellow]")
                    time.sleep(60)
                    n_errors += 1
                    continue
                # escape(exc): exception messages are external, unpredictable
                # text (API error bodies etc.) and may themselves contain
                # literal "[...]" that Rich would otherwise try to parse as
                # markup and silently drop.
                console.print(f"[red]  Error on {chunk_doc['id']}: {escape(str(exc))}[/red]")
                n_errors += 1
                continue

            tokens_used_this_run += qa_result.tokens_used
            chunks_completed_this_run += 1

            for local_idx, pair in enumerate(qa_result.pairs):
                n_generated += 1
                if pair["answer_in_chunk"]:
                    n_grounded += 1
                # ID is derived from source_chunk_id + a local index, not a
                # global running counter: a purely sequential "cand_00000"
                # scheme (the old behaviour) would collide across resumed
                # runs, since each run's in-memory counter restarts at 0
                # while appending to the SAME file. Since resume filtering
                # guarantees a given chunk is only ever processed by one
                # run, (chunk id, local pair index) is unique file-wide.
                row = {
                    "id": f"cand_{chunk_doc['id']}_{local_idx:02d}",
                    "source_chunk_id": chunk_doc["id"],
                    "source": chunk_doc.get("source", ""),
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "answer_in_chunk": pair["answer_in_chunk"],
                    "grounding_score": pair["grounding_score"],
                    # Independent of grounding — see _utility_score(). Written
                    # (not silently dropped) so rejects are inspectable.
                    "utility_score": pair["utility_score"],
                    "reject_reasons": pair["reject_reasons"],
                    "validated": False,   # set to True after human review
                    "run_started_at": run_started_at,
                }
                results.append(row)
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fh.flush()

    console.print(f"\n[bold]Results[/bold]")
    console.print(
        f"  This run:  {chunks_completed_this_run} chunks, {n_generated} pairs, "
        f"{tokens_used_this_run:,} tokens"
    )
    console.print(f"  Grounded (pass):   [green]{n_grounded}[/green] ({100*n_grounded/max(n_generated,1):.0f}%)")
    console.print(f"  Failed grounding:  [yellow]{n_generated - n_grounded}[/yellow]")
    console.print(f"  API errors:        {n_errors}")

    final_rows, n_bad_final = _read_existing_output_rows(out)
    total_pairs = len(final_rows)
    total_chunks = len({r["source_chunk_id"] for r in final_rows if "source_chunk_id" in r})
    tagged_runs = {r["run_started_at"] for r in final_rows if r.get("run_started_at")}
    has_untagged = any(not r.get("run_started_at") for r in final_rows)
    n_runs = len(tagged_runs) + (1 if has_untagged else 0)
    console.print(
        f"  Total:     {total_chunks} chunks, {total_pairs} pairs "
        f"(across {n_runs} run{'s' if n_runs != 1 else ''})"
    )
    if n_bad_final:
        console.print(f"  [yellow]  ({n_bad_final} malformed line(s) in {out} ignored)[/yellow]")

    console.print(f"\n  Saved → [bold]{out}[/bold]")
    console.print(
        f"  [dim]Promote grounded pairs to cache:[/dim]  "
        f"jq 'select(.answer_in_chunk)' {out} | head -20"
    )
    if token_budget > 0:
        console.print(
            f"  [dim]Resume tomorrow:[/dim]  "
            f"uv run python scripts/synth_qa.py --resume --out {out} "
            f"--backend {backend} --max-chunks <total-target> --token-budget {token_budget}"
        )

    # ─── Grounding vs. utility — independence check (this run) ─────────────
    # grounding_score and utility_score measure orthogonal things (see
    # _utility_score's docstring). Reporting both pass rates AND their
    # correlation is the point: if r is near zero, that's the empirical
    # justification for why the pipeline needs both gates, not just one.
    n = len(results)
    if n > 0:
        grounding_scores = [r["grounding_score"] for r in results]
        utility_scores = [r["utility_score"] for r in results]
        n_ground_pass = sum(1 for s in grounding_scores if s >= 0.6)
        n_util_pass = sum(1 for s in utility_scores if s >= 0.6)
        n_both_pass = sum(
            1 for g, u in zip(grounding_scores, utility_scores) if g >= 0.6 and u >= 0.6
        )

        if n >= 2 and len(set(grounding_scores)) > 1 and len(set(utility_scores)) > 1:
            correlation = float(np.corrcoef(grounding_scores, utility_scores)[0, 1])
            corr_str = f"{correlation:.3f}"
        else:
            corr_str = "n/a (insufficient variance)"

        console.print(f"\n[bold]Grounding vs. Utility — independence check (this run)[/bold]")
        console.print(f"  Pairs total:                    {n}")
        console.print(f"  Grounding >= 0.6 (pass):        {n_ground_pass} ({100*n_ground_pass/n:.0f}%)")
        console.print(f"  Utility >= 0.6 (pass):          {n_util_pass} ({100*n_util_pass/n:.0f}%)")
        console.print(f"  Both >= 0.6:                    {n_both_pass} ({100*n_both_pass/n:.0f}%)")
        console.print(f"  Pearson r(grounding, utility):  {corr_str}")

    console.print()


if __name__ == "__main__":
    app()

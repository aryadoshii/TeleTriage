"""
Download 3GPP Technical Specifications from the public FTP archive,
chunk the text, deduplicate near-identical chunks with MinHash LSH,
and write data/real_kb.jsonl — same schema as data/sample_kb.jsonl.

Usage:
    uv run python scripts/scrape_data.py
    uv run python scripts/scrape_data.py --specs 36.300,36.331
    uv run python scripts/scrape_data.py --chunk-tokens 512 --overlap-tokens 64
    uv run python scripts/scrape_data.py --out data/real_kb.jsonl --dry-run

Prerequisites:
    uv add python-docx datasketch   (already in pyproject.toml)

━━━ KEY CONCEPTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recursive character splitter
  Split on coarser separators first (\\n\\n, \\n, ". "), merging pieces
  into windows of target_size with overlap.  Falls back to the next
  separator only when a piece still exceeds target_size.  "Recursive"
  refers to the separator cascade, not the call stack.

Token ↔ character conversion
  1 token ≈ 4 chars for English prose (GPT-4 tokenizer empirical average).
  3GPP specs contain dense notation so the real ratio skews toward 3 chars/tok,
  but 4 is conservative and keeps chunks comfortably inside the 512-token
  context window of BGE-small-en-v1.5.

MinHash deduplication
  MinHash estimates Jaccard similarity J(A,B) = |A∩B| / |A∪B| between
  sets of k-shingles (k=5 words).  With num_perm=128 hash functions the
  standard error on J is ≈ 1/√128 ≈ 0.09 — good enough for threshold 0.85.
  MinHash LSH (locality-sensitive hashing) groups similar items into buckets
  so we avoid the O(n²) pairwise comparison: query is O(1) amortised.
  We remove a chunk if any existing chunk has J ≥ threshold.

3GPP FTP URL structure
  Base:    https://www.3gpp.org/ftp/Specs/archive/
  Series:  {spec_series}_series/{spec_id}/
  Example: 36_series/36.300/36300-h30.zip
  Versioning: the filename suffix encodes the release (letter) and sub-
  version (digits).  Ascending lexicographic order equals ascending version
  (digits < letters in ASCII), so sorted(links)[-1] is always the latest.
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests
import typer
from docx import Document
from datasketch import MinHash, MinHashLSH
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

# ─── Constants ────────────────────────────────────────────────────────────────

FTP_BASE = "https://www.3gpp.org/ftp/Specs/archive"

DEFAULT_SPECS = ["36.300", "36.331", "36.413", "24.301", "23.401"]

CHARS_PER_TOKEN = 4          # 1 token ≈ 4 chars for English prose
MIN_CHUNK_CHARS = 200        # discard chunks shorter than this
MINHASH_NUM_PERM = 128       # hash functions — standard error ≈ 1/√128
MINHASH_SHINGLE_K = 5        # k-word shingles for Jaccard estimation
REQUEST_TIMEOUT = 60         # seconds
INTER_REQUEST_DELAY = 1.0    # seconds between requests (respectful rate limit)

app = typer.Typer(add_completion=False)
console = Console()


# ─── 3GPP FTP helpers ────────────────────────────────────────────────────────


def _series_for(spec_id: str) -> str:
    """'36.300' → '36_series'"""
    return spec_id.split(".")[0] + "_series"


def _find_latest_zip_url(spec_id: str, session: requests.Session) -> str:
    """
    Parse the 3GPP FTP directory listing and return the URL of the
    latest ZIP for the given spec.

    The directory is an Apache index page; links look like:
        <a href="36300-h30.zip">36300-h30.zip</a>
    Lexicographic descending order = newest first (digits < letters in ASCII,
    so higher release letters sort last).
    """
    series = _series_for(spec_id)
    index_url = f"{FTP_BASE}/{series}/{spec_id}/"
    resp = session.get(index_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    # Extract all ZIP hrefs from the Apache directory listing
    links = re.findall(r'href="([^"]+\.zip)"', resp.text, re.IGNORECASE)
    if not links:
        raise ValueError(f"No ZIP files found at {index_url}")

    # Remove any leading path components (keep filename only for sort)
    basenames = [lnk.split("/")[-1] for lnk in links]
    basenames.sort()
    latest = basenames[-1]
    return index_url + latest


def _download_zip(url: str, cache_dir: Path, session: requests.Session) -> Path:
    """
    Download a ZIP to cache_dir if not already present.
    Shows a Rich progress bar during download.
    Returns the local path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest = cache_dir / filename
    if dest.exists():
        console.print(f"    [dim]cache hit:[/dim] {filename}")
        return dest

    with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"  {filename}", total=total or None)
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    progress.advance(task, len(chunk))

    return dest


def _extract_docx_bytes(zip_path: Path) -> bytes | None:
    """
    Open a 3GPP ZIP and return the bytes of the first DOCX file found.
    Returns None if no DOCX is present (old .doc binary format is skipped).
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            # Prefer .docx; fall back to checking any .doc that's also a ZIP
            docx_names = [n for n in names if n.lower().endswith(".docx")]
            if not docx_names:
                # Some specs ship as .doc — try treating them as DOCX (newer
                # .doc is sometimes a renamed DOCX)
                docx_names = [n for n in names if n.lower().endswith(".doc")]
            if not docx_names:
                return None
            return zf.read(docx_names[0])
    except zipfile.BadZipFile:
        return None


def _docx_to_text(docx_bytes: bytes) -> str:
    """
    Extract plain text from a DOCX blob.

    python-docx iterates over paragraphs in document order.  Tables are
    visited cell-by-cell via the iter_inner_content() API — we join cells
    with a tab so column boundaries survive as whitespace.

    ASN.1 code blocks are preserved as-is (they appear as normal paragraphs
    with a monospaced style in 3GPP docs).
    """
    doc = Document(io.BytesIO(docx_bytes))
    parts: list[str] = []

    for block in doc.element.body:
        tag = block.tag.split("}")[-1]  # local tag name

        if tag == "p":                   # paragraph
            para = block.text_content() if hasattr(block, "text_content") else ""
            # python-docx paragraph text
            from docx.oxml.ns import qn
            text = "".join(n.text or "" for n in block.iter() if n.tag == qn("w:t"))
            if text.strip():
                parts.append(text)

        elif tag == "tbl":               # table — flatten to tab-separated rows
            from docx.oxml.ns import qn as _qn
            rows: list[str] = []
            for tr in block.iter(_qn("w:tr")):
                cells = []
                for tc in tr.iter(_qn("w:tc")):
                    cell_text = "".join(
                        n.text or "" for n in tc.iter() if n.tag == _qn("w:t")
                    )
                    cells.append(cell_text.strip())
                row_text = "\t".join(c for c in cells if c)
                if row_text:
                    rows.append(row_text)
            if rows:
                parts.append("\n".join(rows))

    return "\n\n".join(parts)


def _clean_text(text: str) -> str:
    """
    Normalise whitespace and strip common 3GPP artefacts:
    - Table-of-contents dot leaders ("Section 1 ........... 5")
    - Page headers/footers ("3GPP TS 36.300 version 17.3.0 Release 17")
    - Repeated blank lines
    """
    # Remove dot-leader ToC lines
    text = re.sub(r"[^\S\n]*\.{4,}[^\S\n]*\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove lines that look like running headers (starts with "3GPP TS")
    text = re.sub(r"^3GPP TS \d+\.\d+.*$", "", text, flags=re.MULTILINE)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalise other whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ─── Recursive character splitter ────────────────────────────────────────────


class RecursiveCharSplitter:
    """
    Split text into overlapping windows using a separator cascade.

    The separators are tried from coarsest to finest:
        \\n\\n → paragraph break
        \\n    → line break
        ". "   → sentence
        " "    → word
        ""     → character (last resort)

    For each separator, pieces that still exceed chunk_size are recursively
    split using the next finer separator.  Small pieces are merged back into
    windows of at most chunk_size chars, each window overlapping the previous
    by overlap chars so context isn't lost at boundaries.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int, overlap: int) -> None:
        self._size = chunk_size
        self._overlap = overlap

    def split(self, text: str) -> list[str]:
        return self._split(text.strip(), self.SEPARATORS)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []
        if len(text) <= self._size:
            return [text]

        sep, remaining = separators[0], separators[1:]
        pieces = text.split(sep) if sep else list(text)

        # Recursively handle pieces that are still too large
        good_pieces: list[str] = []
        for piece in pieces:
            if len(piece) > self._size:
                if remaining:
                    good_pieces.extend(self._split(piece, remaining))
                else:
                    # Hard split at char boundary
                    for i in range(0, len(piece), self._size - self._overlap):
                        good_pieces.append(piece[i : i + self._size])
            else:
                good_pieces.append(piece)

        return self._merge(good_pieces, sep)

    def _merge(self, pieces: list[str], sep: str) -> list[str]:
        """
        Merge small pieces into chunks ≤ chunk_size with overlap.

        Algorithm: maintain a sliding window (deque of pieces).  When
        adding a piece would overflow, emit the current window as a chunk,
        then drop pieces from the front until total ≤ overlap chars.
        """
        chunks: list[str] = []
        window: list[str] = []
        window_len = 0

        for piece in pieces:
            piece_len = len(piece)
            sep_cost = len(sep) if window else 0
            if window_len + sep_cost + piece_len > self._size and window:
                chunks.append(sep.join(window))
                # Drop from front until window ≤ overlap
                while window and window_len > self._overlap:
                    dropped = window.pop(0)
                    window_len -= len(dropped) + (len(sep) if window else 0)
            window.append(piece)
            window_len += sep_cost + piece_len

        if window:
            chunks.append(sep.join(window))

        return [c for c in chunks if c.strip()]


# ─── Quality filter ───────────────────────────────────────────────────────────


def _is_quality_chunk(text: str, min_chars: int) -> bool:
    """
    Return False for chunks that are noise rather than content:
    - Too short
    - Mostly punctuation / dots (leftover ToC with dot leaders)
    - Mostly digits (page numbers, figure numbers)
    - Table-of-contents lines: 'Section title  142' (text then bare number)

    This filter only screens for PROSE DENSITY — it happily passes
    well-formed prose that is administrative rather than operational
    (title page, foreword, scope, references, copyright notice). See
    _find_body_start_index() / _is_references_dominated() /
    _is_boilerplate() below for the filters that catch those.
    """
    import re as _re
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    non_space = stripped.replace(" ", "")
    if not non_space:
        return False
    # Dot fraction > 15% → likely ToC with dot leaders
    if non_space.count(".") / len(non_space) > 0.15:
        return False
    # Digit fraction > 60% → likely a table of numbers only
    if sum(c.isdigit() for c in non_space) / len(non_space) > 0.60:
        return False
    # ToC without dot leaders: >40% of lines end with a bare page number
    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    if lines:
        toc_lines = sum(1 for l in lines if _re.search(r'\w\s+\d{1,4}$', l))
        if toc_lines / len(lines) > 0.40:
            return False
    return True


# ─── Front-matter filter ───────────────────────────────────────────────────
#
# The density filter above passes the title page, table of contents,
# "1 Scope", "2 References", and "3 Definitions, symbols and abbreviations"
# wholesale — they're well-formed prose, just not operational content. A
# probe run confirmed this: the four worst synthetic Q&A pairs it produced
# all came from these clauses (a release-number question, a "where do I
# get the spec" question, two answers whose entire substance was a
# cross-reference to another clause).
#
# DISCOVERY 1 (verified against database/real_kb.jsonl before writing
# this): 3GPP DOCX->text extraction collapses the tab stop between a
# clause number and its title, so BOTH table-of-contents rows AND real
# body headings read identically as e.g. "4.1Functional Split" — no
# space. The only reliable discriminator is what follows: ToC rows
# additionally collapse the tab-leader to the page number, so they end in
# a bare 1-4 digit page number ("4.1Functional Split34"); real heading
# paragraphs don't.
#
# DISCOVERY 2 (found by testing the first version of this filter against
# the real corpus — the naive "stop at the first depth>=2 heading" rule
# lands on the "3.1Definitions" HEADING ITSELF, which starts clause 3's
# body, not clause 4's. That kept all of clause 3's actual glossary
# content (chunks like "A2X communication: A communication to
# support..."), directly contradicting "drop everything up to and
# including 3 Definitions". Fix: find the clause that contains the first
# depth>=2 heading (empirically always "3", i.e. Definitions), then keep
# scanning until a heading with a DIFFERENT top-level clause number shows
# up (empirically "4") — THAT'S the real body boundary. This still
# doesn't hardcode "3" or "4" anywhere, so it generalises to specs whose
# clause 3 is unusually long or short.
#
# DISCOVERY 3: abbreviation-list entries in this corpus lose their
# separator the exact same way clause headings do ("5GC" + tab + "5G Core
# Network" -> "5GC5G Core Network"), and plenty of telecom abbreviations
# start with digits (5GC, 3GPP2, 2G, 4G, 1xRTT...). "5GC5G Core Network"
# matched an early version of this filter's "any depth" heading check as
# top-level clause "5" and wrongly signalled the clause-4 transition from
# inside clause 3's own abbreviation list. Fix: require depth>=2 (a real
# dot-separated sub-clause number, "4.1") for BOTH the anchor heading and
# the transition heading, not just the anchor. A digit-led acronym is
# never followed by ".digit" — 3GPP body text has no "5G.1SomethingText"
# — so requiring that dot eliminates the entire class of false positive
# instead of chasing individual acronyms.
#
# Verified end-to-end for all 5 specs in this corpus (see the dry-run
# used to calibrate this filter) — each one's transition chunk is real
# clause-4-or-later prose, not glossary content.

_HEADING_DEPTH2_RE = re.compile(r"^(\d+)((?:\.\d+)+)([A-Z].*)$")
_TRAILING_PAGE_NUM_RE = re.compile(r"\d{1,4}$")


def _heading_clause_depth2(para: str) -> int | None:
    """
    If `para` looks like a genuine depth->=2 numbered clause heading
    ("4.1Functional Split" — requires at least one dot-separated sub-
    number, which rules out digit-led acronyms like "5GC" or "3GPP2"),
    return its top-level clause number (e.g. 4). Otherwise None.

    Excludes:
      - table-of-contents rows: same squished shape, but the tab-leader
        to the page number is ALSO collapsed, so they end in a bare
        1-4 digit page number ("4.1Functional Split34") that a real
        heading paragraph's title text doesn't have.
      - glossary/definition entries ("Split bearer: in dual
        connectivity..."): these have a colon early in the paragraph;
        headings don't.
      - anything that isn't short (a heading is a title, not a sentence).
    """
    para = para.strip()
    if not (4 <= len(para) <= 80) or ":" in para[:50]:
        return None
    m = _HEADING_DEPTH2_RE.match(para)
    if not m:
        return None
    if _TRAILING_PAGE_NUM_RE.search(m.group(3)):
        return None  # ToC row
    return int(m.group(1))


def _find_body_start_index(chunks: list[str]) -> int:
    """
    Return the index (into `chunks`, in document order) of the first chunk
    that belongs to the clause AFTER the one containing the first
    depth->=2 heading — i.e. everything through the end of clause 3
    (Definitions, symbols and abbreviations) is front matter; clause 4
    onward is body. Two passes in one loop: the first depth>=2 heading
    found sets the "anchor" clause number (empirically always clause 3);
    the next depth>=2 heading whose top-level number DIFFERS from the
    anchor marks the real body boundary (empirically clause 4). Stopping
    at the anchor itself (a simpler, first-draft version of this filter)
    is wrong — it keeps all of clause 3's own glossary content, which
    directly contradicts "drop everything up to and including 3
    Definitions".

    This doesn't hardcode "3" or "4" anywhere, so it generalises to specs
    whose clause 3 is unusually long or short.

    Returns 0 (nothing dropped) if no such transition is found — fails
    safe rather than dropping the whole document.
    """
    anchor_clause: int | None = None
    for i, chunk in enumerate(chunks):
        for para in chunk.split("\n\n"):
            top = _heading_clause_depth2(para)
            if top is None:
                continue
            if anchor_clause is None:
                anchor_clause = top
            elif top != anchor_clause:
                return i
    return 0


# ─── Reference-list filter ──────────────────────────────────────────────────

_REF_LINE_RE = re.compile(r"^\[\d+\]")
_REF_TS_RE = re.compile(r"3GPP TS \d\d\.\d\d\d")


def _is_references_dominated(text: str) -> bool:
    """
    True if more than 40% of a chunk's lines look like reference-list
    entries — "[12]" citation markers or bare "3GPP TS NN.NNN" pointers.
    Independent of chunk position: catches clause 2 (References) content
    even where the front-matter position filter alone wouldn't (e.g. a
    references list re-appearing in an annex bibliography).
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    ref_lines = sum(
        1 for l in lines
        if _REF_LINE_RE.match(l.strip()) or _REF_TS_RE.search(l)
    )
    return ref_lines / len(lines) > 0.40


# ─── Boilerplate filter ─────────────────────────────────────────────────────

_BOILERPLATE_MARKERS = (
    "All rights reserved",
    "Organizational Partners",
    "Postal address",
    "Internet http://",
    "shall not be construed",
)


def _is_boilerplate(text: str) -> bool:
    """True if the chunk contains 3GPP/ETSI copyright or legal front-matter text."""
    return any(marker in text for marker in _BOILERPLATE_MARKERS)


# ─── MinHash deduplication ────────────────────────────────────────────────────


def _shingles(text: str, k: int = MINHASH_SHINGLE_K) -> set[bytes]:
    """
    Build the set of k-word shingles used for Jaccard estimation.

    Jaccard similarity on word shingles correlates well with semantic
    overlap for near-duplicate detection.  k=5 is long enough to be
    distinctive but short enough to match paraphrased repeats.
    """
    words = text.lower().split()
    if len(words) < k:
        return {text.lower().encode()}
    return {" ".join(words[i : i + k]).encode() for i in range(len(words) - k + 1)}


def _make_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    for s in _shingles(text):
        m.update(s)
    return m


def deduplicate(chunks: list[str], threshold: float) -> tuple[list[int], int]:
    """
    Remove chunks whose Jaccard similarity to any already-kept chunk
    exceeds threshold.  Returns (kept_indices, n_dropped) — indices into
    the original `chunks` list, NOT the text values themselves.

    Returning indices (not text) is deliberate: the caller rebuilds the
    final KB by indexing directly into its source list with these, so
    two chunks that happen to share identical text (e.g. a boilerplate
    note repeated verbatim elsewhere) can never cause one to be silently
    re-included via the other. A prior version of the caller matched
    kept chunks back to source objects by text value + id(), which
    reintroduced 142 exact-duplicate chunks that this function correctly
    dropped — see git history. Returning indices makes that class of bug
    structurally impossible, not just patched.

    MinHash LSH complexity: O(n) insert + O(1) query per chunk,
    vs O(n²) for naive pairwise comparison.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_NUM_PERM)
    kept_indices: list[int] = []
    dropped = 0

    for i, chunk in enumerate(chunks):
        m = _make_minhash(chunk)
        if lsh.query(m):
            dropped += 1
        else:
            lsh.insert(str(i), m)
            kept_indices.append(i)

    return kept_indices, dropped


# ─── Main pipeline ────────────────────────────────────────────────────────────


@app.command()
def main(
    specs: str = typer.Option(
        ",".join(DEFAULT_SPECS),
        "--specs",
        help="Comma-separated 3GPP spec IDs (e.g. 36.300,24.301)",
    ),
    out: Path = typer.Option(
        Path("database/real_kb.jsonl"), "--out", "-o",
        help="Output JSONL path",
    ),
    cache_dir: Path = typer.Option(
        Path("database/raw_zips"), "--cache-dir",
        help="Directory to cache downloaded ZIPs",
    ),
    chunk_tokens: int = typer.Option(512, "--chunk-tokens", help="Target chunk size in tokens"),
    overlap_tokens: int = typer.Option(64, "--overlap-tokens", help="Overlap in tokens"),
    dedup_threshold: float = typer.Option(0.85, "--dedup-threshold", help="MinHash Jaccard threshold"),
    min_chunk_chars: int = typer.Option(MIN_CHUNK_CHARS, "--min-chunk-chars"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Download and parse but don't write output"),
) -> None:
    """
    Scrape 3GPP specs → chunk → deduplicate → data/real_kb.jsonl
    """
    spec_list = [s.strip() for s in specs.split(",") if s.strip()]
    chunk_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    splitter = RecursiveCharSplitter(chunk_chars, overlap_chars)

    console.print(f"\n[bold]TeleTriage — Phase 5 Data Pipeline[/bold]")
    console.print(f"  Specs:          {', '.join(spec_list)}")
    console.print(f"  Chunk size:     {chunk_tokens} tok ({chunk_chars} chars)")
    console.print(f"  Overlap:        {overlap_tokens} tok ({overlap_chars} chars)")
    console.print(f"  Dedup thresh:   Jaccard ≥ {dedup_threshold}")
    console.print(f"  Output:         {out}\n")

    session = requests.Session()
    session.headers["User-Agent"] = "TeleTriage-DataPipeline/0.2 (research)"

    all_chunks: list[dict] = []
    per_spec_stats: list[dict] = []

    for spec_id in spec_list:
        console.rule(f"TS {spec_id}")

        # ── 1. Find and download ──────────────────────────────────────
        try:
            zip_url = _find_latest_zip_url(spec_id, session)
            console.print(f"  Latest: [cyan]{zip_url.split('/')[-1]}[/cyan]")
        except Exception as exc:
            console.print(f"  [red]Could not find ZIP for TS {spec_id}: {exc}[/red]")
            per_spec_stats.append({"spec": spec_id, "error": str(exc)})
            continue

        time.sleep(INTER_REQUEST_DELAY)

        try:
            zip_path = _download_zip(zip_url, cache_dir, session)
        except Exception as exc:
            console.print(f"  [red]Download failed: {exc}[/red]")
            per_spec_stats.append({"spec": spec_id, "error": str(exc)})
            continue

        time.sleep(INTER_REQUEST_DELAY)

        # ── 2. Extract DOCX text ──────────────────────────────────────
        docx_bytes = _extract_docx_bytes(zip_path)
        if docx_bytes is None:
            console.print(
                f"  [yellow]No DOCX found in ZIP — spec may use old .doc binary "
                f"format which requires LibreOffice conversion. Skipping.[/yellow]"
            )
            per_spec_stats.append({"spec": spec_id, "error": "no_docx"})
            continue

        try:
            raw_text = _docx_to_text(docx_bytes)
        except Exception as exc:
            console.print(f"  [red]DOCX parse error: {exc}[/red]")
            per_spec_stats.append({"spec": spec_id, "error": str(exc)})
            continue

        cleaned = _clean_text(raw_text)
        console.print(f"  Extracted {len(cleaned):,} chars from DOCX")

        # ── 3. Chunk ──────────────────────────────────────────────────
        raw_chunks = splitter.split(cleaned)
        before_filter = len(raw_chunks)

        # Stage 1: prose-density filter (existing) — screens ToC dot
        # leaders, digit-heavy tables, bare-page-number lines. Well-formed
        # front matter (title page, Scope, References, Definitions) passes
        # this stage untouched.
        density_pass = [
            (i, c) for i, c in enumerate(raw_chunks) if _is_quality_chunk(c, min_chunk_chars)
        ]

        # Stage 2: front-matter position filter — drop everything before
        # clause 4 (title page, ToC, Scope, References, Definitions).
        body_start_idx = _find_body_start_index(raw_chunks)
        front_matter_dropped = sum(1 for i, _ in density_pass if i < body_start_idx)
        after_front_matter = [(i, c) for i, c in density_pass if i >= body_start_idx]

        # Stage 3: reference-list-dominated chunks (independent of position).
        references_dropped = sum(1 for _, c in after_front_matter if _is_references_dominated(c))
        after_references = [
            (i, c) for i, c in after_front_matter if not _is_references_dominated(c)
        ]

        # Stage 4: copyright / legal boilerplate.
        boilerplate_dropped = sum(1 for _, c in after_references if _is_boilerplate(c))
        quality_chunks = [c for _, c in after_references if not _is_boilerplate(c)]

        console.print(
            f"  Chunks: {before_filter} raw → {len(density_pass)} after density filter "
            f"→ {len(quality_chunks)} after front-matter/references/boilerplate filters"
        )
        console.print(
            f"    dropped — density: {before_filter - len(density_pass)}, "
            f"front-matter: {front_matter_dropped}, "
            f"references-list: {references_dropped}, "
            f"boilerplate: {boilerplate_dropped}"
        )

        source_label = f"3GPP TS {spec_id}"
        spec_docs = [
            {
                "id": f"real_{len(all_chunks) + i:05d}",
                "question": "",
                "answer": chunk,
                "tags": [spec_id, "3gpp", "lte"],
                "source": source_label,
                "spec_id": spec_id,
                "chunk_index": i,
                "char_count": len(chunk),
            }
            for i, chunk in enumerate(quality_chunks)
        ]
        all_chunks.extend(spec_docs)

        per_spec_stats.append({
            "spec": spec_id,
            "chars": len(cleaned),
            "raw_chunks": before_filter,
            "density_dropped": before_filter - len(density_pass),
            "front_matter_dropped": front_matter_dropped,
            "references_dropped": references_dropped,
            "boilerplate_dropped": boilerplate_dropped,
            "quality_chunks": len(quality_chunks),
        })

    if not all_chunks:
        console.print("\n[red]No chunks produced — check network access and spec IDs.[/red]")
        raise typer.Exit(1)

    # ── 4. Global deduplication ───────────────────────────────────────
    console.rule("Deduplication")
    texts = [d["answer"] for d in all_chunks]
    console.print(f"  Running MinHash LSH on {len(texts)} chunks (threshold={dedup_threshold})…")
    kept_indices, n_dropped = deduplicate(texts, dedup_threshold)
    dedup_rate = n_dropped / len(texts) if texts else 0.0

    # Rebuild docs list by indexing directly into all_chunks with the KEPT
    # INDICES from deduplicate(), re-numbering IDs. No identity comparison,
    # no text matching — see deduplicate()'s docstring for why that
    # matters (a prior id()-based version of this rebuild silently
    # reintroduced 142 exact-duplicate chunks that MinHash had correctly
    # dropped).
    kept_docs = [
        {**all_chunks[i], "id": f"real_{j:05d}"}
        for j, i in enumerate(kept_indices)
    ]

    console.print(f"  Dropped {n_dropped} near-duplicates ({100*dedup_rate:.1f}%)")
    console.print(f"  Final KB entries: [bold green]{len(kept_docs)}[/bold green]")

    # ── 5. Write output ───────────────────────────────────────────────
    if not dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            for doc in kept_docs:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        console.print(f"\n  Saved → [bold]{out}[/bold]")

    # ── 6. Summary report ─────────────────────────────────────────────
    char_counts = [d["char_count"] for d in kept_docs]
    if char_counts:
        import statistics
        p25 = sorted(char_counts)[len(char_counts) // 4]
        p50 = sorted(char_counts)[len(char_counts) // 2]
        p75 = sorted(char_counts)[3 * len(char_counts) // 4]
    else:
        p25 = p50 = p75 = 0

    console.rule("Summary")

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Spec", style="cyan")
    tbl.add_column("Chars", justify="right")
    tbl.add_column("Raw chunks", justify="right")
    tbl.add_column("Dropped: density", justify="right")
    tbl.add_column("Dropped: front-matter", justify="right")
    tbl.add_column("Dropped: refs-list", justify="right")
    tbl.add_column("Dropped: boilerplate", justify="right")
    tbl.add_column("Quality", justify="right", style="bold green")

    for stat in per_spec_stats:
        if "error" in stat:
            tbl.add_row(
                stat["spec"], "—", "—", "—", "—", "—", "—",
                f"[red]error: {stat['error']}[/red]",
            )
        else:
            tbl.add_row(
                stat["spec"],
                f"{stat['chars']:,}",
                str(stat["raw_chunks"]),
                str(stat["density_dropped"]),
                str(stat["front_matter_dropped"]),
                str(stat["references_dropped"]),
                str(stat["boilerplate_dropped"]),
                str(stat["quality_chunks"]),
            )
    console.print(tbl)

    console.print(f"\n  Before dedup:  {len(all_chunks)}")
    console.print(f"  After dedup:   {len(kept_docs)}  (−{n_dropped}, {100*dedup_rate:.1f}% dropped)")
    console.print(f"\n  Chunk size distribution (chars):")
    console.print(f"    p25:  {p25}")
    console.print(f"    p50:  {p50}  (target: {chunk_chars})")
    console.print(f"    p75:  {p75}")
    console.print()


if __name__ == "__main__":
    app()

"""
TeleTriage — Production Dashboard
Glassmorphism UI with full observability.

Run:
    uv run streamlit run frontend/dashboard.py --server.port 8501
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

from backend.config import get_config
from backend.tiers.retrieval_tier import load_index_manifest

# ─── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="TeleTriage",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ─── Global CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #050d1a !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: #F5F5DC !important;
}

/* ── Animated background mesh ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(ellipse 80% 50% at 10% 20%, rgba(130,200,229,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(147,197,114,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(130,200,229,0.05) 0%, transparent 70%);
    pointer-events: none;
}

/* ── Sidebar — always visible, CSS overrides Streamlit collapse transform ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"][aria-expanded="false"],
section[data-testid="stSidebar"][aria-expanded="true"] {
    transform: translateX(0) !important;
    min-width: 16rem !important;
    width: 16rem !important;
    background: rgba(8, 20, 40, 0.92) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(130,200,229,0.15) !important;
    padding-top: 0 !important;
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}

/* ── Main content padding ── */
.main .block-container {
    padding: 2rem 2.5rem 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(14, 30, 60, 0.55);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(130,200,229,0.18);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: border-color 0.2s;
}

.glass-card:hover {
    border-color: rgba(130,200,229,0.35);
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.25rem;
}

.metric-card {
    flex: 1;
    background: rgba(14, 30, 60, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(130,200,229,0.18);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(130,200,229,0.6), transparent);
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(148,163,184,0.9);
    margin-bottom: 0.4rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}

.metric-sub {
    font-size: 0.75rem;
    color: rgba(148,163,184,0.7);
    margin-top: 0.3rem;
}

/* ── Tier badges ── */
.tier-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

.tier-cache    { background: rgba(130,200,229,0.18);  color: #82C8E5; border: 1px solid rgba(130,200,229,0.35); }
.tier-retrieval{ background: rgba(147,197,114,0.18); color: #93C572; border: 1px solid rgba(147,197,114,0.35); }
.tier-generative{background: rgba(251,146,60,0.18); color: #fb923c; border: 1px solid rgba(251,146,60,0.35); }
.tier-none     { background: rgba(239,68,68,0.18);  color: #f87171; border: 1px solid rgba(239,68,68,0.35); }

/* ── Answer box ── */
.answer-box {
    background: rgba(8, 20, 40, 0.7);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 3px solid;
    font-size: 0.92rem;
    line-height: 1.7;
    margin: 1rem 0;
    font-family: 'Space Grotesk', sans-serif;
}

.answer-box-cache     { border-color: #82C8E5; }
.answer-box-retrieval { border-color: #93C572; }
.answer-box-generative{ border-color: #fb923c; }
.answer-box-none      { border-color: #f87171; }

/* ── Trace item ── */
.trace-item {
    background: rgba(8, 20, 40, 0.5);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}

/* ── Section heading ── */
.section-heading {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}

.section-sub {
    font-size: 0.875rem;
    color: rgba(148,163,184,0.8);
    margin-bottom: 1.5rem;
}

/* ── Stat highlight ── */
.highlight-number {
    font-family: 'JetBrains Mono', monospace;
    color: #82C8E5;
    font-weight: 700;
}

/* ── Architecture diagram ── */
.arch-box {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}

.arch-tier {
    background: rgba(14, 30, 60, 0.8);
    border: 1px solid rgba(130,200,229,0.25);
    border-radius: 10px;
    padding: 0.75rem 1.25rem;
    text-align: center;
    min-width: 140px;
}

.arch-tier-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; }
.arch-tier-name  { font-size: 1rem; font-weight: 700; margin: 0.2rem 0; }
.arch-tier-speed { font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; }
.arch-arrow      { color: #82C8E5; font-size: 1.25rem; }

/* ── Step cards (onboarding) ── */
.step-card {
    background: rgba(14, 30, 60, 0.55);
    border: 1px solid rgba(130,200,229,0.18);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}

.step-number {
    background: linear-gradient(135deg, #82C8E5, #93C572);
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    flex-shrink: 0;
    color: #fff;
}

.step-content-title { font-weight: 600; font-size: 0.92rem; color: #e2e8f0; margin-bottom: 0.2rem; }
.step-content-desc  { font-size: 0.82rem; color: #94a3b8; line-height: 1.5; }

/* ── Streamlit overrides ── */
div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }
div[data-testid="stMarkdownContainer"] p { color: #F5F5DC !important; }

.stTextArea textarea {
    background: rgba(8, 20, 40, 0.7) !important;
    border: 1px solid rgba(130,200,229,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.92rem !important;
}

.stTextArea textarea:focus {
    border-color: rgba(130,200,229,0.7) !important;
    box-shadow: 0 0 0 3px rgba(130,200,229,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #82C8E5 0%, #93C572 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(130,200,229,0.35) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(130,200,229,0.5) !important;
}

.stSelectbox > div > div {
    background: rgba(8, 20, 40, 0.7) !important;
    border: 1px solid rgba(130,200,229,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

div[data-testid="stExpander"] {
    background: rgba(14, 30, 60, 0.4) !important;
    border: 1px solid rgba(130,200,229,0.15) !important;
    border-radius: 10px !important;
}

/* ── Tables ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
div[data-testid="stDataFrame"] { border-radius: 10px; }

/* ── Sidebar nav items ── */
div[data-testid="stRadio"] > div { gap: 0.25rem !important; }

div[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 0.6rem 1rem !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

div[data-testid="stRadio"] label:hover {
    background: rgba(130,200,229,0.12) !important;
    color: #e2e8f0 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(130,200,229,0.12) !important;
    margin: 1.5rem 0 !important;
}

/* ── Alert / info boxes ── */
div[data-testid="stInfo"] {
    background: rgba(130,200,229,0.1) !important;
    border: 1px solid rgba(130,200,229,0.25) !important;
    border-radius: 10px !important;
    color: #cceaf4 !important;
}

div[data-testid="stWarning"] {
    background: rgba(251,146,60,0.1) !important;
    border: 1px solid rgba(251,146,60,0.25) !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(130,200,229,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(130,200,229,0.5); }

/* ── Hide streamlit branding, keep sidebar toggle ── */
#MainMenu, footer { visibility: hidden !important; }

/* Make header transparent (do NOT hide it — the sidebar toggle lives inside) */
header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

/* Hide the inner toolbar/decoration elements but leave the toggle button alone */
header[data-testid="stHeader"] > div:not(:has([data-testid="stSidebarCollapseButton"])) {
    visibility: hidden !important;
}

/* Force the toggle button and its icon to always be visible */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    display: flex !important;
    pointer-events: all !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)



# ─── Helpers ──────────────────────────────────────────────────────────

TIER_COLORS = {
    "cache":      "#82C8E5",
    "retrieval":  "#93C572",
    "generative": "#fb923c",
    "none":       "#f87171",
}

TIER_ICONS = {
    "cache":      "⚡",
    "retrieval":  "🔍",
    "generative": "🤖",
    "none":       "⚠️",
}

# Reference metadata about the 5 indexed 3GPP specs — spec number, title,
# what it covers, official archive dir, and the exact version file this
# corpus was scraped from. This is static, real-world reference data that
# the index manifest has no way to know (it tracks the KB file/hash it
# was built from, not per-spec provenance) — it is NOT a substitute for
# chunk counts, which come from count_chunks_by_spec() at render time.
# Titles verified against each spec's own title page (real_kb.jsonl,
# chunk_index=0). Archive URLs verified live (curl, browser UA — 3GPP's
# server 403s a bare curl UA but 200s with one) 2026-08-11. Version
# filenames verified present in database/raw_zips/ 2026-08-11.
SPEC_CATALOG = [
    {
        "spec_id": "36.300",
        "title": "E-UTRA and E-UTRAN Overall Description (Stage 2)",
        "covers": "The system-level map of LTE — how eNodeB, MME, S-GW, and UE fit "
                   "together, and which protocol layer handles what.",
        "archive_url": "https://www.3gpp.org/ftp/Specs/archive/36_series/36.300/",
        "version_zip": "36300-j20.zip",
    },
    {
        "spec_id": "36.331",
        "title": "E-UTRA Radio Resource Control (RRC) Protocol Specification",
        "covers": "UE-to-eNodeB signaling — connection setup, handover, measurement "
                   "reporting, and re-establishment after radio link failure.",
        "archive_url": "https://www.3gpp.org/ftp/Specs/archive/36_series/36.331/",
        "version_zip": "36331-j30.zip",
    },
    {
        "spec_id": "36.413",
        "title": "E-UTRAN S1 Application Protocol (S1AP)",
        "covers": "eNodeB-to-MME signaling — UE context setup, handover requests, "
                   "and paging over the S1 interface.",
        "archive_url": "https://www.3gpp.org/ftp/Specs/archive/36_series/36.413/",
        "version_zip": "36413-j20.zip",
    },
    {
        "spec_id": "24.301",
        "title": "Non-Access-Stratum (NAS) Protocol for EPS",
        "covers": "UE-to-core-network mobility and session management — attach, "
                   "tracking area update, and the EMM/ESM cause codes seen in reject logs.",
        "archive_url": "https://www.3gpp.org/ftp/Specs/archive/24_series/24.301/",
        "version_zip": "24301-k00.zip",
    },
    {
        "spec_id": "23.401",
        "title": "GPRS Enhancements for E-UTRAN Access (EPS Architecture)",
        "covers": "The packet-core architecture underneath LTE — bearer and GTP "
                   "tunnel management, mobility procedures at the system level.",
        "archive_url": "https://www.3gpp.org/ftp/Specs/archive/23_series/23.401/",
        "version_zip": "23401-k00.zip",
    },
]


def tier_badge(tier: str) -> str:
    return f'<span class="tier-badge tier-{tier}">{TIER_ICONS.get(tier,"")} {tier.upper()}</span>'


def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return (f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div>{sub_html}</div>")


def glass_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div style="margin-bottom:1.5rem">
        <div class="section-heading">{title}</div>
        {"<div class='section-sub'>" + subtitle + "</div>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def load_eval_reports() -> dict[str, dict]:
    cfg = get_config()
    log_dir = cfg.resolve_path(cfg.paths.log_dir)
    reports = {}
    if log_dir.exists():
        for f in sorted(log_dir.glob("eval_*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
                reports[f.stem.replace("eval_", "")] = data
            except Exception:
                pass
    return reports


def load_query_log() -> list[dict]:
    cfg = get_config()
    path = cfg.resolve_path(cfg.observability.metrics_file)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().strip().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def load_cache() -> dict:
    cfg = get_config()
    p = cfg.resolve_path(cfg.paths.cache_file)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def load_cache_additions() -> list[dict]:
    cfg = get_config()
    p = cfg.resolve_path(cfg.paths.cache_file).parent / "cache_additions.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().strip().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def count_kb_chunks() -> tuple[int, str]:
    """
    Count lines in the KB file the system is actually configured to search
    — cfg.paths.knowledge_base, resolved via config. Deliberately no
    fallback to whichever *_kb.jsonl happens to exist on disk: reporting a
    different file's doc count than the one retrieval actually indexes
    would silently misrepresent the corpus the system searches.

    Returns (doc_count, filename).
    """
    cfg = get_config()
    p = cfg.resolve_path(cfg.paths.knowledge_base)
    if not p.exists():
        return 0, p.name
    count = sum(1 for l in p.read_text().splitlines() if l.strip())
    return count, p.name


def count_chunks_by_spec() -> dict[str, int]:
    """
    Per-spec_id chunk counts in the CURRENTLY BUILT index — read straight
    from database/indexes/faiss_docs.json (the per-doc file retrieval
    actually searches), not the configured KB file. If the index is
    stale relative to config (see kb_index_mismatch()), this reflects
    what's actually searchable right now, not what's merely configured.

    Returns {} if no index is built yet.
    """
    cfg = get_config()
    docs_path = cfg.resolve_path(cfg.paths.faiss_index_dir) / "faiss_docs.json"
    if not docs_path.exists():
        return {}
    try:
        docs = json.loads(docs_path.read_text())
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for d in docs:
        spec = d.get("spec_id")
        if spec:
            counts[spec] = counts.get(spec, 0) + 1
    return counts


def _sha256_prefix(path: Path, n_bytes: int = 1_000_000) -> str:
    """SHA-256 over the first n_bytes of a file.

    Mirrors scripts/build_index.py's manifest hash (same n_bytes) so a
    hash computed here is directly comparable to manifest["kb_sha256"]
    without re-hashing a potentially multi-MB KB file in full.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def kb_index_mismatch() -> dict | None:
    """
    Detect whether the built indexes were made from a different KB than
    the one currently configured.

    Prefers database/indexes/manifest.json (written by
    scripts/build_index.py): compares kb_filename AND the sha256 of the
    first 1MB against the configured KB. The hash check matters because a
    doc-count-only comparison can't see a same-named file whose *content*
    changed, or catch two genuinely different KBs that happen to have the
    same number of lines.

    Falls back to comparing faiss_docs.json's length against the
    configured KB's line count only when no manifest exists (older
    indexes, or manifest.json was deleted). That fallback is explicitly
    weaker — it's a heuristic, not proof — so the returned dict tags which
    method was used and the caller renders a correspondingly hedged message.

    Returns None if no index is built yet, or if everything matches.
    """
    cfg = get_config()
    index_dir = cfg.resolve_path(cfg.paths.faiss_index_dir)
    configured_kb = cfg.resolve_path(cfg.paths.knowledge_base)
    configured_count, configured_name = count_kb_chunks()

    manifest_path = index_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = None

    if manifest is not None:
        mismatched = manifest.get("kb_filename") != configured_name
        if not mismatched and configured_kb.exists():
            mismatched = manifest.get("kb_sha256") != _sha256_prefix(configured_kb)
        if not mismatched:
            return None
        return {
            "method": "manifest",
            "index_kb": manifest.get("kb_filename", "?"),
            "index_docs": manifest.get("n_docs", "?"),
            "built_at": manifest.get("built_at", "?"),
            "configured_kb": configured_name,
            "configured_docs": configured_count,
        }

    # No manifest (older indexes) — fall back to the weaker doc-count-only
    # heuristic that used to be the only check available.
    docs_path = index_dir / "faiss_docs.json"
    if not docs_path.exists():
        return None
    try:
        index_count = len(json.loads(docs_path.read_text()))
    except Exception:
        return None
    if index_count == configured_count:
        return None
    return {
        "method": "doc_count",
        "index_docs": index_count,
        "configured_kb": configured_name,
        "configured_docs": configured_count,
    }


def _decorators_mark_slow_or_integration(decorators: list) -> bool:
    for dec in decorators:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr in ("slow", "integration"):
            return True
    return False


def count_tests() -> tuple[int, int]:
    """
    Count test functions across tests/test_*.py via source AST.

    Every test in this project lives inside a test class, so a naive
    "\\ndef test_" substring count (or a regex anchored at column 0) misses
    all of them — they're indented. Parsing the AST sidesteps indentation
    entirely and also lets us split the total into "fast" vs "slow" by
    checking for @pytest.mark.slow / @pytest.mark.integration on the
    function itself or its enclosing class (marks apply to every method
    in a marked class — see tests/test_retrieval_tier.py's TestEmbedder,
    TestDenseRetriever, TestReranker, TestRetrievalTier and
    tests/test_generative_tier.py's TestGroqIntegration).

    Returns (fast_count, total_count) — "fast" mirrors what
    `pytest -m "not slow and not integration"` actually selects.
    """
    test_dir = PROJECT_ROOT / "tests"
    if not test_dir.exists():
        return 0, 0

    total = 0
    fast = 0
    for f in test_dir.glob("test_*.py"):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_marked = _decorators_mark_slow_or_integration(node.decorator_list)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                        total += 1
                        if not class_marked and not _decorators_mark_slow_or_integration(item.decorator_list):
                            fast += 1
            elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                total += 1
                if not _decorators_mark_slow_or_integration(node.decorator_list):
                    fast += 1
    return fast, total


def plotly_fig_base() -> dict:
    """Base layout for all plotly charts matching the dark glass theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,20,40,0.4)",
        font=dict(family="Space Grotesk, sans-serif", color="#94a3b8"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor="rgba(130,200,229,0.1)", zerolinecolor="rgba(130,200,229,0.15)"),
        yaxis=dict(gridcolor="rgba(130,200,229,0.1)", zerolinecolor="rgba(130,200,229,0.15)"),
    )


# ─── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 1.75rem 1rem 1.25rem 1rem; border-bottom: 1px solid rgba(130,200,229,0.15); margin-bottom: 1rem;">
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.4rem;">
            <span style="font-size:1.6rem;">📡</span>
            <span style="font-size:1.35rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.02em;">TeleTriage</span>
        </div>
        <div style="font-size:0.75rem; color:#64748b; letter-spacing:0.05em; padding-left:2.4rem;">
            HIERARCHICAL FAULT TRIAGE ENGINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview", "⚡  Live Query", "📊  Eval Results", "🗂️  Query Log", "🔍  Cache Inspector"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 1rem; margin: 0 0.25rem; background: rgba(130,200,229,0.06);
                border: 1px solid rgba(130,200,229,0.15); border-radius: 12px;">
        <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                    color:#64748b; margin-bottom:0.75rem;">System Status</div>
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="width:7px; height:7px; background:#82C8E5; border-radius:50%;
                         box-shadow:0 0 6px #82C8E5; display:inline-block;"></span>
            <span style="font-size:0.8rem; color:#94a3b8;">CAG Cache</span>
            <span style="margin-left:auto; font-size:0.75rem; font-family:monospace; color:#82C8E5;">~0ms</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="width:7px; height:7px; background:#93C572; border-radius:50%;
                         box-shadow:0 0 6px #93C572; display:inline-block;"></span>
            <span style="font-size:0.8rem; color:#94a3b8;">RAG Retrieval</span>
            <span style="margin-left:auto; font-size:0.75rem; font-family:monospace; color:#93C572;">~200ms</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <span style="width:7px; height:7px; background:#fb923c; border-radius:50%;
                         box-shadow:0 0 6px #fb923c; display:inline-block;"></span>
            <span style="font-size:0.8rem; color:#94a3b8;">Generative LLM</span>
            <span style="margin-left:auto; font-size:0.75rem; font-family:monospace; color:#fb923c;">~1.1s</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 0 0.5rem;">
        <div style="font-size:0.68rem; color:#334155; text-align:center;">
            v0.2.0 · MIT License · Arya Doshi
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════

if "Overview" in page:

    # Load data once at the top of the page — used by both cascade diagram and metrics row
    cache = load_cache()
    reports = load_eval_reports()
    kb_chunks, kb_filename = count_kb_chunks()
    n_tests_fast, n_tests_total = count_tests()

    # ── Centered title ────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center; font-size:2.8rem; font-weight:900; "
        "letter-spacing:-0.03em; color:#f1f5f9; margin:2rem 0 0.4rem 0;'>📡 TeleTriage</h1>"
        "<p style='text-align:center; font-size:1rem; color:#64748b; margin-bottom:2.5rem;'>"
        "AI-powered hierarchical fault-resolution engine for telecom networks</p>"
        "<hr>",
        unsafe_allow_html=True,
    )

    # ── Warning: indexes stale relative to configured KB ────────────────
    _mismatch = kb_index_mismatch()
    if _mismatch is not None:
        if _mismatch["method"] == "manifest":
            st.warning(
                f"⚠️ Indexes were built from a different KB than the one configured. "
                f"Run: `uv run python scripts/build_index.py`\n\n"
                f"Index was built from **{_mismatch['index_kb']}** "
                f"({_mismatch['index_docs']} docs, built {_mismatch['built_at']}) — "
                f"configured KB is **{_mismatch['configured_kb']}** "
                f"({_mismatch['configured_docs']:,} docs)."
            )
        else:
            st.warning(
                f"⚠️ Indexes were built from a different KB than the one configured "
                f"(doc-count check only — no manifest.json found; rebuild to get a "
                f"reliable check). Run: `uv run python scripts/build_index.py`\n\n"
                f"Index contains **{_mismatch['index_docs']:,}** docs — configured KB "
                f"(`{_mismatch['configured_kb']}`) has **{_mismatch['configured_docs']:,}**."
            )

    cfg = get_config()

    # ── 1. THE PROBLEM ──────────────────────────────────────────────────
    st.markdown(
        "<div style='margin-bottom:2rem;'>"
        "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#82C8E5;margin-bottom:0.6rem;'>The Problem</div>"
        "<div style='font-size:2.2rem;font-weight:800;line-height:1.2;letter-spacing:-0.03em;color:#f1f5f9;margin-bottom:0.9rem;'>"
        "3GPP specs are the standards every network is built to comply with —<br>"
        "<span style='color:#82C8E5;'>and they're nearly unsearchable.</span></div>"
        "<div style='font-size:1rem;color:#94a3b8;line-height:1.8;'>"
        "They're distributed as Word documents running to 1,000+ pages of dense normative prose, heavily "
        "cross-referenced, with no good search. When a network operations engineer sees an unfamiliar reject "
        "cause in a log at 2am, finding out what it means requires grepping a 900-page document — and often "
        "the answer lives in a <em>different</em> spec than the one naming the code. TS 36.331 alone is "
        "<span class='highlight-number'>3.2 million</span> characters."
        "</div></div><hr>",
        unsafe_allow_html=True,
    )

    # ── 2. WHAT THIS DOES ───────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
                    color: #82C8E5; margin-bottom: 0.6rem;">What This Does</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem; letter-spacing: -0.02em;">
            Describe the symptom in plain language. Get an answer grounded in the actual spec text — with a citation.
        </div>
        <div style="font-size: 0.95rem; color: #94a3b8; line-height: 1.8; margin-bottom: 1.5rem;">
            Type what you're seeing, the way you'd describe it to a colleague — a fault description, a reject
            cause, a log line. TeleTriage searches the indexed specifications, finds the passages that actually
            address it, and returns a structured answer that cites the exact spec and chunk it came from, so you
            can go verify it yourself instead of taking a model's word for it.
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # ── 3. THE CORPUS — chunk counts computed from the built index, never
    # hardcoded; "built from"/"built at" come straight from the manifest ──
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
                    color: #82C8E5; margin-bottom: 0.6rem;">The Corpus</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem; letter-spacing: -0.02em;">
            Five live 3GPP specifications, chunked and indexed.
        </div>
    </div>
    """, unsafe_allow_html=True)

    spec_counts = count_chunks_by_spec()
    _manifest = load_index_manifest(cfg.resolve_path(cfg.paths.faiss_index_dir))
    if _manifest:
        try:
            _built_dt = datetime.fromisoformat(_manifest.get("built_at", "").replace("Z", "+00:00"))
            _built_at_fmt = _built_dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            _built_at_fmt = _manifest.get("built_at") or "—"
        _manifest_note = (
            f"Index built from <strong style='color:#F5F5DC;'>{_manifest.get('kb_filename', '—')}</strong>"
            f" &nbsp;·&nbsp; <span class='highlight-number'>{_manifest.get('n_docs', 0):,}</span> chunks total"
            f" &nbsp;·&nbsp; built <strong style='color:#F5F5DC;'>{_built_at_fmt}</strong>"
        )
    else:
        _manifest_note = "No index manifest found — run <code>uv run python scripts/build_index.py</code>."
    st.markdown(f"<div style='font-size:0.82rem;color:#64748b;margin-bottom:1rem;'>{_manifest_note}</div>", unsafe_allow_html=True)

    _rows_html = ""
    for spec in SPEC_CATALOG:
        sid = spec["spec_id"]
        n_chunks = spec_counts.get(sid)
        n_chunks_str = f"{n_chunks:,}" if n_chunks else "—"
        _rows_html += (
            "<tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>"
            "<td style='padding:0.75rem 1rem;vertical-align:top;'>"
            f"<div style='font-weight:700;color:#F5F5DC;font-family:JetBrains Mono,monospace;font-size:0.85rem;'>TS {sid}</div>"
            f"<div style='font-size:0.75rem;color:#64748b;margin-top:0.15rem;max-width:220px;'>{spec['title']}</div>"
            "</td>"
            f"<td style='padding:0.75rem 1rem;vertical-align:top;color:#94a3b8;font-size:0.82rem;max-width:340px;'>{spec['covers']}</td>"
            f"<td style='padding:0.75rem 1rem;vertical-align:top;text-align:right;font-family:JetBrains Mono,monospace;color:#82C8E5;font-size:0.9rem;white-space:nowrap;'>{n_chunks_str}</td>"
            "<td style='padding:0.75rem 1rem;vertical-align:top;white-space:nowrap;'>"
            f"<a href='{spec['archive_url']}' target='_blank' rel='noopener' "
            "style='color:#82C8E5;text-decoration:none;border-bottom:1px dotted rgba(130,200,229,0.5);font-size:0.82rem;'>3GPP archive &#8599;</a>"
            f"<div style='font-size:0.68rem;color:#475569;margin-top:0.25rem;font-family:JetBrains Mono,monospace;'>{spec['version_zip']}</div>"
            "</td></tr>"
        )

    st.markdown(
        "<div class='glass-card' style='padding:0.5rem 0.5rem;'>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='border-bottom:1px solid rgba(130,200,229,0.3);'>"
        "<th style='text-align:left;padding:0.6rem 1rem;color:#82C8E5;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;'>Spec</th>"
        "<th style='text-align:left;padding:0.6rem 1rem;color:#82C8E5;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;'>What it covers</th>"
        "<th style='text-align:right;padding:0.6rem 1rem;color:#82C8E5;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;'>Chunks indexed</th>"
        "<th style='text-align:left;padding:0.6rem 1rem;color:#82C8E5;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;'>Archive</th>"
        "</tr></thead><tbody>"
        + _rows_html +
        "</tbody></table></div>"
        "<div style='font-size:0.76rem;color:#475569;margin-top:0.75rem;line-height:1.6;'>"
        "3GPP distributes these as a ZIP containing a single <code>.docx</code> file — not PDF. The version "
        "filenames above are exactly what this index was built from. 3GPP revises these specifications "
        "continuously, so re-running the scraper may fetch a newer revision than what's indexed today."
        "</div><hr>",
        unsafe_allow_html=True,
    )

    # ── 4. HOW IT ANSWERS ───────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
                    color: #82C8E5; margin-bottom: 0.6rem;">How It Answers</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem; letter-spacing: -0.02em;">
            A three-tier cascade — each tier only answers what it can back up.
        </div>
        <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.7; margin-bottom: 1.5rem;">
            TeleTriage tries the cheapest, most verifiable path first, and escalates only when confidence falls
            below an empirically measured threshold — not a guessed one.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cascade diagram — no blank lines inside (blank lines trigger Markdown code-block mode)
    _cache_n = len(cache)
    _kb_n = sum(spec_counts.values()) if spec_counts else kb_chunks
    _kb_str = f"{_kb_n:,}" if _kb_n else "—"
    t1 = (f"<div style='flex:1;min-width:160px;padding:1.25rem;border-right:1px solid rgba(130,200,229,0.12);'>"
          f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem;'>Tier 1 · Known repeats</div>"
          f"<div style='font-size:1.25rem;font-weight:800;color:#82C8E5;margin-bottom:0.3rem;'>&#9889; Cache</div>"
          f"<div style='font-size:0.78rem;color:#94a3b8;line-height:1.5;margin-bottom:0.75rem;'>Exact &#8594; normalised &#8594; fuzzy match against {_cache_n} hand-curated Q&amp;A pairs. If this exact fault has been triaged before, you get the same vetted answer back — no LLM call, ~0ms.</div>"
          f"<div style='font-size:0.72rem;background:rgba(130,200,229,0.1);border:1px solid rgba(130,200,229,0.2);border-radius:6px;padding:4px 10px;display:inline-block;color:#82C8E5;'>Confidence &ge; 0.85 &#8594; stop</div></div>")
    arr = "<div style='display:flex;align-items:center;padding:0 0.75rem;color:#334155;font-size:0.9rem;font-weight:700;'>&#8250;</div>"
    t2 = (f"<div style='flex:1;min-width:160px;padding:1.25rem;border-right:1px solid rgba(130,200,229,0.12);'>"
          f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem;'>Tier 2 · Grounded, with citations</div>"
          f"<div style='font-size:1.25rem;font-weight:800;color:#93C572;margin-bottom:0.3rem;'>&#128269; Retrieval (RAG)</div>"
          f"<div style='font-size:0.78rem;color:#94a3b8;line-height:1.5;margin-bottom:0.75rem;'>Hybrid BM25 + dense search across {_kb_str} indexed 3GPP chunks, reranked by a cross-encoder, synthesized into an answer that cites the spec and chunk it came from. Corpus covers it &#8594; sourced answer, not a guess.</div>"
          f"<div style='font-size:0.72rem;background:rgba(147,197,114,0.1);border:1px solid rgba(147,197,114,0.2);border-radius:6px;padding:4px 10px;display:inline-block;color:#93C572;'>Rerank logit &ge; 0.20 &#8594; stop</div></div>")
    t3 = ("<div style='flex:1;min-width:160px;padding:1.25rem;'>"
          "<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem;'>Tier 3 · Outside the corpus</div>"
          "<div style='font-size:1.25rem;font-weight:800;color:#fb923c;margin-bottom:0.3rem;'>&#129302; Generative</div>"
          "<div style='font-size:0.78rem;color:#94a3b8;line-height:1.5;margin-bottom:0.75rem;'>Closed-book: a general-purpose LLM (Groq Llama 3.3 70B &#8594; Gemini Flash &#8594; local Qwen) answers from its own training knowledge — no citation, because there's nothing in the corpus to cite. Always answers, so the system never just fails.</div>"
          "<div style='font-size:0.72rem;background:rgba(251,146,60,0.1);border:1px solid rgba(251,146,60,0.2);border-radius:6px;padding:4px 10px;display:inline-block;color:#fb923c;'>Always answers</div></div>")
    footer = ("<div style='margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(130,200,229,0.1);font-size:0.78rem;color:#475569;line-height:1.6;'>"
              "Thresholds are <span style='color:#82C8E5;'>empirically calibrated</span> on 20 held-out queries via a three-point ablation curve — not guessed. "
              "The system is designed to know the boundary of what it knows: tiers 1&ndash;2 only answer when they're confident and grounded, and tier 3's answers are "
              "never presented as sourced from the spec corpus, because they aren't.</div>")
    st.markdown(
        f"<div class='glass-card' style='margin-bottom:1.5rem;'><div style='display:flex;align-items:stretch;gap:0;flex-wrap:wrap;'>{t1}{arr}{t2}{arr}{t3}</div>{footer}</div>",
        unsafe_allow_html=True,
    )

    # Key results — all values derived from actual files (loaded at page top)
    best = max(reports.values(), key=lambda r: r.get("overall", {}).get("bertscore_f1_mean", 0)) if reports else {}
    best_bs = best.get("overall", {}).get("bertscore_f1_mean", 0)
    best_p50 = best.get("overall", {}).get("p50_sec", 0)

    retrieval_p50_label = f"{best_p50*1000:.0f} ms" if best_p50 else "—"
    kb_label = f"{kb_chunks:,}" if kb_chunks else "—"
    kb_sub = f"{kb_filename} · {kb_chunks:,} docs" if kb_chunks else kb_filename
    test_label = str(n_tests_fast) if n_tests_total else "—"
    test_sub = f"{n_tests_fast} fast · {n_tests_total} total" if n_tests_total else "Fast suite, no regressions"

    st.markdown(
        "<div class='metric-row'>"
        + metric_card("Best BERTScore", f"{best_bs:.3f}" if best_bs else "—", "On 20 held-out queries")
        + metric_card("Cache Entries", str(len(cache)), "Hand-curated Q&A pairs")
        + metric_card("Retrieval p50", retrieval_p50_label, "Embed + rerank latency")
        + metric_card("Fast Suite Tests", test_label, test_sub)
        + metric_card("KB Chunks", kb_label, kb_sub)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("""<hr>""", unsafe_allow_html=True)

    # ── Tools Used ────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
                color: #82C8E5; margin-bottom: 0.6rem;">Tools &amp; Technology</div>
    <div style="font-size: 1.35rem; font-weight: 700; color: #f1f5f9; margin-bottom: 1.25rem; letter-spacing: -0.02em;">
        Every component chosen for a reason, benchmarked, not assumed.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
                        color:#93C572; margin-bottom:0.75rem;">Retrieval Stack</div>
            <div style="display:flex; flex-direction:column; gap:0.5rem; font-size:0.82rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Sparse</span>
                    <span style="color:#93C572; font-family:monospace; font-size:0.78rem;">BM25Okapi</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Embeddings</span>
                    <span style="color:#93C572; font-family:monospace; font-size:0.78rem;">BGE-small-en-v1.5</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Vector index</span>
                    <span style="color:#93C572; font-family:monospace; font-size:0.78rem;">FAISS IndexFlatIP</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Fusion</span>
                    <span style="color:#93C572; font-family:monospace; font-size:0.78rem;">RRF  k=60</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Cross-encoder</span>
                    <span style="color:#93C572; font-family:monospace; font-size:0.78rem;">BGE-reranker-base</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
                        color:#fb923c; margin-bottom:0.75rem;">Generation Stack</div>
            <div style="display:flex; flex-direction:column; gap:0.5rem; font-size:0.82rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Primary LLM</span>
                    <span style="color:#fb923c; font-family:monospace; font-size:0.78rem;">Llama 3.3 70B</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Fallback</span>
                    <span style="color:#fb923c; font-family:monospace; font-size:0.78rem;">Gemini Flash</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Local fallback</span>
                    <span style="color:#fb923c; font-family:monospace; font-size:0.78rem;">Qwen 2.5 1.5B</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Inference API</span>
                    <span style="color:#fb923c; font-family:monospace; font-size:0.78rem;">Groq (free tier)</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Cache dedup</span>
                    <span style="color:#fb923c; font-family:monospace; font-size:0.78rem;">RapidFuzz Levenshtein</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
                        color:#82C8E5; margin-bottom:0.75rem;">Data &amp; Eval</div>
            <div style="display:flex; flex-direction:column; gap:0.5rem; font-size:0.82rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">KB source</span>
                    <span style="color:#82C8E5; font-family:monospace; font-size:0.78rem;">5 live 3GPP specs</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Dedup</span>
                    <span style="color:#82C8E5; font-family:monospace; font-size:0.78rem;">MinHash LSH ≥0.85</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Quality metric</span>
                    <span style="color:#82C8E5; font-family:monospace; font-size:0.78rem;">BERTScore F1</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Synth Q&amp;A</span>
                    <span style="color:#82C8E5; font-family:monospace; font-size:0.78rem;">Llama 3.3 70B</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Test suite</span>
                    <span style="color:#82C8E5; font-family:monospace; font-size:0.78rem;">142 tests · pytest</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════
# PAGE: LIVE QUERY
# ═══════════════════════════════════════════════════════════════════════

elif "Live Query" in page:
    glass_header(
        "⚡ Live Query",
        "Run any telecom fault description through the full CAG → RAG → Generative cascade."
    )

    # ── Example pills — clicking auto-fills the text area ─────────────
    # session_state key "query_text" is shared with the text_area below
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = ""

    PILL_EXAMPLES = [
        ("⚡", "high packet loss",                               "cache"),
        ("⚡", "handover failure",                               "cache"),
        ("⚡", "volte call failure",                             "cache"),
        ("🔍", "eNB not generating handover despite A3 event",   "retrieval"),
        ("🔍", "S1-AP reset PDU storm from one eNB",             "retrieval"),
        ("🤖", "PDCP SN wrap-around during extended coverage",   "generative"),
        ("🤖", "eSIM LPA profile download 403 error",            "generative"),
    ]

    PILL_COLORS = {
        "cache":      ("rgba(130,200,229,0.12)",  "rgba(130,200,229,0.3)",  "#82C8E5"),
        "retrieval":  ("rgba(147,197,114,0.12)", "rgba(147,197,114,0.3)", "#93C572"),
        "generative": ("rgba(251,146,60,0.12)", "rgba(251,146,60,0.3)", "#fb923c"),
    }

    st.markdown(
        "<div style='font-size:0.75rem;color:#64748b;margin-bottom:0.5rem;'>Try an example — click to fill:</div>",
        unsafe_allow_html=True,
    )

    # Render pills as real Streamlit buttons in two rows via columns
    # CSS hack: override small pill-column buttons inline via style injection
    st.markdown("""<style>
    div[data-testid="column"] .stButton>button {
        padding:3px 14px !important;
        border-radius:20px !important;
        font-size:0.77rem !important;
        font-weight:500 !important;
        height:auto !important;
        min-height:0 !important;
        line-height:1.5 !important;
        box-shadow:none !important;
        background:transparent !important;
        border:1px solid rgba(130,200,229,0.35) !important;
        color:#94a3b8 !important;
        transform:none !important;
        transition:background 0.15s,color 0.15s !important;
    }
    div[data-testid="column"] .stButton>button:hover {
        background:rgba(130,200,229,0.12) !important;
        color:#e2e8f0 !important;
        border-color:rgba(130,200,229,0.6) !important;
    }
    </style>""", unsafe_allow_html=True)

    for row_start in range(0, len(PILL_EXAMPLES), 4):
        batch = PILL_EXAMPLES[row_start:row_start + 4]
        cols = st.columns(len(batch))
        for col, (icon, label, tier) in zip(cols, batch):
            bg, bd, fg = PILL_COLORS[tier]
            col.markdown(
                f"<style>div[data-testid='column']:nth-child({list(cols).index(col)+1}) "
                f".stButton>button{{background:{bg}!important;border-color:{bd}!important;"
                f"color:{fg}!important;}}</style>",
                unsafe_allow_html=True,
            )
            with col:
                if st.button(f"{icon} {label}", key=f"pill_{label[:25]}",
                             width='stretch'):
                    st.session_state["query_text"] = label
                    st.rerun()

    st.markdown(
        "<div style='font-size:0.72rem;color:#334155;margin:0.5rem 0 0.75rem 0;'>"
        "🔵 Blue = likely cache &nbsp;·&nbsp; 🟢 Green = likely retrieval (RAG, cited) &nbsp;·&nbsp; 🟠 Orange = likely generative"
        "</div>",
        unsafe_allow_html=True,
    )

    query_text = st.text_area(
        "Fault description",
        placeholder="e.g. UE keeps reselecting to 3G even when LTE coverage is adequate",
        height=100,
        key="query_text",
        label_visibility="collapsed",
    )

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        run = st.button("Run Query →", width='stretch')
    with col_hint:
        st.markdown(
            "<div style='padding:0.6rem 0;font-size:0.8rem;color:#475569;'>"
            "Type any fault description above and hit Run. The cascade tries Cache first, "
            "then Retrieval, then Generative — stopping as soon as confidence is high enough."
            "</div>",
            unsafe_allow_html=True,
        )

    # query_text is bound to st.session_state["query_text"] via key=
    if run and st.session_state.get("query_text", "").strip():
        query_text = st.session_state["query_text"].strip()
        with st.spinner("Running cascade..."):
            try:
                from backend.config import get_config
                from backend.observability import setup_logging
                from backend.router import Router
                from backend.types import Query

                setup_logging()
                cfg = get_config()
                router = Router()
                response = router.route(Query(text=query_text.strip()))

                # Log the query
                log_path = cfg.resolve_path(cfg.observability.metrics_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "query": response.query.text,
                    "answered_by": response.answered_by.value,
                    "confidence": round(response.confidence, 4),
                    "latency_ms": round(response.total_latency_sec * 1000, 1),
                    "tiers_tried": [t.tier.value for t in response.tier_trace],
                }
                with log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")

                tier = response.answered_by.value
                tier_color = TIER_COLORS.get(tier, "#94a3b8")

                # Result metrics
                st.markdown(
                    "<div class='metric-row' style='margin-top:1.5rem'>"
                    + metric_card("Answered By", f"{TIER_ICONS.get(tier,'')} {tier.upper()}", "Tier that responded")
                    + metric_card("Confidence", f"{response.confidence:.0%}", "Tier-reported confidence")
                    + metric_card("Latency", f"{response.total_latency_sec*1000:.0f}ms", "End-to-end wall-clock")
                    + metric_card("Tiers Tried", str(len(response.tier_trace)), "Before final answer")
                    + "</div>",
                    unsafe_allow_html=True,
                )

                # Answer box
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
                        <span style="font-size:0.8rem; font-weight:600; color:#94a3b8; text-transform:uppercase;
                                     letter-spacing:0.08em;">Answer</span>
                        {tier_badge(tier)}
                    </div>
                    <div class="answer-box answer-box-{tier}" style="border-color:{tier_color}">
                        {response.answer.replace(chr(10), '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Tier trace
                with st.expander("🔬 Tier trace — full cascade detail"):
                    for t in response.tier_trace:
                        tc = TIER_COLORS.get(t.tier.value, "#94a3b8")
                        answered_str = "✓ answered" if t.answered else "→ delegated"
                        details_json = json.dumps(t.details, indent=2)
                        st.markdown(f"""
                        <div class="trace-item">
                            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.4rem;">
                                {tier_badge(t.tier.value)}
                                <span style="color:{tc}; font-weight:600;">{answered_str}</span>
                                <span style="color:#64748b; margin-left:auto;">
                                    conf={t.confidence:.3f} &nbsp;·&nbsp; {t.latency_sec*1000:.1f}ms
                                </span>
                            </div>
                            <pre style="margin:0; color:#64748b; font-size:0.75rem; white-space:pre-wrap;">{details_json}</pre>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error running query: {e}")
                st.info("Make sure indexes are built: `uv run python scripts/build_index.py`")

    elif run:
        st.warning("Please enter a fault description above.")

    # Quick example buttons
    st.markdown("""<hr>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem; font-weight:600; color:#64748b; margin-bottom:0.75rem;
                text-transform:uppercase; letter-spacing:0.08em;">Quick examples</div>
    """, unsafe_allow_html=True)

    eg_cols = st.columns(3)
    examples = [
        ("⚡ Cache hit", "high packet loss"),
        ("⚡ Cache hit", "handover failure"),
        ("🤖 Generative", "NB-IoT RACH failure despite adequate coverage"),
    ]
    for col, (label, q) in zip(eg_cols, examples):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="padding:0.875rem 1rem; cursor:pointer;">
                <div style="font-size:0.68rem; color:#64748b; text-transform:uppercase;
                             letter-spacing:0.08em; margin-bottom:0.3rem;">{label}</div>
                <div style="font-size:0.82rem; color:#F5F5DC;">{q}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: EVAL RESULTS
# ═══════════════════════════════════════════════════════════════════════

elif "Eval Results" in page:
    glass_header(
        "📊 Evaluation Results",
        "All evaluation runs on the 20 held-out telecom fault queries."
    )

    reports = load_eval_reports()
    if not reports:
        st.info("No evaluation reports found. Run: `uv run python scripts/run_eval.py`")
    else:
        # Report selector
        selected_key = st.selectbox(
            "Select report",
            list(reports.keys()),
            format_func=lambda k: f"{k}  —  BERTScore {reports[k].get('overall',{}).get('bertscore_f1_mean',0):.3f}",
            label_visibility="collapsed",
        )
        report = reports[selected_key]
        overall = report.get("overall", {})
        tier_dist = report.get("tier_distribution", {})
        per_tier = report.get("per_tier", {})

        # Overall metrics
        st.markdown(
            "<div class='metric-row'>"
            + metric_card("ROUGE-L", f"{overall.get('rouge_l_mean',0):.3f}", f"median {overall.get('rouge_l_median',0):.3f}")
            + metric_card("BERTScore F1", f"{overall.get('bertscore_f1_mean',0):.3f}", f"median {overall.get('bertscore_f1_median',0):.3f}")
            + metric_card("p50 Latency", f"{overall.get('p50_sec',0)*1000:.0f}ms", f"p95 {overall.get('p95_sec',0)*1000:.0f}ms")
            + metric_card("Queries", str(report.get("n_queries", 20)), "Held-out test set")
            + "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "**BERTScore F1** measures semantic similarity using contextual embeddings — "
            "not word overlap (ROUGE). A score of 0.80+ means the answer is semantically "
            "equivalent to the reference, even if phrased differently. "
            "ROUGE-L is reported for completeness; it under-scores the KB's dense 3GPP notation."
        )

        # Tier distribution donut + per-tier quality
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("""
            <div style="font-size:0.85rem; font-weight:600; color:#94a3b8; margin-bottom:0.75rem;
                        text-transform:uppercase; letter-spacing:0.08em;">Tier Distribution</div>
            """, unsafe_allow_html=True)
            tiers_with_data = {k: v for k, v in tier_dist.items() if v.get("count", 0) > 0}
            if tiers_with_data:
                labels = list(tiers_with_data.keys())
                values = [v["count"] for v in tiers_with_data.values()]
                colors = [TIER_COLORS.get(l, "#94a3b8") for l in labels]
                fig = go.Figure(go.Pie(
                    labels=[f"{TIER_ICONS.get(l,'')} {l}" for l in labels],
                    values=values,
                    hole=0.6,
                    marker=dict(colors=colors, line=dict(color="#050d1a", width=2)),
                    textfont=dict(family="Space Grotesk", size=11),
                ))
                fig.update_layout(**plotly_fig_base(), height=220, showlegend=True,
                                  legend=dict(orientation="h", y=-0.1, font=dict(size=10)))
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

        with col2:
            st.markdown("""
            <div style="font-size:0.85rem; font-weight:600; color:#94a3b8; margin-bottom:0.75rem;
                        text-transform:uppercase; letter-spacing:0.08em;">BERTScore by Tier</div>
            """, unsafe_allow_html=True)
            tier_names = list(per_tier.keys())
            bs_values = [per_tier[t].get("bertscore_f1_mean", 0) for t in tier_names]
            if tier_names:
                fig = go.Figure(go.Bar(
                    x=[f"{TIER_ICONS.get(t,'')} {t}" for t in tier_names],
                    y=bs_values,
                    marker_color=[TIER_COLORS.get(t, "#94a3b8") for t in tier_names],
                    marker_line_color="rgba(0,0,0,0)",
                    text=[f"{v:.3f}" for v in bs_values],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=11, color="#e2e8f0"),
                ))
                fig.update_layout(**plotly_fig_base(), height=220, yaxis_range=[0.6, 0.9])
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

        with col3:
            st.markdown("""
            <div style="font-size:0.85rem; font-weight:600; color:#94a3b8; margin-bottom:0.75rem;
                        text-transform:uppercase; letter-spacing:0.08em;">Latency Percentiles</div>
            """, unsafe_allow_html=True)
            pctls = ["p50", "p95", "p99"]
            vals = [overall.get(f"{p}_sec", 0) * 1000 for p in pctls]
            fig = go.Figure(go.Bar(
                x=pctls, y=vals,
                marker_color=["#82C8E5", "#93C572", "#fb923c"],
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{v:.0f}ms" for v in vals],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#e2e8f0"),
            ))
            fig.update_layout(**plotly_fig_base(), height=220)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

        # Calibration history table
        if len(reports) > 1:
            st.markdown("""<hr>""", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.85rem; font-weight:600; color:#94a3b8; margin-bottom:0.75rem;
                        text-transform:uppercase; letter-spacing:0.08em;">Calibration History — All Runs</div>
            """, unsafe_allow_html=True)
            best_ts = max(reports, key=lambda k: reports[k].get("overall", {}).get("bertscore_f1_mean", 0))
            rows = []
            for ts, r in reports.items():
                ov = r.get("overall", {})
                td = r.get("tier_distribution", {})
                rows.append({
                    "Run": f"★ {ts}" if ts == best_ts else ts,
                    "ROUGE-L": f"{ov.get('rouge_l_mean',0):.4f}",
                    "BERTScore": f"{ov.get('bertscore_f1_mean',0):.4f}",
                    "p50 (ms)": f"{ov.get('p50_sec',0)*1000:.0f}",
                    "Retrieval %": f"{td.get('retrieval',{}).get('pct',0):.0f}%",
                    "Generative %": f"{td.get('generative',{}).get('pct',0):.0f}%",
                    "Cache %": f"{td.get('cache',{}).get('pct',0):.0f}%",
                })
            df = pd.DataFrame(rows)
            # Highlight the best row (★ prefix) with a gold background via pandas Styler
            def _highlight_best(row: pd.Series) -> list[str]:
                if str(row["Run"]).startswith("★"):
                    return ["background-color:rgba(250,204,21,0.12);color:#fbbf24;font-weight:600"] * len(row)
                return [""] * len(row)
            st.dataframe(df.style.apply(_highlight_best, axis=1), width='stretch', hide_index=True)

        # Per-query breakdown
        st.markdown("""<hr>""", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.85rem; font-weight:600; color:#94a3b8; margin-bottom:0.75rem;
                    text-transform:uppercase; letter-spacing:0.08em;">Per-Query Breakdown</div>
        """, unsafe_allow_html=True)

        pq = report.get("per_query", [])
        if pq:
            none_count = sum(1 for q in pq if q.get("answered_by") == "none")
            if none_count:
                st.markdown(
                    f"<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);"
                    f"border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.75rem;font-size:0.82rem;'>"
                    f"<strong style='color:#f87171;'>⚠ {none_count} query{'ies' if none_count>1 else ''} returned tier = none.</strong> "
                    f"This means every tier exceeded its confidence threshold and the router had no fallback "
                    f"answer to return — typically caused by missing API keys (Groq/Gemini) or indexes not built. "
                    f"Run <code>uv run python scripts/build_index.py</code> and verify <code>.env</code>.</div>",
                    unsafe_allow_html=True,
                )
            rows = []
            for q in pq:
                tier_val = q.get("answered_by", "")
                tier_display = tier_val if tier_val != "none" else "⚠ none"
                rows.append({
                    "ID": q.get("query_id", ""),
                    "Query": q.get("query_text", "")[:60] + "…" if len(q.get("query_text","")) > 60 else q.get("query_text",""),
                    "Tier": tier_display,
                    "Conf": f"{q.get('confidence',0):.3f}",
                    "ROUGE-L": f"{q.get('rouge_l',0):.4f}",
                    "BERTScore": f"{q.get('bertscore_f1',0):.4f}",
                    "Latency ms": f"{q.get('latency_ms',0):.0f}",
                    "Tier Match": "✓" if q.get("tier_match") else "✗",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, width='stretch', hide_index=True)

            csv = df.to_csv(index=False)
            st.download_button(
                "⬇ Download CSV",
                csv,
                f"teletriage_eval_{selected_key}.csv",
                "text/csv",
            )


# ═══════════════════════════════════════════════════════════════════════
# PAGE: QUERY LOG
# ═══════════════════════════════════════════════════════════════════════

elif "Query Log" in page:
    glass_header(
        "🗂️ Query Log",
        "Every query run through Live Query, logged automatically."
    )

    rows = load_query_log()

    if not rows:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">📋</div>
            <div style="font-size:1rem; font-weight:600; color:#94a3b8; margin-bottom:0.5rem;">
                No queries logged yet
            </div>
            <div style="font-size:0.85rem; color:#475569; max-width:400px; margin:0 auto;">
                Run some queries in the <strong style="color:#82C8E5">Live Query</strong> tab.
                Each query is automatically appended to <code>outputs/query_metrics.jsonl</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Summary metrics
        tier_counts = df["answered_by"].value_counts()
        avg_conf = df["confidence"].mean()
        avg_lat = df["latency_ms"].mean()

        st.markdown(
            "<div class='metric-row'>"
            + metric_card("Total Queries", str(len(df)), "Logged sessions")
            + metric_card("Avg Confidence", f"{avg_conf:.0%}", "Across all tiers")
            + metric_card("Avg Latency", f"{avg_lat:.0f}ms", "End-to-end")
            + metric_card("Cache Hits", str(tier_counts.get("cache", 0)), f"{tier_counts.get('cache',0)/len(df)*100:.0f}% of traffic")
            + "</div>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("""
            <div style="font-size:0.8rem; font-weight:600; color:#64748b; margin-bottom:0.6rem;
                        text-transform:uppercase; letter-spacing:0.08em;">Tier Distribution</div>
            """, unsafe_allow_html=True)
            labels = list(tier_counts.index)
            values = list(tier_counts.values)
            colors = [TIER_COLORS.get(l, "#94a3b8") for l in labels]
            fig = go.Figure(go.Pie(
                labels=[f"{TIER_ICONS.get(l,'')} {l}" for l in labels],
                values=values, hole=0.55,
                marker=dict(colors=colors, line=dict(color="#050d1a", width=2)),
                textfont=dict(family="Space Grotesk", size=11),
            ))
            fig.update_layout(**plotly_fig_base(), height=240, showlegend=True,
                              legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

        with col2:
            st.markdown("""
            <div style="font-size:0.8rem; font-weight:600; color:#64748b; margin-bottom:0.6rem;
                        text-transform:uppercase; letter-spacing:0.08em;">Recent Queries</div>
            """, unsafe_allow_html=True)
            display_df = df[["ts", "query", "answered_by", "confidence", "latency_ms"]].copy()
            display_df.columns = ["Timestamp", "Query", "Tier", "Confidence", "Latency ms"]
            display_df["Query"] = display_df["Query"].str[:55] + "…"
            display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.0%}")
            display_df["Latency ms"] = display_df["Latency ms"].apply(lambda x: f"{x:.0f}")
            st.dataframe(display_df.head(20), width='stretch', hide_index=True)

        # Top misses
        generative_rows = df[df["answered_by"] == "generative"]
        if len(generative_rows) > 0:
            low_conf = generative_rows[generative_rows["confidence"] < 0.65]
            if len(low_conf) > 0:
                st.markdown("""<hr>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="glass-card" style="border-color:rgba(251,146,60,0.3);">
                    <div style="font-size:0.85rem; font-weight:700; color:#fb923c; margin-bottom:0.75rem;">
                        🟠 Top Cache Promotion Candidates ({len(low_conf)} queries)
                    </div>
                    <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:0.75rem;">
                        These queries fell through to generative with low confidence —
                        prime candidates for manual review and cache promotion via Phase 6.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                miss_df = low_conf[["query", "confidence", "latency_ms"]].copy()
                miss_df.columns = ["Query", "Confidence", "Latency ms"]
                st.dataframe(miss_df, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: CACHE INSPECTOR
# ═══════════════════════════════════════════════════════════════════════

elif "Cache Inspector" in page:
    glass_header(
        "🔍 Cache Inspector",
        "Browse, search, and inspect the CAG tier's instant-response knowledge base."
    )

    cache = load_cache()
    additions = load_cache_additions()

    avg_words = sum(len(v["answer"].split()) for v in cache.values()) // max(len(cache), 1)
    n_sources = len(set(v.get("source", "") for v in cache.values()))
    st.markdown(
        "<div class='metric-row'>"
        + metric_card("Cache Entries", str(len(cache)), "Hand-curated Q&A pairs")
        + metric_card("Phase 6 Promotions", str(len(additions)), "Auto-promoted from generative")
        + metric_card("Avg Answer Length", str(avg_words), "words per answer")
        + metric_card("Sources", str(n_sources), "Distinct references")
        + "</div>",
        unsafe_allow_html=True,
    )

    # Search
    search = st.text_input(
        "Search",
        placeholder="Filter by keyword — searches questions and answers...",
        label_visibility="collapsed",
    )

    filtered = {
        k: v for k, v in cache.items()
        if not search or search.lower() in k.lower() or search.lower() in v.get("answer", "").lower()
    }

    st.markdown(f"""
    <div style="font-size:0.8rem; color:#64748b; margin-bottom:0.75rem;">
        Showing <span class="highlight-number">{len(filtered)}</span> of
        <span class="highlight-number">{len(cache)}</span> entries
        {f'matching <em>"{search}"</em>' if search else ""}
    </div>
    """, unsafe_allow_html=True)

    # Table
    rows = []
    for q, v in filtered.items():
        rows.append({
            "Question": q,
            "Answer preview": v.get("answer", "")[:90] + "…" if len(v.get("answer","")) > 90 else v.get("answer",""),
            "Source": v.get("source", "—"),
            "Added": v.get("added", "—"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)

    col_dl, _ = st.columns([1, 4])
    with col_dl:
        st.download_button(
            "⬇ Export CSV",
            df.to_csv(index=False),
            "teletriage_cache.csv",
            "text/csv",
        )

    st.markdown("""<hr>""", unsafe_allow_html=True)

    # Entry detail
    st.markdown("""
    <div style="font-size:0.85rem; font-weight:600; color:#94a3b8; margin-bottom:0.75rem;
                text-transform:uppercase; letter-spacing:0.08em;">Entry Detail</div>
    """, unsafe_allow_html=True)

    if filtered:
        selected_q = st.selectbox(
            "Select entry",
            list(filtered.keys()),
            label_visibility="collapsed",
        )
        entry = filtered[selected_q]
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;
                         color:#64748b; margin-bottom:0.5rem;">Question</div>
            <div style="font-size:1rem; font-weight:600; color:#f1f5f9; margin-bottom:1.25rem;">
                {selected_q}
            </div>
            <div class="answer-box answer-box-cache" style="margin:0 0 1rem 0">
                {entry.get('answer','').replace(chr(10), '<br>')}
            </div>
            <div style="display:flex; gap:2rem; font-size:0.8rem; color:#64748b;">
                <span>📄 <strong style="color:#94a3b8">Source:</strong> {entry.get('source','—')}</span>
                <span>📅 <strong style="color:#94a3b8">Added:</strong> {entry.get('added','—')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Phase 6 additions
    if additions:
        st.markdown("""<hr>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.85rem; font-weight:600; color:#82C8E5; margin-bottom:0.75rem;
                    text-transform:uppercase; letter-spacing:0.08em;">
            ✅ Phase 6 Auto-Promoted Entries ({len(additions)})
        </div>
        """, unsafe_allow_html=True)
        add_rows = [{"Question": r.get("question",""), "Source": r.get("source",""),
                     "Confidence": f"{r.get('confidence',0):.2f}"} for r in additions]
        st.dataframe(pd.DataFrame(add_rows), width='stretch', hide_index=True)
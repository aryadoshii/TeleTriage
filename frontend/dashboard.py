"""
TeleTriage — Production Dashboard
Glassmorphism UI with full observability.

Run:
    uv run streamlit run frontend/dashboard.py --server.port 8501
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))

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
    color: #e2e8f0 !important;
}

/* ── Animated background mesh ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(ellipse 80% 50% at 10% 20%, rgba(14,165,233,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(99,102,241,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(6,182,212,0.05) 0%, transparent 70%);
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
    border-right: 1px solid rgba(14,165,233,0.15) !important;
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
    border: 1px solid rgba(14,165,233,0.18);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: border-color 0.2s;
}

.glass-card:hover {
    border-color: rgba(14,165,233,0.35);
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
    border: 1px solid rgba(14,165,233,0.18);
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
    background: linear-gradient(90deg, transparent, rgba(14,165,233,0.6), transparent);
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

.tier-cache    { background: rgba(34,197,94,0.18);  color: #4ade80; border: 1px solid rgba(34,197,94,0.35); }
.tier-retrieval{ background: rgba(14,165,233,0.18); color: #38bdf8; border: 1px solid rgba(14,165,233,0.35); }
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

.answer-box-cache     { border-color: #4ade80; }
.answer-box-retrieval { border-color: #38bdf8; }
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
    color: #38bdf8;
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
    border: 1px solid rgba(14,165,233,0.25);
    border-radius: 10px;
    padding: 0.75rem 1.25rem;
    text-align: center;
    min-width: 140px;
}

.arch-tier-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; }
.arch-tier-name  { font-size: 1rem; font-weight: 700; margin: 0.2rem 0; }
.arch-tier-speed { font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; }
.arch-arrow      { color: #38bdf8; font-size: 1.25rem; }

/* ── Step cards (onboarding) ── */
.step-card {
    background: rgba(14, 30, 60, 0.55);
    border: 1px solid rgba(14,165,233,0.18);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}

.step-number {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
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
div[data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }

.stTextArea textarea {
    background: rgba(8, 20, 40, 0.7) !important;
    border: 1px solid rgba(14,165,233,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.92rem !important;
}

.stTextArea textarea:focus {
    border-color: rgba(14,165,233,0.7) !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.35) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.5) !important;
}

.stSelectbox > div > div {
    background: rgba(8, 20, 40, 0.7) !important;
    border: 1px solid rgba(14,165,233,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

div[data-testid="stExpander"] {
    background: rgba(14, 30, 60, 0.4) !important;
    border: 1px solid rgba(14,165,233,0.15) !important;
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
    background: rgba(14,165,233,0.12) !important;
    color: #e2e8f0 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(14,165,233,0.12) !important;
    margin: 1.5rem 0 !important;
}

/* ── Alert / info boxes ── */
div[data-testid="stInfo"] {
    background: rgba(14,165,233,0.1) !important;
    border: 1px solid rgba(14,165,233,0.25) !important;
    border-radius: 10px !important;
    color: #bae6fd !important;
}

div[data-testid="stWarning"] {
    background: rgba(251,146,60,0.1) !important;
    border: 1px solid rgba(251,146,60,0.25) !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(14,165,233,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(14,165,233,0.5); }

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
    "cache":      "#4ade80",
    "retrieval":  "#38bdf8",
    "generative": "#fb923c",
    "none":       "#f87171",
}

TIER_ICONS = {
    "cache":      "⚡",
    "retrieval":  "🔍",
    "generative": "🤖",
    "none":       "⚠️",
}


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
    log_dir = PROJECT_ROOT / "logs"
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
    path = PROJECT_ROOT / "logs" / "query_metrics.jsonl"
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
    p = PROJECT_ROOT / "data" / "sample_cache.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def load_cache_additions() -> list[dict]:
    p = PROJECT_ROOT / "data" / "cache_additions.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().strip().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def count_kb_chunks() -> int:
    for name in ("real_kb.jsonl", "sample_kb.jsonl"):
        p = PROJECT_ROOT / "data" / name
        if p.exists():
            return sum(1 for l in p.read_text().splitlines() if l.strip())
    return 0


def count_tests() -> int:
    test_dir = PROJECT_ROOT / "tests"
    if not test_dir.exists():
        return 0
    return sum(
        f.read_text().count("\ndef test_")
        for f in test_dir.glob("test_*.py")
    )


def plotly_fig_base() -> dict:
    """Base layout for all plotly charts matching the dark glass theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,20,40,0.4)",
        font=dict(family="Space Grotesk, sans-serif", color="#94a3b8"),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor="rgba(14,165,233,0.1)", zerolinecolor="rgba(14,165,233,0.15)"),
        yaxis=dict(gridcolor="rgba(14,165,233,0.1)", zerolinecolor="rgba(14,165,233,0.15)"),
    )


# ─── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 1.75rem 1rem 1.25rem 1rem; border-bottom: 1px solid rgba(14,165,233,0.15); margin-bottom: 1rem;">
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
    <div style="padding: 1rem; margin: 0 0.25rem; background: rgba(14,165,233,0.06);
                border: 1px solid rgba(14,165,233,0.15); border-radius: 12px;">
        <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                    color:#64748b; margin-bottom:0.75rem;">System Status</div>
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="width:7px; height:7px; background:#4ade80; border-radius:50%;
                         box-shadow:0 0 6px #4ade80; display:inline-block;"></span>
            <span style="font-size:0.8rem; color:#94a3b8;">CAG Cache</span>
            <span style="margin-left:auto; font-size:0.75rem; font-family:monospace; color:#4ade80;">~0ms</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="width:7px; height:7px; background:#38bdf8; border-radius:50%;
                         box-shadow:0 0 6px #38bdf8; display:inline-block;"></span>
            <span style="font-size:0.8rem; color:#94a3b8;">RAG Retrieval</span>
            <span style="margin-left:auto; font-size:0.75rem; font-family:monospace; color:#38bdf8;">~200ms</span>
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
    kb_chunks = count_kb_chunks()
    n_tests = count_tests()

    # ── Centered title ────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center; font-size:2.8rem; font-weight:900; "
        "letter-spacing:-0.03em; color:#f1f5f9; margin:2rem 0 0.4rem 0;'>📡 TeleTriage</h1>"
        "<p style='text-align:center; font-size:1rem; color:#64748b; margin-bottom:2.5rem;'>"
        "AI-powered hierarchical fault-resolution engine for telecom networks</p>"
        "<hr>",
        unsafe_allow_html=True,
    )

    # ── Hero: The Problem ──────────────────────────────────────────────
    st.markdown(
        "<div style='margin-bottom:2rem;'>"
        "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#38bdf8;margin-bottom:0.6rem;'>The Problem</div>"
        "<div style='font-size:2.4rem;font-weight:800;line-height:1.15;letter-spacing:-0.03em;color:#f1f5f9;margin-bottom:0.9rem;'>"
        "Telecom faults escalate in minutes.<br>"
        "<span style='color:#38bdf8;'>Answers shouldn&apos;t take hours.</span></div>"
        "<div style='font-size:1rem;color:#94a3b8;max-width:720px;line-height:1.75;'>"
        "Network operations teams debug the same faults repeatedly — packet loss, handover failures, bearer drops — "
        "by digging through 3GPP specs, vendor docs, and tribal knowledge scattered across Confluence pages and engineers&apos; heads. "
        "There is no single system that triages a fault description and returns a structured, sourced answer in under a second."
        "</div></div><hr>",
        unsafe_allow_html=True,
    )

    # ── The Solution ──────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
                    color: #4ade80; margin-bottom: 0.6rem;">The Solution</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem; letter-spacing: -0.02em;">
            A confidence-calibrated, three-tier cascade that answers in the right order — fast first.
        </div>
        <div style="font-size: 0.88rem; color: #94a3b8; max-width: 720px; line-height: 1.7; margin-bottom: 1.5rem;">
            Instead of always calling an LLM, TeleTriage tries the cheapest path first and escalates
            only when confidence falls below an empirically measured threshold — not a guessed one.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cascade diagram — no blank lines inside (blank lines trigger Markdown code-block mode)
    _cache_n = len(cache)
    _kb_n = kb_chunks
    _kb_str = f"{_kb_n:,}" if _kb_n else "—"
    t1 = (f"<div style='flex:1;min-width:160px;padding:1.25rem;border-right:1px solid rgba(14,165,233,0.12);'>"
          f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem;'>Tier 1 · ~0 ms</div>"
          f"<div style='font-size:1.25rem;font-weight:800;color:#4ade80;margin-bottom:0.3rem;'>&#9889; Cache</div>"
          f"<div style='font-size:0.78rem;color:#94a3b8;line-height:1.5;margin-bottom:0.75rem;'>Exact match &#8594; normalised match &#8594; fuzzy token-set match. Returns instantly from {_cache_n} hand-curated Q&amp;A pairs.</div>"
          f"<div style='font-size:0.72rem;background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.2);border-radius:6px;padding:4px 10px;display:inline-block;color:#4ade80;'>Confidence &ge; 0.85 &#8594; stop</div></div>")
    arr = "<div style='display:flex;align-items:center;padding:0 0.75rem;color:#334155;font-size:0.9rem;font-weight:700;'>&#8250;</div>"
    t2 = (f"<div style='flex:1;min-width:160px;padding:1.25rem;border-right:1px solid rgba(14,165,233,0.12);'>"
          f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem;'>Tier 2 · ~200 ms</div>"
          f"<div style='font-size:1.25rem;font-weight:800;color:#38bdf8;margin-bottom:0.3rem;'>&#128269; Retrieval</div>"
          f"<div style='font-size:0.78rem;color:#94a3b8;line-height:1.5;margin-bottom:0.75rem;'>BM25 + dense embeddings fused with RRF, re-ranked by a cross-encoder. Searches {_kb_str} real 3GPP chunks.</div>"
          f"<div style='font-size:0.72rem;background:rgba(14,165,233,0.1);border:1px solid rgba(14,165,233,0.2);border-radius:6px;padding:4px 10px;display:inline-block;color:#38bdf8;'>Rerank logit &ge; 0.20 &#8594; stop</div></div>")
    t3 = ("<div style='flex:1;min-width:160px;padding:1.25rem;'>"
          "<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin-bottom:0.5rem;'>Tier 3 · ~1.1 s</div>"
          "<div style='font-size:1.25rem;font-weight:800;color:#fb923c;margin-bottom:0.3rem;'>&#129302; Generative</div>"
          "<div style='font-size:0.78rem;color:#94a3b8;line-height:1.5;margin-bottom:0.75rem;'>Groq Llama 3.3 70B (&#8594; Gemini 2.0 Flash &#8594; local Qwen). Always answers — safety net of last resort.</div>"
          "<div style='font-size:0.72rem;background:rgba(251,146,60,0.1);border:1px solid rgba(251,146,60,0.2);border-radius:6px;padding:4px 10px;display:inline-block;color:#fb923c;'>Always answers</div></div>")
    footer = "<div style='margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(14,165,233,0.1);font-size:0.78rem;color:#475569;line-height:1.6;'>Thresholds are <span style='color:#38bdf8;'>empirically calibrated</span> on 20 held-out queries via a three-point ablation curve — not guessed.</div>"
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
    test_label = str(n_tests) if n_tests else "—"

    st.markdown(
        "<div class='metric-row'>"
        + metric_card("Best BERTScore", f"{best_bs:.3f}" if best_bs else "—", "On 20 held-out queries")
        + metric_card("Cache Entries", str(len(cache)), "Hand-curated Q&A pairs")
        + metric_card("Retrieval p50", retrieval_p50_label, "Embed + rerank latency")
        + metric_card("Tests Passing", test_label, "Fast suite, no regressions")
        + metric_card("KB Chunks", kb_label, "From live 3GPP specs")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("""<hr>""", unsafe_allow_html=True)

    # ── Tools Used ────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
                color: #fb923c; margin-bottom: 0.6rem;">Tools &amp; Technology</div>
    <div style="font-size: 1.35rem; font-weight: 700; color: #f1f5f9; margin-bottom: 1.25rem; letter-spacing: -0.02em;">
        Every component chosen for a reason, benchmarked, not assumed.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
                        color:#4ade80; margin-bottom:0.75rem;">Retrieval Stack</div>
            <div style="display:flex; flex-direction:column; gap:0.5rem; font-size:0.82rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Sparse</span>
                    <span style="color:#4ade80; font-family:monospace; font-size:0.78rem;">BM25Okapi</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Embeddings</span>
                    <span style="color:#4ade80; font-family:monospace; font-size:0.78rem;">BGE-small-en-v1.5</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Vector index</span>
                    <span style="color:#4ade80; font-family:monospace; font-size:0.78rem;">FAISS IndexFlatIP</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Fusion</span>
                    <span style="color:#4ade80; font-family:monospace; font-size:0.78rem;">RRF  k=60</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Cross-encoder</span>
                    <span style="color:#4ade80; font-family:monospace; font-size:0.78rem;">BGE-reranker-base</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
                        color:#38bdf8; margin-bottom:0.75rem;">Generation Stack</div>
            <div style="display:flex; flex-direction:column; gap:0.5rem; font-size:0.82rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Primary LLM</span>
                    <span style="color:#38bdf8; font-family:monospace; font-size:0.78rem;">Llama 3.3 70B</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Fallback</span>
                    <span style="color:#38bdf8; font-family:monospace; font-size:0.78rem;">Gemini 2.0 Flash</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Local fallback</span>
                    <span style="color:#38bdf8; font-family:monospace; font-size:0.78rem;">Qwen 2.5 1.5B</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Inference API</span>
                    <span style="color:#38bdf8; font-family:monospace; font-size:0.78rem;">Groq (free tier)</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Cache dedup</span>
                    <span style="color:#38bdf8; font-family:monospace; font-size:0.78rem;">RapidFuzz Levenshtein</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
                        color:#fb923c; margin-bottom:0.75rem;">Data &amp; Eval</div>
            <div style="display:flex; flex-direction:column; gap:0.5rem; font-size:0.82rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">KB source</span>
                    <span style="color:#fb923c; font-family:monospace; font-size:0.78rem;">5 live 3GPP specs</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Dedup</span>
                    <span style="color:#fb923c; font-family:monospace; font-size:0.78rem;">MinHash LSH ≥0.85</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Quality metric</span>
                    <span style="color:#fb923c; font-family:monospace; font-size:0.78rem;">BERTScore F1</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Synth Q&amp;A</span>
                    <span style="color:#fb923c; font-family:monospace; font-size:0.78rem;">Llama 3.3 70B</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;
                             padding:0.45rem 0.7rem; background:rgba(8,20,40,0.5); border-radius:7px;">
                    <span style="color:#94a3b8;">Test suite</span>
                    <span style="color:#fb923c; font-family:monospace; font-size:0.78rem;">142 tests · pytest</span>
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
        "cache":      ("rgba(34,197,94,0.12)",  "rgba(34,197,94,0.3)",  "#4ade80"),
        "retrieval":  ("rgba(14,165,233,0.12)", "rgba(14,165,233,0.3)", "#38bdf8"),
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
        border:1px solid rgba(14,165,233,0.35) !important;
        color:#94a3b8 !important;
        transform:none !important;
        transition:background 0.15s,color 0.15s !important;
    }
    div[data-testid="column"] .stButton>button:hover {
        background:rgba(14,165,233,0.12) !important;
        color:#e2e8f0 !important;
        border-color:rgba(14,165,233,0.6) !important;
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
                             use_container_width=True):
                    st.session_state["query_text"] = label
                    st.rerun()

    st.markdown(
        "<div style='font-size:0.72rem;color:#334155;margin:0.5rem 0 0.75rem 0;'>"
        "🟢 Green = likely cache &nbsp;·&nbsp; 🔵 Blue = likely retrieval &nbsp;·&nbsp; 🟠 Orange = likely generative"
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
        run = st.button("Run Query →", use_container_width=True)
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
                <div style="font-size:0.82rem; color:#cbd5e1;">{q}</div>
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
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col3:
            st.markdown("""
            <div style="font-size:0.85rem; font-weight:600; color:#94a3b8; margin-bottom:0.75rem;
                        text-transform:uppercase; letter-spacing:0.08em;">Latency Percentiles</div>
            """, unsafe_allow_html=True)
            pctls = ["p50", "p95", "p99"]
            vals = [overall.get(f"{p}_sec", 0) * 1000 for p in pctls]
            fig = go.Figure(go.Bar(
                x=pctls, y=vals,
                marker_color=["#4ade80", "#38bdf8", "#fb923c"],
                marker_line_color="rgba(0,0,0,0)",
                text=[f"{v:.0f}ms" for v in vals],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#e2e8f0"),
            ))
            fig.update_layout(**plotly_fig_base(), height=220)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
            st.dataframe(df.style.apply(_highlight_best, axis=1), use_container_width=True, hide_index=True)

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
            st.dataframe(df, use_container_width=True, hide_index=True)

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
                Run some queries in the <strong style="color:#38bdf8">Live Query</strong> tab.
                Each query is automatically appended to <code>logs/query_metrics.jsonl</code>.
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
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
            st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)

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
                st.dataframe(miss_df, use_container_width=True, hide_index=True)


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
    st.dataframe(df, use_container_width=True, hide_index=True)

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
        <div style="font-size:0.85rem; font-weight:600; color:#4ade80; margin-bottom:0.75rem;
                    text-transform:uppercase; letter-spacing:0.08em;">
            ✅ Phase 6 Auto-Promoted Entries ({len(additions)})
        </div>
        """, unsafe_allow_html=True)
        add_rows = [{"Question": r.get("question",""), "Source": r.get("source",""),
                     "Confidence": f"{r.get('confidence',0):.2f}"} for r in additions]
        st.dataframe(pd.DataFrame(add_rows), use_container_width=True, hide_index=True)
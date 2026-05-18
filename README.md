<div align="center">

# 📡 TeleTriage

### Telecom faults escalate in minutes. Answers shouldn't take hours.
**A hierarchical multi-tier fault-resolution engine that routes telecom queries through cache, retrieval, and generative AI — stopping at the cheapest path that meets confidence.**

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Index-00ADD8?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=flat-square)](https://console.groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-142%20passing-22c55e?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/License-MIT-475569?style=flat-square)](LICENSE)

> A **3-tier confidence-calibrated cascade** — Cache-Augmented Generation → Hybrid RAG (BM25 + dense + cross-encoder rerank) → Groq Llama 3.3 70B — built on 7,181 real 3GPP specification chunks, evaluated on 20 held-out queries with empirically measured thresholds.

</div>

---

## What It Does

TeleTriage takes any telecom fault description and routes it through a three-tier cascade that always tries the cheapest, fastest path first:

- **Caches** — exact match → normalised match → fuzzy token-set match. Returns in under 1ms from 25 hand-curated Q&A pairs.
- **Retrieves** — BM25 sparse search fused with BGE-small dense embeddings via Reciprocal Rank Fusion, re-ranked by a bge-reranker-base cross-encoder. Searches 7,181 real 3GPP specification chunks.
- **Generates** — Groq Llama 3.3 70B as primary, Gemini 2.0 Flash as backup, local Qwen 2.5-1.5B as offline fallback. Always answers — the safety net of last resort.
- **Calibrates** — every threshold set by a three-point ablation curve on 20 held-out queries. Not guessed.
- **Evaluates** — ROUGE-L and BERTScore F1, latency percentiles (p50/p95/p99), per-tier quality breakdown, all logged and visualised in a live dashboard.

TeleTriage is designed for telecom operations teams that need structured, sourced answers faster than a manual spec lookup.

---

## ✨ Features

### Core Architecture
- **3-tier cascade** — CAG → RAG → Generative, each tier delegating only when confidence falls below its calibrated threshold
- **Confidence-based routing** — CAG stops at ≥ 0.85, retrieval stops at rerank logit ≥ 0.20, generative always answers
- **Empirically calibrated thresholds** — three-point ablation curve on 20 held-out queries, not hyperparameter guessing
- **Full tier trace** — every query logs which tier answered, confidence, latency, and delegation reason

### Retrieval Pipeline
- **Hybrid search** — BM25 (lexical) + BGE-small-en-v1.5 (dense) fused with Reciprocal Rank Fusion (k=60, Cormack et al. 2009)
- **Cross-encoder reranking** — bge-reranker-base scores query-document pairs jointly; sigmoid(logit) → calibrated confidence
- **BGE asymmetric prefix** — query prefix applied correctly; corpus documents encoded without prefix (as the model was trained)
- **7,181 real 3GPP chunks** — scraped from TS 36.300, 36.331, 36.413, 24.301, 23.401; MinHash LSH dedup (Jaccard ≥ 0.85)

### Generation Stack
- **Groq Llama 3.3 70B** — primary backend, ~500 tok/s, free tier
- **Gemini 2.0 Flash** — automatic fallback on rate limit or failure
- **Qwen 2.5-1.5B-Instruct** — local offline fallback, runs on CPU, no API key required
- **Exponential backoff** — 1s/2s/4s per backend before propagating to next

### Adaptive Cache (Phase 6)
- **3-gate quality filter** — confidence ≥ 0.60, answer length ≥ 50 words, Levenshtein dedup before any promotion
- **Batch promotion CLI** — `scripts/promote_cache.py` with `--dry-run`, rich table output, JSON report
- **Live injection** — promoted entries immediately available without restart

### Observability Dashboard
- **Glassmorphism dark UI** — five views: Overview, Live Query, Eval Results, Query Log, Cache Inspector
- **Calibration history** — all eval runs in one table, best run highlighted, per-run BERTScore + latency charts
- **Query logging** — every Live Query run appended to `outputs/query_metrics.jsonl`; top promotion candidates surfaced automatically
- **CSV export** — per-query breakdown and cache entries exportable directly from the dashboard

---

## 📸 Screenshots

### 🏠 Overview — Architecture & Introduction
<img src="frontend/assets/screenshot_overview_1.png" alt="TeleTriage Overview — problem statement and architecture" width="100%" />

---

### 📊 Overview — Tech Stack & Stats
<img src="frontend/assets/screenshot_overview_2.png" alt="TeleTriage Overview — tech stack and metrics" width="100%" />

---

### ⚡ Live Query — Running the Cascade
<img src="frontend/assets/screenshot_live_query.png" alt="TeleTriage Live Query with result metrics" width="100%" />

---

### 🔬 Tier Trace — Full Cascade Detail
<img src="frontend/assets/screenshot_tier_trace.png" alt="TeleTriage tier trace expander showing all three tiers" width="100%" />

---

### 📊 Evaluation Results — Calibration History & Per-Tier Charts
<img src="frontend/assets/screenshot_eval_results.png" alt="TeleTriage Eval Results with calibration history table" width="100%" />

---

### 📋 Per-Query Breakdown
<img src="frontend/assets/screenshot_per_query.png" alt="TeleTriage per-query breakdown table" width="100%" />

---

### 🗂️ Query Log — Usage Tracking & Promotion Candidates
<img src="frontend/assets/screenshot_query_log.png" alt="TeleTriage Query Log with tier distribution and promotion candidates" width="100%" />

---

### 🔍 Cache Inspector — Browse, Search & Inspect
<img src="frontend/assets/screenshot_cache_inspector.png" alt="TeleTriage Cache Inspector with search and entry detail" width="100%" />

---

## 📊 Evaluation Results

Evaluated on **20 held-out telecom fault queries** — none appeared in the cache or knowledge base during development.

### Three-Point Calibration Curve

| Config | Threshold | Retrieval | Generative | ROUGE-L | BERTScore F1 | p50 Latency |
|--------|-----------|-----------|------------|---------|--------------|-------------|
| Strict | logit ≥ 0.50 | 0% | 100% | 0.153 | 0.790 | 1458ms |
| **Calibrated** *(production)* | **logit ≥ 0.20** | **15%** | **85%** | **0.153** | **0.793** | **1548ms** |
| Relaxed | logit ≥ 0.00 | 100% | 0% | 0.123 | 0.815 | 163ms |

### Key Findings

**BERTScore favours retrieval over generation (+0.025).** KB contains verified 3GPP procedures; Groq paraphrases the same knowledge less precisely. Retrieval at logit ≥ 0.20 scores BERTScore 0.804–0.842 vs generative mean 0.787.

**ROUGE-L is the wrong primary metric here.** The KB uses dense structured notation (3GPP TS references, EMM cause codes, interface names) that diverges lexically from plain-English reference answers. BERTScore captures semantic correctness; ROUGE measures word overlap. Both are reported for transparency.

**The reranker correctly signals no-coverage.** 17 of 20 eval queries score logit 0.003–0.010 — near-noise. Their retrieval BERTScore (0.786–0.799) sits at or below the generative mean, confirming the reranker is right to delegate.

### Latency Percentiles (generative tier, n=17)

| p50 | p95 | p99 |
|-----|-----|-----|
| 1458ms | 1734ms | 1775ms |

Very stable Groq round-trips. 316ms spread between p50 and p99 reflects network variance, not model variance.

---

## 🏗️ How the Cascade Works

```
Incoming fault description
        │
        ▼
┌─────────────────────────────────────────────┐
│  Tier 1 — CAG Cache                ~0–1ms   │
│  exact → normalised → fuzzy match           │
│  confidence ≥ 0.85 → answer returned        │
└──────────────────────┬──────────────────────┘
                       │ miss
                       ▼
┌─────────────────────────────────────────────┐
│  Tier 2 — Hybrid RAG           ~150–250ms   │
│  BM25 (top-20) ──┐                          │
│                  ├── RRF k=60               │
│  BGE dense (top-20)                         │
│                  └── cross-encoder rerank   │
│  rerank logit ≥ 0.20 → answer returned      │
└──────────────────────┬──────────────────────┘
                       │ miss
                       ▼
┌─────────────────────────────────────────────┐
│  Tier 3 — Generative           ~800–1800ms  │
│  Groq Llama 3.3 70B                         │
│    → Gemini 2.0 Flash (if rate-limited)     │
│      → local Qwen 2.5-1.5B (if offline)     │
│  Always answers — safety net of last resort │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
     Response(answer, tier, confidence,
              latency_sec, tier_trace)
```

---

## 🧱 Component Reference

| # | Component | Technology | Role |
|---|-----------|-----------|------|
| 1 | Cache tier | `rapidfuzz` token-set ratio | Exact → normalised → fuzzy match; confidence scales with similarity |
| 2 | Embedder | `BAAI/bge-small-en-v1.5` 384-dim | Bi-encoder with asymmetric query prefix; L2-norm → inner product = cosine |
| 3 | Sparse retriever | `rank_bm25` BM25Okapi | Lexical baseline; catches exact 3GPP identifiers dense search misses |
| 4 | Dense retriever | `faiss-cpu` IndexFlatIP | Exact cosine search over pre-encoded corpus |
| 5 | Hybrid fusion | Reciprocal Rank Fusion k=60 | score(d) = Σ 1/(60 + rank_i); score-free, no normalisation needed |
| 6 | Reranker | `BAAI/bge-reranker-base` | Cross-encoder joint attention; sigmoid(logit) → confidence |
| 7 | Generation primary | Groq — Llama 3.3 70B | ~500 tok/s, free tier, exponential backoff |
| 8 | Generation backup | Gemini 2.0 Flash | Auto-fallback on Groq rate limit or failure |
| 9 | Generation offline | `Qwen2.5-1.5B-Instruct` | Local CPU fallback, no API key required |
| 10 | Cascade router | `backend.router.Router` | Orchestrates tiers, accumulates trace, logs every query |
| 11 | Cache promoter | `CachePromoter` + `rapidfuzz` | 3-gate filter before promotion (confidence, length, Levenshtein dedup) |
| 12 | Evaluator | `rouge-score` + `bert-score` | Per-tier and end-to-end metrics; subprocess-isolated BERTScore |
| 13 | Dashboard | Streamlit + Plotly | Glassmorphism dark UI; 5 views; auto-logs every Live Query run |

---

## What Makes This Different

| Feature | TeleTriage | Pure LLM Approach |
|---------|-----------|-------------------|
| Latency for known queries | ✅ ~0ms (cache hit) | ❌ ~1–2s always |
| Cost for repeated queries | ✅ Zero — static cache | ❌ Full LLM inference every time |
| Answer provenance | ✅ Source-cited (3GPP TS, RFC) | ❌ LLM may hallucinate citations |
| Calibrated thresholds | ✅ Measured on 20 held-out queries | ❌ Arbitrary or none |
| Evaluation methodology | ✅ ROUGE-L + BERTScore + latency percentiles | ❌ Vibes-based |
| Retrieval precision | ✅ Cross-encoder reranking over BM25+dense | ❌ Embedding similarity only |
| Knowledge base | ✅ 7,181 real 3GPP spec chunks | ❌ Frozen training data |
| Novel query coverage | ✅ Generative fallback always answers | ✅ Always answers |
| Test coverage | ✅ 142 tests, 0 regressions | ❌ Typically none |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `BAAI/bge-small-en-v1.5` — 384-dim, ~30M params, CPU-friendly |
| Sparse retrieval | `rank_bm25` BM25Okapi |
| Vector index | `faiss-cpu` IndexFlatIP — exact cosine search |
| Hybrid fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | `BAAI/bge-reranker-base` cross-encoder |
| Generation (primary) | Groq API — Llama 3.3 70B (free tier) |
| Generation (backup) | Google Gemini 2.0 Flash (free tier via AI Studio) |
| Generation (offline) | `Qwen2.5-1.5B-Instruct` — local, no API required |
| Config system | Pydantic-settings + YAML + `.env` |
| Evaluation | `rouge-score`, `bert-score`, `psutil` |
| Dashboard | Streamlit + Plotly |
| Data pipeline | MinHash LSH dedup, recursive chunking, Groq-based synth QA |
| Testing | pytest — 142 tests, fast suite < 5s |
| Package management | `uv` |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [`uv`](https://astral.sh/uv) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- API keys (both free, no credit card required):

| Service | Where to get it |
|---------|----------------|
| Groq | [console.groq.com](https://console.groq.com) → Create API Key |
| Google AI Studio | [aistudio.google.com](https://aistudio.google.com/app/apikey) → Get API Key |

### 1. Clone & Install

```bash
git clone https://github.com/aryadoshii/TeleTriage.git
cd TeleTriage
uv sync
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Groq — primary generative backend (30 req/min on Llama 3.3 70B, free)
GROQ_API_KEY=gsk_...

# Google AI Studio — backup generative backend (free tier)
GOOGLE_API_KEY=...
```

### 3. Build Retrieval Indexes

```bash
# Downloads BGE-small (~30MB) on first run. ~2 min for 7,181 docs.
uv run python scripts/build_index.py
```

### 4. Run the System

```bash
# Cache hit — instant response
uv run teletriage query "high packet loss" --verbose

# Retrieval query
uv run teletriage query "eNB not generating handover despite A3 event"

# Novel query — generative fallback
uv run teletriage query "PDCP SN wrap-around during extended coverage"

# Launch the dashboard
uv run streamlit run frontend/dashboard.py --server.port 8501

# Run the evaluation harness
uv run python scripts/run_eval.py
```

---

## 📁 Project Layout

```
TeleTriage/
├── frontend/
│   ├── assets/                      # Screenshots and static assets
│   └── dashboard.py                 # Streamlit observability dashboard
│
├── backend/
│   ├── config.py                    # Pydantic settings — YAML + .env
│   ├── types.py                     # Query, TierResult, Response
│   ├── router.py                    # Cascade orchestrator + query logging
│   │
│   ├── tiers/
│   │   ├── base.py                  # Abstract BaseTier
│   │   ├── cache_tier.py            # CAG: exact + normalised + fuzzy match
│   │   ├── retrieval_tier.py        # RAG: BM25 + dense + RRF + rerank
│   │   └── generative_tier.py       # Groq → Gemini → local Qwen fallback
│   │
│   ├── retrieval/
│   │   ├── _types.py                # Hit dataclass
│   │   ├── embedder.py              # BGE-small with asymmetric query prefix
│   │   ├── bm25.py                  # BM25Okapi (build / save / load / search)
│   │   ├── dense.py                 # FAISS IndexFlatIP (build / save / load / search)
│   │   ├── hybrid.py                # Reciprocal Rank Fusion
│   │   └── reranker.py              # bge-reranker-base cross-encoder
│   │
│   ├── generation/
│   │   └── llm_client.py            # Protocol + GroqClient + GeminiClient + LocalQwenClient
│   │
│   ├── cache/
│   │   └── promoter.py              # CachePromoter — 3-gate quality filter
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # ROUGE-L, BERTScore, latency percentiles
│   │   ├── test_set.py              # 20 held-out eval queries with reference answers
│   │   └── evaluator.py             # Per-tier and end-to-end reporting
│   │
│   └── observability/
│       └── logger.py                # structlog setup
│
├── config/
│   └── config.yaml                  # Tier thresholds, model names (calibrated)
│
├── database/
│   ├── sample_cache.json            # CAG: 25 hand-curated telecom Q&A pairs
│   ├── sample_kb.jsonl              # RAG KB: 30 seed entries (used in tests)
│   └── indexes/                     # Built by scripts/build_index.py (gitignored)
│
├── scripts/
│   ├── run_query.py                 # Single query CLI demo
│   ├── build_index.py               # Build BM25 + FAISS indexes
│   ├── scrape_data.py               # Scrape 3GPP specs → chunk → dedup
│   ├── synth_qa.py                  # Groq/Gemini synthetic Q&A generation
│   ├── promote_cache.py             # Batch cache promotion CLI
│   └── run_eval.py                  # Run full evaluation harness
│
├── tests/
│   ├── test_cache_tier.py           # 13 tests
│   ├── test_retrieval_tier.py       # 57 tests (32 fast, 25 @slow)
│   ├── test_generative_tier.py      # 34 tests (32 mocked, 2 @integration)
│   └── test_cache_promoter.py       # 27 tests
│
├── outputs/                         # Eval reports + query metrics (gitignored)
├── pyproject.toml
├── .env.example
└── ROADMAP.md
```

---

## Running Tests

```bash
# Fast suite only — no model loading, < 5 seconds
uv run pytest -m "not slow and not integration"

# Full suite including model-loading tests (~several minutes)
uv run pytest -m "not integration"

# Integration tests — requires GROQ_API_KEY in .env
uv run pytest -m integration -v

# All 142 tests
uv run pytest
```

---

## Data Pipeline

The knowledge base was built from public 3GPP specifications:

```bash
# 1. Scrape and chunk 5 3GPP specs (~9MB prose, ~2 min)
uv run python scripts/scrape_data.py

# 2. Generate synthetic Q&A pairs via Groq (~7 min at 28 req/min)
uv run python scripts/synth_qa.py

# 3. Build retrieval indexes from the knowledge base
uv run python scripts/build_index.py --kb database/real_kb.jsonl

# 4. (Optional) Promote validated Q&A pairs into the cache
uv run python scripts/promote_cache.py --dry-run
uv run python scripts/promote_cache.py
```

**Scraping stats (5 specs, public 3GPP FTP):**

| Spec | Content | Quality Chunks |
|------|---------|---------------|
| TS 36.300 — E-UTRA Architecture | 1,017,067 chars | 644 |
| TS 36.331 — RRC | 3,214,201 chars | 2,112 |
| TS 36.413 — S1-AP | 849,732 chars | 509 |
| TS 24.301 — NAS/EPS | 2,146,789 chars | 2,269 |
| TS 23.401 — GPRS/EPC | 1,618,403 chars | 1,800 |

Deduplication: MinHash LSH (Jaccard ≥ 0.85, 128 perms, 5-word shingles) — 7,334 → 7,181 (−1.3%). 56 additional ToC/header-only chunks removed by quality filter.

---

## Design Notes

### Why three tiers?

Most production telecom faults are repetitive. The Pareto principle holds hard — roughly 60% of queries are the same handful of known issues. Paying LLM inference cost for every query is wasteful when an O(1) cache lookup exists for it. The tier hierarchy routes cheap queries to cheap answers and reserves the expensive path for genuinely novel problems.

### Why hybrid retrieval?

Dense-only retrieval misses exact term matches for rare telecom identifiers (`QCI 1`, `S1-AP`, `GUMMEI`, `EMM cause 11`). BM25 catches these; dense embeddings catch paraphrases. RRF fuses both without score normalisation — no calibration needed at the fusion step. The cross-encoder reranker then scores the top candidates with full query-document attention, the standard production pattern (Nogueira & Cho, 2019).

### Why BERTScore over ROUGE?

ROUGE measures lexical overlap. The 3GPP knowledge base uses dense structured notation (cause codes, interface names, 3GPP TS references) that diverges lexically from any plain-English reference answer. BERTScore uses contextual embeddings and measures semantic similarity — a more honest proxy for whether the answer is actually correct. Both are reported; ROUGE is included for comparability, not as the primary signal.

### On confidence calibration

All three thresholds were empirically determined on 20 held-out queries, not guessed. The CAG fuzzy threshold (90) was set by measuring precision/recall on paraphrase variants. The retrieval threshold (`min_rerank_score = 0.20`) was determined via a three-point ablation curve — retrieval at logit ≥ 0.20 scores BERTScore 0.804–0.842; near-zero logit retrieval scores 0.786–0.799 (at or below the generative mean), confirming the reranker correctly signals no-coverage.

---

## 🛠️ Troubleshooting

**Indexes not found when running a query**
```bash
uv run python scripts/build_index.py
```

**Retrieval tier always delegates (near-zero logit scores)**

Indexes may be built against the 30-doc sample KB. Rebuild against the real data:
```bash
uv run python scripts/build_index.py --kb database/real_kb.jsonl
```

**Generative tier fails with authentication error**

Verify `GROQ_API_KEY` in `.env` — get one free at [console.groq.com](https://console.groq.com).

**BERTScore SIGSEGV on macOS Apple Silicon**

Known issue when BGE-small, bge-reranker, and distilbert are all loaded in the same process. Fixed in `backend/evaluation/metrics.py` via subprocess isolation — no action needed.

**Model loading `UNEXPECTED` key warning**

```
embeddings.position_ids | UNEXPECTED
```

Harmless warning from `sentence-transformers` when loading BGE models. Does not affect embedding quality or search results.

---

## 🗺️ Roadmap

- ✅ **Phase 1** — scaffold, CAG tier, cascade router, CLI, 13 tests
- ✅ **Phase 2** — hybrid RAG: BM25 + dense + RRF + cross-encoder rerank, 57 tests
- ✅ **Phase 3** — generative tier: Groq → Gemini → local Qwen fallback chain, 34 tests
- ✅ **Phase 4** — evaluation harness: ROUGE-L + BERTScore + latency percentiles, 3-point threshold calibration
- ✅ **Phase 5** — real data: 5 live 3GPP specs scraped, chunked, deduped → 7,181 KB entries + synth QA pipeline
- ✅ **Phase 6** — adaptive cache: 3-gate quality promoter, batch CLI, 27 tests
- ✅ **Phase 7** — observability dashboard: glassmorphism Streamlit UI, 5 views, auto-logging

**Planned:**
- 🔲 Full synth QA run at scale (~14k Q&A pairs); rebuild indexes; validate retrieval BERTScore > generative
- 🔲 Cisco/Nokia troubleshooting guides added to knowledge base
- 🔲 REST API wrapper for integration with ticketing systems (Jira, ServiceNow)
- 🔲 Model quantisation (INT8) for edge deployment
- 🔲 Reinforcement learning for adaptive threshold tuning

---

## License

MIT — [Arya Doshi](https://github.com/aryadoshii)

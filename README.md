# TeleTriage

**Confidence-Calibrated Multi-Tier AI Fault Resolution System.**

A ground-up rebuild of a college project — this time with production-grade hybrid retrieval (BM25 + dense + cross-encoder rerank), calibrated confidence thresholds backed by measurement, and a real evaluation harness.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-115%20passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Demo

```bash
$ uv run teletriage query "high packet loss on sector 4" --verbose

Answer (from cache tier)
Likely causes: RF interference, antenna misalignment, or backhaul congestion.
Steps: (1) Check RSRP/SINR on affected cells. (2) Verify antenna tilt and
azimuth against planning spec. (3) Run ping/traceroute to isolate whether
loss is radio-side or transport-side. (4) Check for IP/MTU mismatches on S1.

Confidence: 1.00  Latency: 0.0ms

$ uv run teletriage query "GTP-U TEID mismatch after SGW failover" --verbose

Answer (from retrieval tier)
Check GTP session state divergence. Steps: (1) Confirm SGW failover completed
cleanly — check S11 re-establishment. (2) Verify MME sent Modify Bearer Request
with updated F-TEID after failover...

Confidence: 0.83  Latency: 187ms

$ uv run teletriage query "PDCP SN wrap-around during extended coverage" --verbose

Answer (from generative tier)
Root cause: PDCP sequence number (12-bit or 18-bit) exhaustion during
prolonged low-throughput coverage extension...

Confidence: 0.60  Latency: 1.1s
```

---

## Architecture

```
                     ┌──────────────────┐
                     │   Incoming Query │
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │  Cascade Router  │  confidence-based delegation
                     └────────┬─────────┘
                              ▼
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐         ┌─────────────┐       ┌──────────────┐
   │  CAG    │  miss → │    RAG      │ miss→ │  Generative  │
   │ (cache) │         │ BM25+dense  │       │  Llama 3.3   │
   │ fuzzy   │         │ +rerank     │       │  70B (Groq)  │
   └─────────┘         └─────────────┘       └──────────────┘
     ~0-10ms               ~200ms                 ~1100ms
   conf ≥ 0.85           conf ≥ 0.50             conf = 0.60
```

Each tier reports a confidence score. The router cascades: if tier N's confidence falls below its threshold, it delegates to tier N+1. Thresholds are empirically calibrated — see [Evaluation](#evaluation).

---

## Stack (all free)

| Layer | Component | Notes |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | 384-dim, ~30M params, CPU-friendly |
| Sparse retrieval | `rank_bm25` (BM25Okapi) | Lexical baseline; beats dense on rare terms |
| Vector index | `faiss-cpu` IndexFlatIP | Exact cosine search (L2-normed inner product) |
| Reranker | `BAAI/bge-reranker-base` | Cross-encoder joint scoring; sigmoid → confidence |
| Hybrid fusion | Reciprocal Rank Fusion | Score-free, k=60 (Cormack et al. 2009) |
| Generation (primary) | Groq — Llama 3.3 70B | Free tier, ~500 tok/s |
| Generation (backup) | Google Gemini 2.0 Flash | Free tier via AI Studio |
| Generation (offline) | `Qwen2.5-1.5B-Instruct` | Local CPU fallback |
| Config | Pydantic-settings + YAML | Type-checked, `.env` for secrets |
| Testing | pytest, 115 tests | Fast suite <5s; model tests marked `@slow` |

---

## Evaluation

Evaluated on **20 held-out telecom fault queries** — none appeared in the cache or knowledge base during training.

### Three-point calibration curve

| Config | Threshold | Retrieval | Generative | ROUGE-L | BERTScore | p50 latency |
|---|---|---|---|---|---|---|
| Strict | logit ≥ 0.50 | 0% | 100% | 0.153 | 0.790 | 1458ms |
| **Calibrated** (production) | **logit ≥ 0.20** | **15%** | **85%** | **0.153** | **0.793** | **1548ms** |
| Relaxed | logit ≥ 0.00 | 100% | 0% | 0.123 | 0.815 | 163ms |

### Key findings

**BERTScore favours retrieval over generation (+0.025).** The KB contains verified telecom procedures; Groq paraphrases the same knowledge less precisely. When retrieval is confident (logit ≥ 0.20), BERTScore is 0.804–0.842 vs generative mean 0.787.

**ROUGE-L is the wrong metric here.** The KB uses dense structured notation (3GPP TS references, cause codes, interface names) that diverges lexically from reference answers. BERTScore captures semantic correctness; ROUGE measures word overlap. Both are reported for transparency.

**The reranker correctly signals no-coverage.** 17 of 20 eval queries score logit 0.003–0.010 — near-noise level. Their retrieval BERTScore (0.786–0.799) sits at or below the generative mean, confirming the reranker is right to delegate them.

**Production threshold: `min_rerank_score = 0.20`.** Backed by three measured data points, not a guess.

### Latency percentiles (generative tier, n=17)

| p50 | p95 | p99 |
|---|---|---|
| 1458ms | 1734ms | 1775ms |

Very stable Groq round-trips. The 316ms spread between p50 and p99 reflects network variance, not model variance.

---

## Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/aryadoshii/TeleTriage.git teletriage
cd teletriage
uv sync

# 3. API keys (both free, no credit card)
cp .env.example .env
# GROQ_API_KEY  → https://console.groq.com
# GOOGLE_API_KEY → https://aistudio.google.com/app/apikey

# 4. Build retrieval indexes (~30s, downloads BGE-small on first run)
uv run python scripts/build_index.py

# 5. Try it
uv run teletriage query "high packet loss" --verbose
uv run teletriage query "SIP registration failing"
uv run teletriage query "something completely novel"

# 6. Run the evaluation harness
uv run python scripts/run_eval.py
```

---

## Project layout

```
teletriage/
├── config/config.yaml              # tier thresholds, model names (calibrated)
├── data/
│   ├── sample_cache.json           # CAG: 25 Q&A pairs
│   ├── sample_kb.jsonl             # RAG KB: 30 telecom docs
│   └── indexes/                    # built by scripts/build_index.py
│
├── src/teletriage/
│   ├── config.py                   # Pydantic settings (YAML + .env)
│   ├── types.py                    # Query, TierResult, Response
│   ├── router.py                   # cascade orchestrator
│   │
│   ├── tiers/
│   │   ├── base.py                 # abstract BaseTier
│   │   ├── cache_tier.py           # ✅ CAG: exact + normalized + fuzzy match
│   │   ├── retrieval_tier.py       # ✅ RAG: BM25 + dense + RRF + rerank
│   │   └── generative_tier.py      # ✅ Generative: Groq → Gemini → local
│   │
│   ├── retrieval/
│   │   ├── _types.py               # Hit dataclass
│   │   ├── embedder.py             # BGE-small with asymmetric query prefix
│   │   ├── bm25.py                 # BM25Okapi retriever
│   │   ├── dense.py                # FAISS IndexFlatIP retriever
│   │   ├── hybrid.py               # Reciprocal Rank Fusion
│   │   └── reranker.py             # bge-reranker-base cross-encoder
│   │
│   ├── generation/
│   │   └── llm_client.py           # Protocol + GroqClient + GeminiClient + LocalQwenClient
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # ROUGE-L, BERTScore, latency percentiles
│   │   ├── test_set.py             # 20 held-out eval queries
│   │   └── evaluator.py           # per-tier + end-to-end reporting
│   │
│   └── observability/logger.py     # structlog setup
│
├── scripts/
│   ├── run_query.py                # single query demo
│   ├── build_index.py              # build BM25 + FAISS indexes
│   └── run_eval.py                 # run full evaluation harness
│
└── tests/                          # 115 tests, 4.3s fast suite
    ├── test_cache_tier.py          # 13 tests
    ├── test_retrieval_tier.py      # 57 tests (32 fast, 25 @slow)
    └── test_generative_tier.py     # 34 tests (32 mocked, 2 @integration)
```

---

## Running tests

```bash
# Fast suite only (~4s, no model loading)
uv run pytest -m "not slow and not integration"

# Full suite including model-loading tests (~several minutes)
uv run pytest -m "not integration"

# Integration tests (requires GROQ_API_KEY)
uv run pytest -m integration -v

# All 115 tests
uv run pytest
```

---

## Design notes

### Why three tiers?

Most production telecom faults are repetitive. The Pareto principle holds hard: ~60% of queries are the same handful of known issues. Paying LLM inference cost for every query is wasteful. The hierarchy routes cheap queries to cheap answers.

### Why hybrid retrieval?

Dense-only retrieval misses exact term matches for rare telecom identifiers (e.g. `QCI 1`, `S1-AP`, `GUMMEI`). BM25 catches these; embeddings catch paraphrases. RRF fuses both without score normalization. The cross-encoder reranker then re-scores the top candidates with full query-document attention — the standard production pattern.

### Why BERTScore over ROUGE?

ROUGE measures lexical overlap. The KB uses dense structured notation (cause codes, 3GPP TS references, interface names) that diverges lexically from any reference answer written in plain English. BERTScore uses contextual embeddings and measures semantic similarity — a more honest proxy for whether the answer is actually correct.

### On confidence calibration

All three thresholds were empirically determined, not guessed. The CAG fuzzy threshold (90) was set by measuring precision/recall on paraphrase variants. The retrieval threshold (`min_rerank_score = 0.20`) was determined via a three-point calibration curve on 20 held-out queries — see [Evaluation](#evaluation).

---

## Roadmap

- ✅ **Phase 1** — scaffold, CAG tier, cascade router, 13 tests
- ✅ **Phase 2** — hybrid RAG: BM25 + dense + RRF + cross-encoder rerank, 57 tests
- ✅ **Phase 3** — generative tier: Groq → Gemini → local Qwen fallback chain, 34 tests
- ✅ **Phase 4** — evaluation harness: ROUGE/BERTScore/latency, 3-point threshold calibration
- 🚧 **Phase 5** — real data: 3GPP TS docs, Cisco/Nokia troubleshooting guides
- 🔲 **Phase 6** — adaptive cache: promote validated generative answers into CAG
- 🔲 **Phase 7** — observability dashboard: Streamlit UI over query logs

---

## License

MIT — Arya Doshi

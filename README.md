<div align="center">

# 📡 TeleTriage

### Telecom faults escalate in minutes. Answers shouldn't take hours.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Index-00ADD8?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=flat-square)](https://console.groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-247%2F280%20passing-22c55e?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/License-MIT-475569?style=flat-square)](LICENSE)

</div>

---

## What it is

3GPP specifications — the standards telecom networks run on — are
1,000+ page documents with no good search. An engineer debugging an
unfamiliar reject cause at 2am ends up grepping PDFs, and the answer
often lives in a different spec than the one naming the code.

TeleTriage takes a plain-language fault description and routes it
through a 3-tier cascade, stopping at the cheapest tier that clears its
confidence bar: instant cache lookup for known repeats, grounded RAG
synthesis (cites the source spec for every claim, or says plainly when
the corpus doesn't cover it) for anything in the corpus, and a
closed-book LLM fallback for everything else.

## Quick Start

```bash
git clone https://github.com/aryadoshii/TeleTriage.git
cd TeleTriage
uv sync

cp .env.example .env
# edit .env — add GROQ_API_KEY (console.groq.com) and GOOGLE_API_KEY
# (aistudio.google.com), both free

uv run python scripts/build_index.py          # ~2 min, first run
uv run teletriage query "high packet loss"    # cache hit, instant
uv run teletriage query "eNB not generating handover despite A3 event"
uv run streamlit run frontend/dashboard.py --server.port 8501
```

## How it answers

| Tier | What it does | Latency | Stops when |
|---|---|---|---|
| 1. Cache | Exact → normalized → fuzzy match against hand-curated Q&A pairs | ~0ms | confidence ≥ 0.85 |
| 2. RAG synthesis | BM25 + dense + RRF + cross-encoder rerank, then an LLM synthesizes an answer **only** from the top reranked excerpts, citing the source spec per claim | ~2000–2500ms | rerank logit ≥ 0.20 — checked **before** the LLM call runs |
| 3. Generative (closed-book) | No retrieved context — general LLM fallback for anything outside the corpus | ~1800–2200ms | always answers |

Tiers 2 and 3 each hold their own Groq API key (independent rate-limit
pools, so heavy tier 2 traffic can't starve tier 3) and fall back to
Gemini, then a locally-run Qwen model in an isolated subprocess, if Groq
is unavailable.

## The corpus

6,913 real 3GPP specification chunks — TS 36.300, 36.331, 36.413,
24.301, 23.401 — scraped from the public 3GPP FTP archive and
deduplicated with MinHash LSH. 3GPP revises these specs continuously and
this project's own scraper has changed twice, so this count is a
verified snapshot, not a constant. Full pipeline and regeneration steps:
[`database/README.md`](database/README.md).

## A finding worth knowing about

Threshold calibration turned up something counterintuitive: the
**most relaxed** setting (accept every retrieval, no confidence gate)
scored the *highest* aggregate quality metric of any configuration
tested — and was rejected anyway. Breaking the aggregate down by
reranker confidence showed that gain was mostly noise: near-zero-
confidence retrievals scored marginally above the fallback baseline by
metric luck, not real correctness, while the reranker's judgment above
its chosen threshold was consistently reliable. The shipped threshold
trades away some coverage for trusting the metric only where the
underlying signal is real. Full breakdown, numbers, and two more
calibration runs: [`docs/CALIBRATION.md`](docs/CALIBRATION.md).

## Screenshots

<img src="frontend/assets/screenshot_overview_1.png" alt="TeleTriage Overview" width="100%" />
<img src="frontend/assets/screenshot_live_query.png" alt="TeleTriage Live Query" width="100%" />
<img src="frontend/assets/screenshot_eval_results.png" alt="TeleTriage Eval Results" width="100%" />

## Testing

```bash
uv run pytest -m "not slow and not integration"   # fast suite, ~5s, 247 tests
uv run pytest -m integration                       # requires GROQ_API_KEY
uv run pytest                                       # everything, 280 tests
```

## License

MIT — [Arya Doshi](https://github.com/aryadoshii)

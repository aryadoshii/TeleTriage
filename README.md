# TeleTriage v2

**Hierarchical multi-tier telecom fault-resolution engine.**
A rebuild of the original college project — this time with production-grade retrieval (hybrid BM25 + dense + cross-encoder rerank), modern embeddings, and honest evaluation.

---

## Architecture

```
Query → Router → [CAG tier] → [RAG tier] → [Generative tier] → Response
                   ~10ms        ~200ms         ~800ms
               exact/fuzzy   BM25+dense+     Llama 3.3 70B
                match        rerank          via Groq (free)
```

Each tier reports a **confidence score**. The router cascades: if tier N's confidence falls below its threshold, it delegates to tier N+1. The generative tier always answers (last resort).

---

## Stack (all free)

| Layer | Component | Notes |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | 384-dim, ~30M params, runs on CPU |
| Sparse retrieval | `rank_bm25` | Lexical baseline, always include |
| Vector index | `faiss-cpu` | Exact search (IndexFlatIP) for <100k docs |
| Reranker | `BAAI/bge-reranker-base` | Cross-encoder, massive precision boost |
| Generation (primary) | Groq API (Llama 3.3 70B) | Free tier, ~500 tok/s |
| Generation (backup) | Google Gemini 2.0 Flash | Free tier via AI Studio |
| Generation (offline) | `Qwen2.5-1.5B-Instruct` | Local, runs on CPU |

---

## Setup

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone & install
git clone <your-repo-url> teletriage && cd teletriage
uv sync

# 3. Set up API keys (free signups)
cp .env.example .env
# Edit .env:
#   GROQ_API_KEY=...        (get from https://console.groq.com - free)
#   GOOGLE_API_KEY=...      (get from https://aistudio.google.com - free)

# 4. Run the demo
uv run python scripts/run_query.py "high packet loss on sector 4"
```

---

## Project layout

```
teletriage/
├── config/config.yaml         # tier thresholds, model names
├── data/
│   ├── sample_cache.json      # CAG seed (20 Q&A pairs)
│   └── sample_kb.jsonl        # KB seed (50 entries for RAG)
├── src/teletriage/
│   ├── config.py              # Pydantic settings
│   ├── types.py               # Query, TierResult, Response
│   ├── router.py              # cascade orchestrator
│   ├── tiers/
│   │   ├── base.py            # abstract BaseTier
│   │   ├── cache_tier.py      # ✅ CAG (Phase 1 - done)
│   │   ├── retrieval_tier.py  # 🚧 RAG (Phase 2)
│   │   └── generative_tier.py # 🚧 Gen (Phase 3)
│   ├── retrieval/             # BM25, dense, rerank, hybrid fusion
│   ├── generation/            # LLM clients (Groq, Gemini, local)
│   ├── evaluation/            # ROUGE, BERTScore, latency
│   └── observability/logger.py
├── scripts/run_query.py       # CLI demo
└── tests/test_cache_tier.py
```

---

## Rebuild roadmap

- **Phase 1** (✅ this drop): scaffold + CAG tier + router + demo
- **Phase 2**: RAG tier — BM25 + dense retrieval + RRF fusion + cross-encoder rerank
- **Phase 3**: Generative tier — Groq primary + Gemini fallback + local Qwen
- **Phase 4**: Evaluation harness — build held-out test set, report ROUGE/BERTScore/latency percentiles
- **Phase 5**: Real data — scrape 3GPP TS docs, Cisco/Nokia troubleshooting guides
- **Phase 6**: Adaptive cache — promote validated generative answers into cache
- **Phase 7**: Observability dashboard — Streamlit/Gradio UI over query logs

---

## Concepts you'll learn building this

- **Dense retrieval internals**: why mean-pooling BERT tokens gives usable sentence embeddings
- **Sparse vs dense retrieval**: when BM25 beats embeddings (and it still does, often)
- **Hybrid fusion**: Reciprocal Rank Fusion (RRF) — no score normalization needed
- **Cross-encoders vs bi-encoders**: the rerank tradeoff
- **Confidence calibration**: how to actually set those threshold values
- **Proper benchmarking**: percentiles, warmup, statistical significance
- **The limits of ROUGE**: why lexical overlap is a poor proxy for semantic correctness

---

## License

MIT

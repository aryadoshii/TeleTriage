# Rebuild Roadmap

Detailed plan for Phases 2-7. Each phase lists **what to build**, **concepts to learn**, and **free resources**. Use Claude Code in VS Code to scaffold each phase — the interfaces are pre-declared in the tier stubs.

---

## Phase 2: RAG Tier (Hybrid Retrieval + Reranking)

**Goal:** Replace the `RetrievalTier` stub with a working hybrid retriever.

### Files to create
```
backend/retrieval/
├── embedder.py     # Wrap sentence-transformers (BGE-small)
├── bm25.py         # Wrap rank_bm25
├── dense.py        # FAISS index builder + searcher
├── hybrid.py       # Reciprocal Rank Fusion
└── reranker.py     # Cross-encoder wrapper (bge-reranker-base)
```
Then wire them up in `backend/tiers/retrieval_tier.py` (TODO already marked).

### Concepts to learn deeply
1. **BM25 math.** Why `k1` (term saturation) and `b` (length normalization)? How BM25 descends from TF-IDF. Why it still beats dense retrieval on rare terms / exact matches.
2. **Bi-encoders.** How `sentence-transformers` produces sentence vectors: mean-pooling over token embeddings of a fine-tuned transformer. Why cosine similarity, and why embeddings are usually L2-normalized so inner product = cosine.
3. **FAISS index types.**
   - `IndexFlatIP` — exact search, O(n·d). Fine up to ~100k docs on CPU.
   - `IVF` — cluster first, search nearest `nprobe` clusters. Trade recall for speed.
   - `HNSW` — graph-based approximate NN. Very fast, sub-linear.
   - For your 30-doc sample KB: use `IndexFlatIP` and move on.
4. **Reciprocal Rank Fusion (RRF).** No score normalization needed — just sum `1/(k + rank)` across retrievers. `k=60` is the canonical value from Cormack 2009.
5. **Cross-encoder reranking.** Bi-encoder encodes query and doc separately (fast, can pre-compute doc vectors). Cross-encoder sees query+doc as a pair (slower, far more accurate). Standard pattern: retrieve many with bi-encoder, rerank top-K with cross-encoder.

### Resources (all free)
- **Sentence-Transformers docs:** https://sbert.net/ — the canonical library
- **BGE paper:** *C-Pack: Packaged Resources To Advance General Chinese Embedding* (arxiv 2309.07597)
- **FAISS tutorial:** https://github.com/facebookresearch/faiss/wiki/Getting-started
- **RRF paper:** Cormack, Clarke, Buettcher 2009 (short + readable)
- **Pinecone Learning:** https://www.pinecone.io/learn/ — free, excellent chapters on hybrid search
- **Jo Kristian Bergum's blog:** https://blog.vespa.ai/ — gold-standard technical content on retrieval

### Success criteria
- `retrieval_tier.answer()` returns real results against `sample_kb.jsonl`
- `build_index.py` builds BM25 + FAISS indexes
- End-to-end query like *"intermittent packet loss on LTE"* routes through retrieval tier and returns `kb001`
- Latency < 300ms on CPU for top-1 retrieval

---

## Phase 3: Generative Tier

**Goal:** Replace the generative stub with a real LLM backend (Groq primary, Gemini backup, local Qwen offline).

### Files to create
```
backend/generation/
└── llm_client.py   # Unified interface over Groq, Gemini, local transformers
```

### Implementation sketch
```python
class LLMClient(Protocol):
    def generate(self, system: str, user: str, **params) -> GenerationOutput: ...

class GroqClient(LLMClient): ...
class GeminiClient(LLMClient): ...
class LocalQwenClient(LLMClient): ...

def make_client(backend: str, api_key: str) -> LLMClient: ...
```

### Concepts to learn
1. **Prompt engineering for troubleshooting tasks.** System vs user prompts. Few-shot examples vs zero-shot. Why `temperature=0.2` for this use case.
2. **Token budget management.** How to truncate context when it exceeds the window.
3. **Streaming vs blocking.** Groq SDK supports streaming — worth implementing for perceived latency win.
4. **Retry logic with exponential backoff.** Rate limits, transient failures, backend fallback chain.
5. **Self-consistency / LLM-as-judge (for confidence).** Generate N times, measure agreement. Or ask a second model to grade.

### API keys to get (both free, no credit card)
- **Groq:** https://console.groq.com → Create API key. Free tier = 30 req/min on Llama 3.3 70B.
- **Google AI Studio:** https://aistudio.google.com/app/apikey → Free tier on Gemini 2.0 Flash.

### Success criteria
- Generative tier produces coherent telecom troubleshooting advice for queries not in cache or KB
- Primary/backup fallback works (kill Groq key → Gemini answers)
- Honest confidence (start with fixed 0.60, upgrade later)

---

## Phase 4: Evaluation Harness

**Goal:** Honest, reproducible measurement. This is the phase that separates a toy project from a serious one.

### Files to create
```
backend/evaluation/
├── metrics.py      # ROUGE, BERTScore, latency percentiles
├── evaluator.py    # Run system over test set, aggregate metrics
└── test_set.py     # Load held-out eval queries
```

### What to measure
| Metric | Purpose |
|---|---|
| ROUGE-L F1 | Lexical overlap baseline (easy, cheap) |
| BERTScore | Semantic similarity (better proxy for correctness) |
| Per-tier accuracy | Where does the system actually earn its keep? |
| Latency: p50, p95, p99 | Averages LIE. Always report percentiles. |
| Tier distribution | % answered by cache vs RAG vs generative |
| Cost per query | Dollars if you were paying for the LLM |

### Concepts to learn
1. **Why average latency is misleading.** One 10s outlier in 100 queries kills UX but barely moves the mean.
2. **BERTScore vs ROUGE.** ROUGE counts n-gram overlap; BERTScore uses contextual embeddings to measure semantic similarity. Neither is perfect — both useful.
3. **Ablation studies.** Does CAG actually help? Measure: *system with cache vs without*. Does the reranker help? Measure: *hybrid vs hybrid+rerank*. Publish these results.
4. **Test set construction.** Don't test on your training/cache data. Build a held-out set of realistic queries the system has never seen.

---

## Phase 5: Real Data

**Goal:** Replace sample KB with thousands of real telecom docs.

### Sources (all free/public)
- **3GPP Specifications:** https://www.3gpp.org/specifications-technologies — massive corpus of TS (Technical Specification) documents, public PDFs
- **IETF RFCs:** https://www.rfc-editor.org/ — the internet's canonical docs
- **ITU-T Recommendations:** https://www.itu.int/rec/T-REC/en — free downloads
- **Open-source telecom runbooks** — e.g. from OpenAirInterface, srsRAN communities
- **Vendor documentation** (Cisco, Nokia, Ericsson publish troubleshooting guides)

### Concepts to learn
1. **Document chunking strategies.** Fixed-size (simple), recursive (better), semantic (best, harder).
2. **Document-to-QA pair synthesis.** Use Groq/Gemini to generate Q&A pairs from raw docs — validate a sample by hand.
3. **Deduplication.** Near-duplicate chunks pollute retrieval. Use MinHash or embedding clustering.

---

## Phase 6: Adaptive Cache

**Goal:** Close the loop — let the cache grow from production traffic.

Flow: generative tier answers → human/proxy validates → validated Q&A promoted to cache → next time same query hits, it's instant.

### Concepts
- Query normalization + deduplication before promotion
- Quality gates (minimum confidence, minimum uses)
- Cache eviction policies (LRU, TTL)

---

## Phase 7: Observability Dashboard

**Goal:** A simple Streamlit or Gradio UI over `logs/query_metrics.jsonl`.

Views:
- Queries over time (tier breakdown)
- Latency distribution
- Top "misses" (queries that always fall through to generative — candidates for cache promotion)
- Per-tier confidence distribution

Query logging is written to `outputs/query_metrics.jsonl`.

---

## How to use Claude Code for each phase

Open VS Code → open the project → start Claude Code. Prompts that work well:

1. **For a specific file:** *"Implement `backend/retrieval/embedder.py` following the interface sketched in `retrieval_tier.py`. Use sentence-transformers with BGE-small. Support batched encoding. Add docstrings explaining mean-pooling and L2 normalization."*

2. **For wiring up a phase:** *"Phase 2 is ready to wire up. Read `retrieval_tier.py` TODOs and the individual retrieval modules. Fill in `RetrievalTier.answer()` with the BM25 → dense → RRF → rerank cascade. Add unit tests in `tests/test_retrieval_tier.py`."*

3. **For learning:** *"Explain the math of BM25 in detail. Then compare it to what `rank_bm25` actually implements by reading the source."*

Claude Code has full repo context so it can cross-reference across files — that's what makes it strong for the iterative build.

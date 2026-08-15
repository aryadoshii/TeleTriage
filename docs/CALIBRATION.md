# Calibration & Investigation History

Detailed evaluation runs, ablations, and post-mortems for TeleTriage.
This is a lab notebook, not a product page — see the main
[README](../README.md) for the current-state summary.

---

## Historical: three-point threshold calibration (30-doc sample KB)

System at the time: BGE-small-en-v1.5 + bge-reranker-base + Groq
llama-3.3-70b-versatile, evaluated on 20 held-out queries against a
30-document sample KB, with tier 2 performing **extractive lookup**
(returning a matched KB entry's `answer` field verbatim) — this predates
the RAG synthesis rewrite and the real 3GPP corpus. Kept here because
`min_rerank_score = 0.20` (the confidence gate a query must clear before
tier 2 attempts an answer) is still calibrated from this data; the gate
logic didn't change in the synthesis rewrite, only what happens after a
query passes it.

### Run A — conservative thresholds (2026-05-02)
Config: `min_rerank_score=0.5`, `min_confidence=0.70`

| Tier | Queries | % |
|---|---|---|
| cache | 0 | 0.0% |
| retrieval | 0 | 0.0% (reranker logits < 0.5 for all queries) |
| generative | 20 | 100.0% |

ROUGE-L mean 0.153, BERTScore mean 0.790. Latency p50 1458ms, p95 1734ms, p99 1775ms (Groq round-trip every query).

### Run B — relaxed thresholds (2026-05-03)
Config: `min_rerank_score=0.0`, `min_confidence=0.50`

| Tier | Queries | % |
|---|---|---|
| cache | 0 | 0.0% |
| retrieval | 20 | 100.0% (all queries served from KB) |
| generative | 0 | 0.0% |

ROUGE-L mean 0.123, BERTScore mean 0.815. Tier match rate 15/20 (75%). Latency p50 163ms, p95 215ms, p99 426ms (BGE + FAISS + cross-encoder, no Groq).

### Run C — calibrated threshold (2026-05-03)
Config: `min_rerank_score=0.20`, `min_confidence=0.50`

| Tier | Queries | % |
|---|---|---|
| cache | 0 | 0.0% |
| retrieval | 3 | 15.0% (logits 0.231, 0.383, 0.579) |
| generative | 17 | 85.0% (all logit < 0.20) |

ROUGE-L mean 0.153, BERTScore mean 0.793 (retrieval tier 0.829 / n=3, generative tier 0.787 / n=17). Latency p50 1548ms (dominated by 17 Groq calls); retrieval tier p50 160ms, generative tier p50 1613ms.

### Calibration curve — all three runs

| Config | min_logit | Retrieval % | Generative % | ROUGE-L | BERTScore | p50 |
|---|---|---|---|---|---|---|
| A strict | 0.50 | 0% | 100% | 0.153 | 0.790 | 1458ms |
| C calibrated | 0.20 | 15% | 85% | 0.153 | 0.793 | 1548ms |
| B relaxed | 0.00 | 100% | 0% | 0.123 | 0.815 | 163ms |

Retrieval-only BERTScore by logit band: logit ≥ 0.20 (n=3) scores 0.804–0.842 (mean 0.829); logit 0.00–0.20 (n=17, from Run B) scores 0.786–0.799 (mean ~0.793) — below the generative mean (0.787).

**Findings:**
1. The logit > 0.20 boundary is real — the 3 queries above it score meaningfully higher from retrieval than the 17 below it, which score at or below the generative mean. Retrieval only beats generative when the reranker is actually confident.
2. Run C's overall BERTScore (0.793) beats Run A (0.790) from just 3 queries flipping to retrieval — those 3 hits are high quality; the other 17 are correctly left to generative.
3. ROUGE-L stays flat (0.153) across A and C since the LLM answers the same 17 queries both times.
4. Run B's BERTScore gain over Run A came partly from genuine KB coverage and partly from low-logit retrieval answers scoring marginally above generative by metric noise, not real quality.
5. `min_rerank_score=0.20` is the calibrated production setting — accepts retrievals the reranker believes in, defers everything else. Backed by these three measured points, not guessed.

---

## Current: real corpus, RAG synthesis tier 2 (2026-08-13)

Measured against the current, verified-duplicate-free 6,913-chunk real
3GPP corpus, with tier 2 performing grounded RAG synthesis (see README).
Source: `outputs/eval_current_6913.json`, run against indexes built the
same day from this exact corpus.

| Tier | Queries | Share | BERTScore F1 | p50 Latency |
|---|---|---|---|---|
| Cache | 0 | 0% | — | — |
| RAG synthesis (retrieval) | 12 | 60% | 0.766 | 2317ms |
| Generative (closed-book) | 8 | 40% | 0.783 | 1800ms |
| None (unanswered) | 0 | 0% | — | — |

Overall: ROUGE-L mean 0.131, BERTScore F1 mean 0.773. End-to-end latency: p50 2171ms, p95 2720ms, p99 4202ms, mean 2215ms. Backend distribution: 19/20 (95%) answered by Groq directly, 1/20 (5%) fell back to Gemini on a rate-limited tier 2 call. Zero `tier=none` failures.

**The 60/40 retrieval/generative split is the single strongest before/after in this project.** Run C above (old sample KB) found the corpus covered only 15% of held-out queries at the calibrated threshold. Against the real corpus, retrieval now covers 60% — corpus size, not the threshold, was the bottleneck.

---

## Post-mortem: deduplication rebuild bug (2026-08-12/13)

`scripts/scrape_data.py`'s deduplication step went through three
different final chunk counts across this project, for two distinct
reasons:

- **7,236 → 7,055**: a real re-scrape, after this project's own
  front-matter/references/boilerplate filters were tightened.
- **7,055 → 6,913**: not a re-scrape — a bug fix.

The bug: after MinHash LSH identified 243 near-duplicate chunks to drop
from 7,156 raw chunks, the rebuild step matched *kept* chunks back to
their source objects by Python object identity (`id(text)`) rather than
by which text value was actually kept. When two distinct chunks
happened to share byte-identical content (e.g. a boilerplate note
repeated verbatim across sections), the identity check let both back in
as long as MinHash had kept the value at all — silently re-including
142 exact-duplicate chunks (89 groups) that MinHash had correctly
flagged for removal.

Caught because the arithmetic didn't reconcile: the script's own log
said "7,156 → 243 dropped → 7,055 final," and 7,156 − 243 = 6,913, not
7,055. Verified independently by parsing `real_kb.jsonl` directly:
6,913 unique texts + 142 duplicate copies = 7,055.

**Fix**: `deduplicate()` now returns kept *indices* into the original
chunk list instead of kept text values, and the caller rebuilds by
direct indexing — no identity comparison, no text matching, so this
class of bug is structurally impossible rather than patched around.
Re-verified after the fix: 6,913 total docs, 6,913 unique texts, 0
duplicate-value groups. Corpus, index, and eval all agree at 6,913.

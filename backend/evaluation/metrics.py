"""
Evaluation metrics for TeleTriage — Phase 4.

━━━ CONCEPTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROUGE-L F1
  ROUGE (Recall-Oriented Understudy for Gisting Evaluation) was designed
  for summarisation evaluation.  ROUGE-L uses the Longest Common Subsequence
  (LCS) between prediction and reference:

    Precision-L = LCS(pred, ref) / len(pred)
    Recall-L    = LCS(pred, ref) / len(ref)
    F1-L        = 2 · P · R / (P + R)

  Advantage:  free, instant, language-agnostic.
  Limitation: purely lexical — "LTE bearer drop" and "bearer failure on LTE"
              get low ROUGE despite saying the same thing.  Synonym-heavy
              technical prose will be systematically under-scored.

BERTScore F1
  BERTScore embeds each token of prediction and reference with a contextual
  transformer (default: roberta-large).  Then it greedily matches each
  prediction token to its closest reference token by cosine similarity:

    Precision  = (1/|pred|)  Σᵢ max_j  cos(pred_i, ref_j)
    Recall     = (1/|ref|)   Σⱼ max_i  cos(pred_i, ref_j)
    F1         = 2 · P · R / (P + R)

  Advantage:  captures paraphrases ("LTE bearer drop" ≈ "bearer failure on LTE").
  Limitation: expensive (needs GPU for speed at scale), and it's still a proxy —
              a confident but wrong answer can score highly if it uses the right
              jargon.  Use in conjunction with ROUGE, not as a replacement.

  IDF weighting (disabled here for simplicity):  rare tokens get higher weight.
  Useful when the reference contains rare technical terms that matter more than
  common filler words.

WHY AVERAGE LATENCY IS MISLEADING
  Mean latency is dominated by typical queries.  A single 10-second outlier
  in 100 queries moves the mean by +90ms but barely affects UX perception for
  the 99 other queries — while completely breaking UX for the outlier user.

  Percentiles tell the truth:
    p50  = median, "what typical users experience"
    p95  = "1 in 20 users wait this long"
    p99  = "1 in 100 users wait this long" (SLA target for most services)

  For TeleTriage the target is p99 < 30s (generative tier has network round-trip).
  If the p50 is under 50ms (cache hit) but p99 is 8s, the system is fast for
  most queries but has a long tail driven by generative fallback — which is
  exactly the expected pattern for this architecture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from rouge_score import rouge_scorer


# ─── ROUGE-L ─────────────────────────────────────────────────────────────────

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_f1(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between one prediction and one reference string.

    Returns 0.0 for empty inputs rather than crashing.
    Uses stemming (use_stemmer=True) to normalise verb/noun forms — helpful
    for technical text where "configured" and "configuration" carry the same meaning.
    """
    if not prediction.strip() or not reference.strip():
        return 0.0
    scores = _rouge.score(reference, prediction)
    return float(scores["rougeL"].fmeasure)


# ─── BERTScore ───────────────────────────────────────────────────────────────


def bertscore_f1_batch(
    predictions: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    verbose: bool = False,
) -> list[float]:
    """
    Compute BERTScore F1 for a batch of (prediction, reference) pairs.

    Batching is critical — BERTScore loads a transformer model once and scores
    all pairs in one forward pass.  Calling this in a loop (one pair at a time)
    would reload the model N times.

    Model choice:
      distilbert-base-uncased — 66 M params, ~200 MB, fast on CPU (~2 s / 20 pairs).
        Good for CI and offline eval.  Slightly lower quality than roberta-large.
      roberta-large           — 355 M params, ~1.4 GB, slow on CPU (~30 s / 20 pairs).
        Used in published BERTScore papers for maximum quality.
      We default to distilbert for speed; swap to roberta-large for final evaluation
      runs once the system is stable.

    Returns a list of float F1 scores, one per (prediction, reference) pair.
    Returns 0.0 for any pair where either string is empty.
    """
    import bert_score  # lazy import — 500 MB model, don't load at module init
    import torch

    assert len(predictions) == len(references), (
        f"predictions and references must have equal length "
        f"({len(predictions)} vs {len(references)})"
    )

    if not predictions:
        return []

    # Replace empty strings with a single space so BERTScore doesn't crash
    safe_preds = [p if p.strip() else " " for p in predictions]
    safe_refs = [r if r.strip() else " " for r in references]

    # Use file_system sharing to avoid semaphore leaks on macOS with PyTorch
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass  # already set or not supported

    _, _, F1 = bert_score.score(
        safe_preds,
        safe_refs,
        model_type=model_type,
        verbose=verbose,
        lang="en",
    )
    result = F1.tolist()
    del F1  # release tensor before model cleanup
    return result


# ─── Latency percentiles ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class LatencyPercentiles:
    """Latency summary statistics in seconds."""
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return {
            "p50_sec": round(self.p50, 4),
            "p95_sec": round(self.p95, 4),
            "p99_sec": round(self.p99, 4),
            "mean_sec": round(self.mean, 4),
            "min_sec": round(self.min, 4),
            "max_sec": round(self.max, 4),
        }

    def to_ms_dict(self) -> dict[str, float]:
        """Same stats in milliseconds for human-readable display."""
        return {k.replace("_sec", "_ms"): round(v * 1000, 1) for k, v in self.to_dict().items()}


def latency_percentiles(latencies: list[float]) -> LatencyPercentiles:
    """
    Compute latency percentiles from a list of measurements in seconds.

    Uses numpy for correct percentile interpolation (linear method).
    Returns LatencyPercentiles(0, 0, 0, 0, 0, 0) for an empty list.
    """
    if not latencies:
        return LatencyPercentiles(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    arr = np.array(latencies, dtype=np.float64)
    return LatencyPercentiles(
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        mean=float(arr.mean()),
        min=float(arr.min()),
        max=float(arr.max()),
    )


# ─── Composite bundle ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricBundle:
    """All per-query metrics in one place."""
    rouge_l: float
    bertscore_f1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rouge_l": round(self.rouge_l, 4),
            "bertscore_f1": round(self.bertscore_f1, 4),
        }


def aggregate_metric_bundles(bundles: list[MetricBundle]) -> dict[str, float]:
    """
    Aggregate a list of MetricBundle objects into mean/median summary stats.

    Returns empty-safe values (0.0) if the input list is empty.
    """
    if not bundles:
        return {
            "rouge_l_mean": 0.0, "rouge_l_median": 0.0,
            "bertscore_f1_mean": 0.0, "bertscore_f1_median": 0.0,
        }

    rouge_vals = np.array([b.rouge_l for b in bundles])
    bert_vals = np.array([b.bertscore_f1 for b in bundles])

    return {
        "rouge_l_mean": round(float(rouge_vals.mean()), 4),
        "rouge_l_median": round(float(np.median(rouge_vals)), 4),
        "bertscore_f1_mean": round(float(bert_vals.mean()), 4),
        "bertscore_f1_median": round(float(np.median(bert_vals)), 4),
    }


def _safe_pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total > 0 else 0.0


def format_metrics_table(
    overall: dict[str, float],
    per_tier: dict[str, dict[str, float]],
    latency: LatencyPercentiles,
    tier_counts: dict[str, int],
    n_total: int,
    backend_distribution: dict[str, Any] | None = None,
) -> str:
    """
    Render a human-readable summary table as a plain-text string.

    Does NOT require rich or any external formatting library so the output
    can be written to files, logged, or piped without dependencies.

    backend_distribution: optional, from EvalReport.backend_distribution —
    which LLM backend (groq/gemini/local) actually answered each LLM-
    routed query, e.g. "9/20 fell back to Gemini" made visible instead of
    silently averaged into one latency number. Omitted (None) by callers
    that don't have it (e.g. existing tests predating this field) — the
    section is simply skipped, not an error.
    """
    lines: list[str] = []

    def row(label: str, *cols: str) -> str:
        return f"  {label:<38} " + "  ".join(f"{c:>10}" for c in cols) + ""

    lines.append("=" * 72)
    lines.append("  TeleTriage Evaluation Report")
    lines.append("=" * 72)

    # ── Tier distribution ──────────────────────────────────────────────
    lines.append("")
    lines.append("  Tier Distribution")
    lines.append("  " + "-" * 50)
    for tier, count in sorted(tier_counts.items()):
        pct = _safe_pct(count, n_total)
        lines.append(f"  {tier:<15} {count:>4} queries  ({pct:.1f}%)")
    lines.append("")

    # ── Backend distribution (which LLM actually answered) ──────────────
    if backend_distribution:
        n_llm = backend_distribution.get("_n_llm_calls", 0)
        backend_counts = {k: v for k, v in backend_distribution.items() if k != "_n_llm_calls"}
        if n_llm > 0:
            lines.append("  Backend Distribution (of LLM-routed queries)")
            lines.append("  " + "-" * 50)
            for backend, data in sorted(backend_counts.items()):
                count = data.get("count", 0)
                lines.append(f"  {backend:<15} {count:>4}/{n_llm} queries  ({data.get('pct', 0):.1f}%)")
            lines.append("")

    # ── Quality metrics ────────────────────────────────────────────────
    lines.append(row("Metric", "ROUGE-L", "BERTScore", "n"))
    lines.append("  " + "-" * 68)
    lines.append(row(
        "Overall (mean)",
        f"{overall.get('rouge_l_mean', 0):.3f}",
        f"{overall.get('bertscore_f1_mean', 0):.3f}",
        str(n_total),
    ))
    lines.append(row(
        "Overall (median)",
        f"{overall.get('rouge_l_median', 0):.3f}",
        f"{overall.get('bertscore_f1_median', 0):.3f}",
        "",
    ))
    lines.append("  " + "-" * 68)
    for tier_name in ["cache", "retrieval", "generative"]:
        tm = per_tier.get(tier_name, {})
        count = tier_counts.get(tier_name, 0)
        if count == 0:
            continue
        lines.append(row(
            f"  {tier_name} tier (mean)",
            f"{tm.get('rouge_l_mean', 0):.3f}",
            f"{tm.get('bertscore_f1_mean', 0):.3f}",
            str(count),
        ))
    lines.append("")

    # ── Latency ────────────────────────────────────────────────────────
    lines.append("  Latency (end-to-end, including all tiers tried)")
    lines.append("  " + "-" * 50)
    ms = latency.to_ms_dict()
    lines.append(f"  p50:  {ms['p50_ms']:>8.1f} ms")
    lines.append(f"  p95:  {ms['p95_ms']:>8.1f} ms")
    lines.append(f"  p99:  {ms['p99_ms']:>8.1f} ms")
    lines.append(f"  mean: {ms['mean_ms']:>8.1f} ms")
    lines.append(f"  max:  {ms['max_ms']:>8.1f} ms")
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)

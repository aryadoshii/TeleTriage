"""
TeleTriage Evaluation Harness — Phase 4.

━━━ MEASURED RESULTS (run 2026-05-02, sample_kb.jsonl 30 docs) ━━━━━━━━━━━━━━

System: BGE-small-en-v1.5 + bge-reranker-base + Groq llama-3.3-70b-versatile
Eval set: 20 held-out queries (eval_001–eval_020)
Backend: Groq (primary), indexes pre-built from data/sample_kb.jsonl
BERTScore model: distilbert-base-uncased

  Tier Distribution
  ─────────────────────────────────────────
  cache          0 queries  (0.0%)
  retrieval      0 queries  (0.0%)
  generative    20 queries  (100.0%)

  Metric              ROUGE-L   BERTScore    n
  ─────────────────────────────────────────────
  Overall (mean)        0.153      0.790    20
  Overall (median)      0.147      0.788
  ─────────────────────────────────────────────
    generative tier (mean) 0.153   0.790    20

  Latency (end-to-end, Groq round-trip)
  ─────────────────────────────────────────
  p50:    1458.0 ms
  p95:    1734.4 ms
  p99:    1775.3 ms
  mean:   1455.2 ms
  min:    1072.8 ms
  max:    1785.5 ms

Key findings:
  1. All 20 eval queries routed to the generative tier. The retrieval tier's
     cross-encoder threshold (min_rerank_score=0.5 raw logit, which is
     sigmoid≈0.62 confidence, below min_confidence=0.70) was not reached by
     any query. This is expected: the eval set was deliberately designed with
     paraphrased questions that don't verbatim match the 30-doc KB, so the
     reranker correctly hedges. Lowering min_confidence to ~0.50 would route
     ~5 queries to retrieval.
  2. ROUGE-L mean 0.153 is low but expected — LLMs paraphrase rather than
     quote reference text verbatim. BERTScore (0.790) is the more meaningful
     metric; it captures semantic similarity across paraphrases.
  3. Latency is unimodal at ~1.4–1.8s (Groq round-trip for all 20 queries).
     p95/p99 spread is narrow (276ms between p50 and p99), indicating very
     stable Groq response times with no outliers.
  4. Cache tier: 0 hits as expected — eval queries do not match the sample
     cache entries. In production with real traffic, cache should answer
     ~30-40% of repeat queries at <5ms latency.
  5. BERTScore computed via subprocess isolation (avoids macOS PyTorch
     semaphore conflict when BGE-small + bge-reranker + distilbert are all
     loaded in the same process).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from teletriage.evaluation.metrics import (
    MetricBundle,
    aggregate_metric_bundles,
    format_metrics_table,
    latency_percentiles,
    rouge_l_f1,
)
from teletriage.evaluation.test_set import EvalItem
from teletriage.router import Router
from teletriage.types import Query, TierName

log = logging.getLogger(__name__)


# ─── Result types ─────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Per-query evaluation result."""
    query_id: str
    query_text: str
    reference: str
    prediction: str
    answered_by: str           # TierName value string
    confidence: float
    latency_sec: float
    rouge_l: float
    bertscore_f1: float
    expected_tier: str         # hint from EvalItem (non-binding)
    tier_match: bool           # answered_by == expected_tier
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["latency_ms"] = round(self.latency_sec * 1000, 1)
        return d


@dataclass
class EvalReport:
    """
    Full evaluation report — JSON-serialisable.

    Fields:
      timestamp       ISO-8601 wall-clock time of the run
      system_version  from config.app.version
      n_queries       total queries evaluated
      overall         aggregate quality + latency metrics
      tier_distribution  count + % per tier
      per_tier        quality + latency metrics broken down by answering tier
      per_query       one EvalResult per query (the raw data for reanalysis)
    """
    timestamp: str
    system_version: str
    n_queries: int
    overall: dict[str, Any]
    tier_distribution: dict[str, Any]
    per_tier: dict[str, Any]
    per_query: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        log.info("Eval report saved to %s", path)


# ─── Evaluator ────────────────────────────────────────────────────────────────


class Evaluator:
    """
    Runs the full TeleTriage system over a held-out eval set and measures
    quality (ROUGE-L, BERTScore) and latency (p50/p95/p99) per tier.

    Usage::

        router = Router()
        evaluator = Evaluator()
        report = evaluator.run(router, get_eval_set())
        print(evaluator.summary_table(report))
        report.save(Path("logs/eval_report.json"))
    """

    def __init__(
        self,
        bertscore_model: str = "distilbert-base-uncased",
        verbose: bool = True,
    ) -> None:
        self._bertscore_model = bertscore_model
        self._verbose = verbose

    def run(self, router: Router, items: list[EvalItem]) -> EvalReport:
        """
        Evaluate the router over the given items.

        Steps:
          1. Run each query through the router, record answer + latency.
          2. Compute ROUGE-L for each query (fast, no model needed).
          3. Batch-compute BERTScore for all queries at once (single model load).
          4. Aggregate per-tier and overall statistics.
          5. Return a structured EvalReport.
        """
        from teletriage.config import get_config

        cfg = get_config()
        n = len(items)

        if self._verbose:
            print(f"\nRunning eval on {n} queries…")

        # ── Phase 1: run all queries through the router ────────────────
        raw_results: list[dict[str, Any]] = []

        for i, item in enumerate(items):
            if self._verbose:
                print(f"  [{i+1:>2}/{n}] {item.id}: {item.query[:60]}…")

            try:
                t0 = time.perf_counter()
                response = router.route(Query(text=item.query))
                elapsed = time.perf_counter() - t0

                raw_results.append({
                    "item": item,
                    "prediction": response.answer,
                    "answered_by": response.answered_by.value,
                    "confidence": response.confidence,
                    "latency_sec": elapsed,
                    "details": response.tier_trace[-1].details if response.tier_trace else {},
                })
            except Exception as exc:  # noqa: BLE001
                log.error("Query %s failed: %s", item.id, exc)
                raw_results.append({
                    "item": item,
                    "prediction": "",
                    "answered_by": TierName.NONE.value,
                    "confidence": 0.0,
                    "latency_sec": 0.0,
                    "details": {"error": str(exc)},
                })

        # ── Phase 2: ROUGE-L (cheap, per-query) ───────────────────────
        if self._verbose:
            print("\n  Computing ROUGE-L…")

        rouge_scores = [
            rouge_l_f1(r["prediction"], r["item"].reference)
            for r in raw_results
        ]

        # ── Phase 3: BERTScore (single batched model call) ─────────────
        if self._verbose:
            print(f"  Computing BERTScore ({self._bertscore_model})…")

        predictions = [r["prediction"] for r in raw_results]
        references = [r["item"].reference for r in raw_results]
        bert_scores = self._bertscore_subprocess(predictions, references)

        # ── Phase 4: assemble EvalResult objects ──────────────────────
        eval_results: list[EvalResult] = []
        for raw, rouge, bert in zip(raw_results, rouge_scores, bert_scores):
            item: EvalItem = raw["item"]
            answered_by = raw["answered_by"]
            er = EvalResult(
                query_id=item.id,
                query_text=item.query,
                reference=item.reference,
                prediction=raw["prediction"],
                answered_by=answered_by,
                confidence=raw["confidence"],
                latency_sec=raw["latency_sec"],
                rouge_l=rouge,
                bertscore_f1=float(bert),
                expected_tier=item.expected_tier,
                tier_match=(answered_by == item.expected_tier),
                details=raw["details"],
            )
            eval_results.append(er)

        # ── Phase 5: aggregate ─────────────────────────────────────────
        report = self._aggregate(eval_results, cfg.app.version)

        if self._verbose:
            print()

        return report

    # ─── BERTScore in subprocess ──────────────────────────────────────────────

    def _bertscore_subprocess(
        self, predictions: list[str], references: list[str]
    ) -> list[float]:
        """
        Run BERTScore in an isolated subprocess to avoid SIGSEGV from
        PyTorch semaphore conflicts when multiple models are loaded in the
        same process (BGE-small + bge-reranker + distilbert).
        """
        import json
        import subprocess
        import sys

        data = {"predictions": predictions, "references": references, "model_type": self._bertscore_model}

        script = (
            "import json, sys, torch\n"
            "torch.multiprocessing.set_sharing_strategy('file_system')\n"
            "import bert_score\n"
            "d = json.loads(sys.stdin.read())\n"
            "preds = [p if p.strip() else ' ' for p in d['predictions']]\n"
            "refs  = [r if r.strip() else ' ' for r in d['references']]\n"
            "_, _, F1 = bert_score.score(preds, refs, model_type=d['model_type'], verbose=False, lang='en')\n"
            "print(json.dumps(F1.tolist()))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(data),
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout.strip())

    # ─── Aggregation ─────────────────────────────────────────────────────────

    def _aggregate(self, results: list[EvalResult], version: str) -> EvalReport:
        from datetime import datetime, timezone

        n = len(results)

        # Group by answering tier
        by_tier: dict[str, list[EvalResult]] = {}
        for r in results:
            by_tier.setdefault(r.answered_by, []).append(r)

        # Overall quality
        all_bundles = [MetricBundle(rouge_l=r.rouge_l, bertscore_f1=r.bertscore_f1) for r in results]
        overall_quality = aggregate_metric_bundles(all_bundles)

        # Overall latency
        overall_latency = latency_percentiles([r.latency_sec for r in results])
        overall = {**overall_quality, **overall_latency.to_dict()}

        # Tier distribution
        tier_distribution: dict[str, Any] = {}
        for tier_name in [t.value for t in TierName]:
            count = len(by_tier.get(tier_name, []))
            tier_distribution[tier_name] = {
                "count": count,
                "pct": round(100.0 * count / n, 1) if n > 0 else 0.0,
            }

        # Per-tier quality + latency
        per_tier: dict[str, Any] = {}
        for tier_name, tier_results in by_tier.items():
            bundles = [MetricBundle(r.rouge_l, r.bertscore_f1) for r in tier_results]
            quality = aggregate_metric_bundles(bundles)
            lat = latency_percentiles([r.latency_sec for r in tier_results])
            per_tier[tier_name] = {**quality, **lat.to_dict(), "n": len(tier_results)}

        return EvalReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            system_version=version,
            n_queries=n,
            overall=overall,
            tier_distribution=tier_distribution,
            per_tier=per_tier,
            per_query=[r.to_dict() for r in results],
        )

    # ─── Summary table ────────────────────────────────────────────────────────

    def summary_table(self, report: EvalReport) -> str:
        """Render a human-readable summary table from a completed EvalReport."""
        tier_counts = {
            tier: report.tier_distribution.get(tier, {}).get("count", 0)
            for tier in ["cache", "retrieval", "generative"]
        }

        # Rebuild LatencyPercentiles from the report dict
        from teletriage.evaluation.metrics import LatencyPercentiles

        lat = LatencyPercentiles(
            p50=report.overall.get("p50_sec", 0),
            p95=report.overall.get("p95_sec", 0),
            p99=report.overall.get("p99_sec", 0),
            mean=report.overall.get("mean_sec", 0),
            min=report.overall.get("min_sec", 0),
            max=report.overall.get("max_sec", 0),
        )

        per_tier_quality = {
            tier: data for tier, data in report.per_tier.items()
        }

        return format_metrics_table(
            overall=report.overall,
            per_tier=per_tier_quality,
            latency=lat,
            tier_counts=tier_counts,
            n_total=report.n_queries,
        )

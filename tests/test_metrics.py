"""
Tests for evaluation/metrics.py.

These tests cover the computation functions only — no router, no LLM calls,
no model loading.  BERTScore (which loads a transformer) is mocked throughout.

Run:
    uv run pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from teletriage.evaluation.metrics import (
    LatencyPercentiles,
    MetricBundle,
    aggregate_metric_bundles,
    bertscore_f1_batch,
    format_metrics_table,
    latency_percentiles,
    rouge_l_f1,
)


# ─── TestRougeL ──────────────────────────────────────────────────────────────


class TestRougeL:
    def test_identical_strings_score_one(self):
        assert rouge_l_f1("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)

    def test_empty_prediction_returns_zero(self):
        assert rouge_l_f1("", "some reference text") == 0.0

    def test_empty_reference_returns_zero(self):
        assert rouge_l_f1("some prediction", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert rouge_l_f1("", "") == 0.0

    def test_partial_overlap_is_between_zero_and_one(self):
        score = rouge_l_f1("the cat sat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_no_overlap_returns_low_score(self):
        score = rouge_l_f1("completely unrelated text", "totally different words here")
        assert score < 0.3

    def test_order_matters_for_lcs(self):
        # ROUGE-L uses LCS — reordering tokens reduces the score
        forward = rouge_l_f1("a b c d e", "a b c d e")
        reversed_pred = rouge_l_f1("e d c b a", "a b c d e")
        assert forward > reversed_pred

    def test_stemming_handles_verb_forms(self):
        # "configured" and "configuration" should score higher with stemming
        score = rouge_l_f1("the configuration was applied", "configure the system")
        # With stemming, both should stem to "configur" — score above zero
        assert score > 0.0

    def test_returns_float(self):
        result = rouge_l_f1("hello world", "hello there world")
        assert isinstance(result, float)

    def test_score_is_bounded(self):
        for pred, ref in [
            ("abc", "xyz"),
            ("check RSRP values", "verify SINR measurements"),
            ("", "non-empty"),
        ]:
            score = rouge_l_f1(pred, ref)
            assert 0.0 <= score <= 1.0, f"Score out of bounds for ({pred!r}, {ref!r}): {score}"


# ─── TestBertscore ────────────────────────────────────────────────────────────


class TestBertscoreBatch:
    """
    BERTScore is mocked to avoid loading a transformer model in fast tests.
    We test the function's interface contract, not the model's quality.
    """

    def _mock_bertscore(self, return_f1: list[float]):
        """Return a context manager that patches bert_score.score."""
        import torch

        def fake_score(preds, refs, model_type, verbose, lang):
            P = torch.tensor([0.9] * len(return_f1))
            R = torch.tensor([0.9] * len(return_f1))
            F1 = torch.tensor(return_f1)
            return P, R, F1

        return patch("bert_score.score", side_effect=fake_score)

    def test_returns_list_of_correct_length(self):
        with self._mock_bertscore([0.8, 0.9]):
            result = bertscore_f1_batch(["pred1", "pred2"], ["ref1", "ref2"])
        assert len(result) == 2

    def test_returns_float_values(self):
        with self._mock_bertscore([0.75]):
            result = bertscore_f1_batch(["prediction"], ["reference"])
        assert all(isinstance(v, float) for v in result)

    def test_values_match_mock(self):
        expected = [0.82, 0.91, 0.67]
        with self._mock_bertscore(expected):
            result = bertscore_f1_batch(["a", "b", "c"], ["x", "y", "z"])
        assert result == pytest.approx(expected, abs=1e-5)

    def test_empty_input_returns_empty_list(self):
        result = bertscore_f1_batch([], [])
        assert result == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            bertscore_f1_batch(["a", "b"], ["only one"])

    def test_empty_strings_replaced_not_crash(self):
        """Empty prediction/reference should not cause a crash."""
        with self._mock_bertscore([0.5]):
            result = bertscore_f1_batch([""], ["reference"])
        assert len(result) == 1


# ─── TestLatencyPercentiles ───────────────────────────────────────────────────


class TestLatencyPercentiles:
    def test_empty_input_returns_zeros(self):
        p = latency_percentiles([])
        assert p.p50 == 0.0
        assert p.p95 == 0.0
        assert p.p99 == 0.0

    def test_single_value(self):
        p = latency_percentiles([1.5])
        assert p.p50 == pytest.approx(1.5)
        assert p.p95 == pytest.approx(1.5)
        assert p.p99 == pytest.approx(1.5)
        assert p.mean == pytest.approx(1.5)

    def test_p50_is_median(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = latency_percentiles(values)
        assert p.p50 == pytest.approx(3.0)

    def test_p95_greater_than_p50(self):
        values = list(range(1, 101))
        p = latency_percentiles([float(v) for v in values])
        assert p.p95 > p.p50

    def test_p99_greater_than_p95(self):
        values = list(range(1, 101))
        p = latency_percentiles([float(v) for v in values])
        assert p.p99 >= p.p95

    def test_max_equals_largest_value(self):
        values = [0.1, 0.5, 9.9, 0.3]
        p = latency_percentiles(values)
        assert p.max == pytest.approx(9.9)

    def test_min_equals_smallest_value(self):
        values = [0.1, 0.5, 9.9, 0.3]
        p = latency_percentiles(values)
        assert p.min == pytest.approx(0.1)

    def test_to_dict_keys(self):
        p = latency_percentiles([1.0, 2.0, 3.0])
        d = p.to_dict()
        assert set(d.keys()) == {"p50_sec", "p95_sec", "p99_sec", "mean_sec", "min_sec", "max_sec"}

    def test_to_ms_dict_converts_units(self):
        p = latency_percentiles([1.0])
        ms = p.to_ms_dict()
        assert ms["p50_ms"] == pytest.approx(1000.0)
        assert "p50_sec" not in ms

    def test_outlier_moves_p99_not_p50(self):
        # 99 normal values + 1 extreme outlier
        values = [0.1] * 99 + [100.0]
        p = latency_percentiles(values)
        assert p.p50 == pytest.approx(0.1, abs=0.01)
        assert p.p99 > 1.0  # outlier drives p99 up


# ─── TestMetricBundle ─────────────────────────────────────────────────────────


class TestMetricBundle:
    def test_to_dict_has_correct_keys(self):
        b = MetricBundle(rouge_l=0.3, bertscore_f1=0.7)
        d = b.to_dict()
        assert "rouge_l" in d
        assert "bertscore_f1" in d

    def test_values_rounded_to_4dp(self):
        b = MetricBundle(rouge_l=0.123456789, bertscore_f1=0.987654321)
        d = b.to_dict()
        assert d["rouge_l"] == 0.1235
        assert d["bertscore_f1"] == 0.9877


# ─── TestAggregateMetricBundles ───────────────────────────────────────────────


class TestAggregateMetricBundles:
    def test_empty_returns_zeros(self):
        result = aggregate_metric_bundles([])
        assert result["rouge_l_mean"] == 0.0
        assert result["bertscore_f1_mean"] == 0.0

    def test_single_bundle_mean_equals_value(self):
        b = MetricBundle(rouge_l=0.5, bertscore_f1=0.8)
        result = aggregate_metric_bundles([b])
        assert result["rouge_l_mean"] == pytest.approx(0.5)
        assert result["bertscore_f1_mean"] == pytest.approx(0.8)

    def test_mean_computed_correctly(self):
        bundles = [
            MetricBundle(rouge_l=0.2, bertscore_f1=0.7),
            MetricBundle(rouge_l=0.4, bertscore_f1=0.9),
        ]
        result = aggregate_metric_bundles(bundles)
        assert result["rouge_l_mean"] == pytest.approx(0.3)
        assert result["bertscore_f1_mean"] == pytest.approx(0.8)

    def test_median_differs_from_mean_for_skewed_data(self):
        bundles = [
            MetricBundle(rouge_l=0.1, bertscore_f1=0.5),
            MetricBundle(rouge_l=0.1, bertscore_f1=0.5),
            MetricBundle(rouge_l=0.9, bertscore_f1=0.9),  # outlier
        ]
        result = aggregate_metric_bundles(bundles)
        # Mean is pulled up by outlier; median is not
        assert result["rouge_l_median"] < result["rouge_l_mean"]

    def test_output_keys(self):
        result = aggregate_metric_bundles([MetricBundle(0.5, 0.8)])
        assert set(result.keys()) == {
            "rouge_l_mean", "rouge_l_median",
            "bertscore_f1_mean", "bertscore_f1_median",
        }


# ─── TestFormatMetricsTable ───────────────────────────────────────────────────


class TestFormatMetricsTable:
    def _make_table(self) -> str:
        lat = LatencyPercentiles(p50=0.1, p95=1.0, p99=2.0, mean=0.3, min=0.05, max=2.0)
        return format_metrics_table(
            overall={"rouge_l_mean": 0.25, "rouge_l_median": 0.22,
                     "bertscore_f1_mean": 0.85, "bertscore_f1_median": 0.84},
            per_tier={
                "retrieval": {"rouge_l_mean": 0.30, "bertscore_f1_mean": 0.88},
                "generative": {"rouge_l_mean": 0.15, "bertscore_f1_mean": 0.78},
            },
            latency=lat,
            tier_counts={"cache": 0, "retrieval": 10, "generative": 5},
            n_total=15,
        )

    def test_returns_non_empty_string(self):
        assert len(self._make_table()) > 0

    def test_contains_overall_metrics(self):
        table = self._make_table()
        assert "0.250" in table or "0.25" in table

    def test_contains_tier_names(self):
        table = self._make_table()
        assert "retrieval" in table
        assert "generative" in table

    def test_contains_latency_section(self):
        table = self._make_table()
        assert "p50" in table
        assert "p95" in table
        assert "p99" in table

    def test_zero_count_tier_not_in_quality_table(self):
        table = self._make_table()
        # Cache has 0 entries — its quality row should be absent
        lines = table.split("\n")
        cache_quality_lines = [l for l in lines if "cache tier" in l.lower()]
        assert len(cache_quality_lines) == 0

"""Tests for the MT module — MTResult dataclass and MT metrics."""

import pytest

from goai_bench.tasks.mt import MTResult
from goai_bench.metrics.mt_metrics import (
    compute_bleu,
    compute_chrf,
    compute_ter,
    compute_bootstrap_ci,
)


class TestMTResult:
    """Tests for the MTResult dataclass."""

    def test_create_mt_result(self):
        result = MTResult(
            model_id="test/model",
            source_lang="fra_Latn",
            target_lang="mos_Latn",
            overall_chrf=25.0,
            overall_bleu=10.0,
            overall_ter=0.85,
            overall_comet=None,
            per_domain={"general": {"chrf": 25.0, "bleu": 10.0}},
            n_samples=100,
            n_samples_per_domain={"general": 100},
            hypotheses=["hyp1"],
            references=["ref1"],
            sources=["src1"],
            domains=["general"],
            timestamp="2026-01-01T00:00:00Z",
            duration_seconds=10.0,
        )
        assert result.model_id == "test/model"
        assert result.overall_chrf == 25.0
        assert result.n_samples == 100

    def test_mt_result_per_domain(self):
        result = MTResult(
            model_id="m", source_lang="fra_Latn", target_lang="mos_Latn",
            overall_chrf=0.0, overall_bleu=0.0, overall_ter=0.0,
            overall_comet=None,
            per_domain={
                "health": {"chrf": 15.0, "bleu": 5.0},
                "education": {"chrf": 20.0, "bleu": 8.0},
            },
            n_samples=50, n_samples_per_domain={"health": 25, "education": 25},
            hypotheses=[], references=[], sources=[], domains=[],
            timestamp="", duration_seconds=0.0,
        )
        assert "health" in result.per_domain
        assert result.per_domain["health"]["chrf"] == 15.0


class TestChrF:
    """Tests for chrF++ computation."""

    def test_chrf_identical(self):
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on the mat"]
        score = compute_chrf(hyps, refs)
        assert score is not None
        assert score > 90.0

    def test_chrf_different(self):
        refs = ["the cat sat on the mat"]
        hyps = ["le chat est assis"]
        score = compute_chrf(hyps, refs)
        assert score is not None
        assert score < 50.0

    def test_chrf_empty_returns_zero_or_value(self):
        score = compute_chrf(["hello"], ["world"])
        assert score is not None


class TestBLEU:
    """Tests for BLEU computation."""

    def test_bleu_perfect(self):
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on the mat"]
        score = compute_bleu(hyps, refs)
        assert score is not None
        assert score > 90.0

    def test_bleu_low(self):
        refs = ["the cat sat on the mat"]
        hyps = ["completely different sentence here"]
        score = compute_bleu(hyps, refs)
        assert score is not None
        assert score < 20.0


class TestTER:
    """Tests for TER computation."""

    def test_ter_identical(self):
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on the mat"]
        score = compute_ter(hyps, refs)
        assert score is not None
        assert score < 0.1

    def test_ter_different(self):
        refs = ["the cat sat on the mat"]
        hyps = ["completely different text"]
        score = compute_ter(hyps, refs)
        assert score is not None
        assert score > 0.5


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_ci_runs(self):
        refs = ["hello world", "good morning"] * 10
        hyps = ["hello world", "good morning"] * 10
        result = compute_bootstrap_ci(hyps, refs, compute_chrf, n_iterations=50)
        assert result is not None
        lower, upper = result
        assert lower <= upper

    def test_bootstrap_empty_input(self):
        result = compute_bootstrap_ci([], [], compute_chrf, n_iterations=10)
        assert result is None

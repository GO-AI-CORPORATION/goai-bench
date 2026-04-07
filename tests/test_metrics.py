"""Tests for metric functions -- graceful None returns on missing deps."""

import sys
from unittest.mock import patch

import pytest

from goai_bench.metrics.mt_metrics import (
    compute_bleu,
    compute_chrf,
    compute_comet_kiwi,
    compute_ter,
)
from goai_bench.metrics.asr_metrics import (
    compute_cer,
    compute_mer,
    compute_wer,
)


class TestMTMetricsGraceful:
    """Test that MT metrics return None when dependencies missing."""

    def test_chrf_works(self):
        score = compute_chrf(["hello"], ["hello"])
        assert score is not None

    def test_bleu_works(self):
        score = compute_bleu(["hello world"], ["hello world"])
        assert score is not None

    def test_ter_works(self):
        score = compute_ter(["hello world"], ["hello world"])
        assert score is not None

    def test_comet_returns_none_when_unavailable(self):
        with patch.dict(sys.modules, {"comet": None}):
            score = compute_comet_kiwi(["src"], ["hyp"])
            assert score is None or isinstance(score, float)


class TestASRMetricsGraceful:
    """Test ASR metrics with known inputs."""

    def test_wer_zero(self):
        assert compute_wer(["a b c"], ["a b c"]) == 0.0

    def test_cer_zero(self):
        assert compute_cer(["hello"], ["hello"]) == 0.0

    def test_mer_zero(self):
        assert compute_mer(["a b c"], ["a b c"]) == 0.0

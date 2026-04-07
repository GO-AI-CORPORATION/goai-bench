"""Tests for the ASR module — ASRResult dataclass and ASR metrics."""

import pytest

from goai_bench.tasks.asr import ASRResult
from goai_bench.metrics.asr_metrics import (
    compute_cer,
    compute_mer,
    compute_wer,
)


class TestASRResult:
    """Tests for ASRResult dataclass creation."""

    def test_create_asr_result(self):
        result = ASRResult(
            model_id="openai/whisper-large-v3",
            language="mos_Latn",
            overall_wer=0.45,
            overall_cer=0.28,
            overall_mer=0.43,
            per_domain={"general": {"wer": 0.45, "cer": 0.28}},
            n_samples=100,
            n_samples_per_domain={"general": 100},
            hypotheses=["hyp"],
            references=["ref"],
            domains=["general"],
            timestamp="2026-01-01T00:00:00Z",
            duration_seconds=30.0,
        )
        assert result.overall_wer == 0.45
        assert result.language == "mos_Latn"


class TestWER:
    """Tests for WER computation."""

    def test_wer_identical(self):
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on the mat"]
        wer = compute_wer(hyps, refs)
        assert wer is not None
        assert wer == 0.0

    def test_wer_completely_wrong(self):
        refs = ["the cat sat on the mat"]
        hyps = ["a b c d e f"]
        wer = compute_wer(hyps, refs)
        assert wer is not None
        assert wer > 0.5

    def test_wer_one_error(self):
        refs = ["the cat sat on the mat"]
        hyps = ["the dog sat on the mat"]
        wer = compute_wer(hyps, refs)
        assert wer is not None
        assert abs(wer - 1.0 / 6.0) < 0.01


class TestCER:
    """Tests for CER computation."""

    def test_cer_identical(self):
        refs = ["hello"]
        hyps = ["hello"]
        cer = compute_cer(hyps, refs)
        assert cer is not None
        assert cer == 0.0

    def test_cer_one_char_error(self):
        refs = ["hello"]
        hyps = ["hallo"]
        cer = compute_cer(hyps, refs)
        assert cer is not None
        assert cer > 0.0


class TestMER:
    """Tests for MER computation."""

    def test_mer_identical(self):
        refs = ["the cat sat"]
        hyps = ["the cat sat"]
        mer = compute_mer(hyps, refs)
        assert mer is not None
        assert mer == 0.0

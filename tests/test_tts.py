"""Tests for TTSResult dataclass."""

import pytest

from goai_bench.tasks.tts import TTSResult


class TestTTSResult:
    """Tests for TTSResult dataclass."""

    def test_create_tts_result(self):
        result = TTSResult(
            model_id="facebook/mms-tts",
            language="mos_Latn",
            overall_utmos=3.2,
            overall_loopback_wer=0.65,
            overall_loopback_cer=0.45,
            per_domain_utmos={"general": 3.2},
            per_domain_loopback_wer={"general": 0.65},
            per_domain_loopback_cer={"general": 0.45},
            n_samples=10,
            n_samples_per_domain={"general": 10},
            loopback_transcripts=["hello"],
            input_texts=["hello"],
            domains=["general"],
            timestamp="2026-01-01T00:00:00Z",
            duration_seconds=60.0,
        )
        assert result.overall_utmos == 3.2
        assert result.overall_loopback_wer == 0.65

    def test_tts_result_none_utmos(self):
        result = TTSResult(
            model_id="m", language="l",
            overall_utmos=None,
            overall_loopback_wer=1.0,
            overall_loopback_cer=1.0,
            per_domain_utmos={}, per_domain_loopback_wer={},
            per_domain_loopback_cer={},
            n_samples=0, n_samples_per_domain={},
            loopback_transcripts=[], input_texts=[],
            domains=[], timestamp="", duration_seconds=0.0,
        )
        assert result.overall_utmos is None

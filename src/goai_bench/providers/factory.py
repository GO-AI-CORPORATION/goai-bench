"""Provider factory — resolves a model_id string to a concrete provider."""

from __future__ import annotations

import logging

from goai_bench.providers.base import ASRProvider, MTProvider, TTSProvider

logger = logging.getLogger(__name__)

_MMS_TTS_PREFIX = "mms-tts:"


def _strip_mms_tts_prefix(model_id: str) -> str:
    if model_id.lower().startswith(_MMS_TTS_PREFIX):
        return model_id[len(_MMS_TTS_PREFIX) :].strip()
    return model_id


def _is_mms_tts_model(model_id: str) -> bool:
    mid = model_id.strip().lower()
    if mid.startswith(_MMS_TTS_PREFIX):
        return True
    return "mms-tts" in mid


def create_mt_provider(model_id: str, device: str = "auto") -> MTProvider:
    """Create an MT provider from a model identifier.

    Args:
        model_id: HuggingFace model ID or local path.
        device: Target device.

    Returns:
        An MTProvider instance.
    """
    from goai_bench.providers.mt.hf_seq2seq import HFSeq2SeqMTProvider

    return HFSeq2SeqMTProvider(model_id, device)


def create_asr_provider(model_id: str, device: str = "auto") -> ASRProvider:
    """Create an ASR provider from a model identifier.

    Routes to the appropriate provider based on model architecture:
    - ``wav2vec2`` / ``mms`` in ID -> :class:`Wav2Vec2ASRProvider`
    - ``whisper`` or other -> :class:`WhisperASRProvider`

    Args:
        model_id: HuggingFace model ID.
        device: Target device.

    Returns:
        An ASRProvider instance.
    """
    if "wav2vec2" in model_id.lower():
        from goai_bench.providers.asr.wav2vec2 import Wav2Vec2ASRProvider

        return Wav2Vec2ASRProvider(model_id, device)

    from goai_bench.providers.asr.whisper import WhisperASRProvider

    return WhisperASRProvider(model_id, device)


def create_tts_provider(model_id: str, device: str = "auto") -> TTSProvider:
    """Create a TTS provider from a model identifier.

    Routes to the appropriate provider:
    - ``mms-tts`` → :class:`MMSTTSProvider`
    - Other → :class:`HFTTSProvider`

    Args:
        model_id: HuggingFace model ID.
        device: Target device.

    Returns:
        A TTSProvider instance.
    """
    if _is_mms_tts_model(model_id):
        from goai_bench.providers.tts.mms_tts import MMSTTSProvider

        hf_id = _strip_mms_tts_prefix(model_id)
        return MMSTTSProvider(hf_id, device)

    from goai_bench.providers.tts.hf_tts import HFTTSProvider

    return HFTTSProvider(model_id, device)

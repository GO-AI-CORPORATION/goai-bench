"""HuggingFace TTS provider (generic ``text-to-speech`` pipeline)."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from goai_bench.core.exceptions import ModelLoadError
from goai_bench.providers.base import ProviderInfo, TTSProvider
from goai_bench.utils.audio_utils import resample_audio

logger = logging.getLogger(__name__)

_TARGET_SR = 16000


class HFTTSProvider(TTSProvider):
    """TTS provider using the HuggingFace ``text-to-speech`` pipeline.

    Models whose ID contains ``mms-tts`` are routed to
    :class:`MMSTTSProvider` by the factory instead.

    Args:
        model_id: HuggingFace model ID.
        device: Target device.
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        from goai_bench.core.device import resolve_device
        from goai_bench.core.model_cache import get_cached, put_cached

        self._model_id = model_id

        cache_key = f"tts:{model_id}"
        cached = get_cached(cache_key)
        if cached:
            self._pipe = cached
            return

        resolved = resolve_device(device)
        try:
            from transformers import pipeline as hf_pipeline

            self._pipe = hf_pipeline(
                "text-to-speech", model=model_id, device=resolved,
            )
        except Exception as exc:
            raise ModelLoadError(model_id, str(exc)) from exc

        put_cached(cache_key, self._pipe)

    @staticmethod
    def _waveform_from_output(speech: dict) -> np.ndarray:
        sr = int(speech.get("sampling_rate", _TARGET_SR))
        audio = speech["audio"]
        if isinstance(audio, list):
            audio = audio[0]
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sr != _TARGET_SR:
            arr = resample_audio(arr, sr, _TARGET_SR)
        return arr.astype(np.float32)

    def synthesize(self, text: str, language: Optional[str] = None) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Input text.
            language: Optional language hint.

        Returns:
            Float32 audio array at 16 kHz.
        """
        try:
            speech = self._pipe(text)
            return self._waveform_from_output(speech)
        except Exception as exc:
            logger.warning("TTS synthesis failed for '%s': %s", text[:50], exc)
            return np.zeros(_TARGET_SR, dtype=np.float32)

    def synthesize_batch(
        self,
        texts: List[str],
        language: Optional[str] = None,
    ) -> List[np.ndarray]:
        return [self.synthesize(t, language) for t in texts]

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="hf_tts",
            model_id=self._model_id,
        )

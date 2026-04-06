"""MMS / VITS TTS via HuggingFace ``text-to-speech`` pipeline."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from goai_bench.providers.base import ProviderInfo, TTSProvider
from goai_bench.utils.audio_utils import resample_audio

logger = logging.getLogger(__name__)

_TARGET_SR = 16000


class MMSTTSProvider(TTSProvider):
    """TTS MMS (Hub VITS / MMS-TTS) avec le pipeline HF ``text-to-speech``.

    Exemples d'IDs : ``facebook/mms-tts-mos``, ``goaicorp/mos-mms-tts-v2``.
    Préfixe optionnel ``mms-tts:`` pour forcer ce backend. Les modèles dont
    l'identifiant contient ``mms-tts`` sont routés ici depuis la factory.

    ``synthesize_batch`` permet à :class:`~goai_bench.tasks.tts.TTSEvaluator`
    d'appeler une synthèse par chunk (comme l'API Spark TTS), sans recharger le pipeline.
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        from goai_bench.core.device import resolve_device

        self._model_id = model_id
        self._device = resolve_device(device)
        self._pipe = None

    def _ensure_pipe(self) -> None:
        if self._pipe is not None:
            return

        from goai_bench.core.model_cache import get_cached, put_cached

        cache_key = f"mms_tts:{self._model_id}"
        cached = get_cached(cache_key)
        if cached:
            self._pipe = cached
            return

        from transformers import pipeline as hf_pipeline
        from transformers import set_seed

        set_seed(2024)
        logger.info(
            "Loading MMS TTS pipeline '%s' on %s", self._model_id, self._device,
        )
        self._pipe = hf_pipeline(
            "text-to-speech",
            model=self._model_id,
            device=self._device,
        )
        put_cached(cache_key, self._pipe)

    def _waveform_from_output(self, speech: dict) -> np.ndarray:
        sr = int(speech.get("sampling_rate", _TARGET_SR))
        audio = speech["audio"]
        if isinstance(audio, list):
            audio = audio[0]
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sr != _TARGET_SR:
            arr = resample_audio(arr, sr, _TARGET_SR)
        return arr.astype(np.float32)

    def _infer_one(self, text: str) -> np.ndarray:
        self._ensure_pipe()
        try:
            speech = self._pipe(text)
            return self._waveform_from_output(speech)
        except Exception as exc:
            logger.warning("MMS TTS failed for '%s': %s", text[:50], exc)
            return np.zeros(_TARGET_SR, dtype=np.float32)

    def synthesize(self, text: str, language: Optional[str] = None) -> np.ndarray:
        return self._infer_one(text)

    def synthesize_batch(
        self,
        texts: List[str],
        language: Optional[str] = None,
    ) -> List[np.ndarray]:
        return [self._infer_one(t) for t in texts]

    def info(self) -> ProviderInfo:
        return ProviderInfo(name="mms_tts", model_id=self._model_id)

"""Whisper ASR provider (HuggingFace ``automatic-speech-recognition`` pipeline)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from goai_bench.providers.base import ASRProvider, ProviderInfo

logger = logging.getLogger(__name__)


class WhisperASRProvider(ASRProvider):
    """ASR provider for OpenAI Whisper models via the HF pipeline.

    Args:
        model_id: HuggingFace model ID (e.g. ``openai/whisper-large-v3``).
        device: Target device.
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        from goai_bench.core.device import resolve_device
        from goai_bench.core.exceptions import ModelLoadError
        from goai_bench.core.model_cache import get_cached, put_cached

        self._model_id = model_id
        self._is_english_only = model_id.lower().endswith(".en")

        cache_key = f"asr_whisper:{model_id}"
        cached = get_cached(cache_key)
        if cached:
            self._pipeline = cached
            return

        resolved = resolve_device(device)
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=resolved if resolved != "cpu" else -1,
                torch_dtype=torch.float16 if resolved == "cuda" else torch.float32,
            )
        except Exception as exc:
            raise ModelLoadError(model_id, str(exc)) from exc

        put_cached(cache_key, self._pipeline)

    def _generate_kwargs(self, language: Optional[str]) -> Dict[str, Any]:
        if self._is_english_only:
            return {}
        from goai_bench.utils.text_utils import whisper_loopback_generate_kwargs

        return whisper_loopback_generate_kwargs(language)

    def _prepare_pipeline_input(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from goai_bench.utils.audio_utils import load_audio, resample_audio

        if "audio_array" in item:
            audio = item["audio_array"]
            sr = item.get("sampling_rate", 16000)
            if sr != 16000:
                audio = resample_audio(audio, sr, 16000)
            return {"raw": audio, "sampling_rate": 16000}
        if "audio_path" in item:
            try:
                audio = load_audio(item["audio_path"], target_sr=16000)
                return {"raw": audio, "sampling_rate": 16000}
            except Exception as exc:
                logger.warning("Failed to load audio: %s", exc)
                return None
        logger.warning("Sample has no audio_array or audio_path")
        return None

    def transcribe(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        from goai_bench.utils.audio_utils import resample_audio

        if sampling_rate != 16000:
            audio = resample_audio(audio, sampling_rate, 16000)

        generate_kwargs = self._generate_kwargs(language)

        try:
            result = self._pipeline(
                {"raw": audio, "sampling_rate": 16000},
                generate_kwargs=generate_kwargs,
            )
            return result.get("text", "")
        except Exception as exc:
            logger.warning("Whisper transcription failed: %s", exc)
            return ""

    def transcribe_batch(
        self,
        data: list,
        language: Optional[str] = None,
        batch_size: int = 8,
    ) -> List[str]:
        """Transcribe many samples using the HF pipeline with GPU batching."""
        from tqdm import tqdm

        gen_kw = self._generate_kwargs(language)
        results: List[str] = [""] * len(data)
        pending: List[tuple[int, Dict[str, Any]]] = []

        for i, item in enumerate(data):
            inp = self._prepare_pipeline_input(item)
            if inp is not None:
                pending.append((i, inp))

        if not pending:
            return results

        rng = range(0, len(pending), batch_size)
        for start in tqdm(rng, desc="Transcribing"):
            chunk = pending[start : start + batch_size]
            indices = [c[0] for c in chunk]
            inputs = [c[1] for c in chunk]
            bs = len(inputs)
            try:
                outs = self._pipeline(
                    inputs,
                    batch_size=bs,
                    generate_kwargs=gen_kw,
                )
                if not isinstance(outs, list):
                    outs = [outs]
                for idx, out in zip(indices, outs):
                    results[idx] = out.get("text", "") if isinstance(out, dict) else ""
            except Exception as exc:
                logger.warning("Batch transcription failed (%s), falling back per item", exc)
                for idx, inp in zip(indices, inputs):
                    try:
                        results[idx] = self.transcribe(
                            inp["raw"], 16000, language,
                        )
                    except Exception:
                        results[idx] = ""

        return results

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="whisper_asr",
            model_id=self._model_id,
        )

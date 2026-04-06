"""Wav2Vec2 / MMS-CTC ASR (direct ``Wav2Vec2ForCTC`` + ``AutoProcessor``)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from goai_bench.providers.base import ASRProvider, ProviderInfo

logger = logging.getLogger(__name__)


def _target_lang_from_hf_language(language: Optional[str]) -> Optional[str]:
    """``mos_Latn`` → ``mos`` for MMS ``target_lang``."""
    if not language:
        return None
    prefix = language.split("_")[0].strip().lower()
    return prefix or None


def _prepare_audio_item(item: Dict[str, Any]) -> Optional[np.ndarray]:
    from goai_bench.utils.audio_utils import load_audio, resample_audio

    if "audio_array" in item:
        audio = np.asarray(item["audio_array"], dtype=np.float32)
        sr = int(item.get("sampling_rate", 16000))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        if sr != 16000:
            audio = resample_audio(audio, sr, 16000)
        return audio.astype(np.float32)
    if "audio_path" in item:
        try:
            return load_audio(item["audio_path"], target_sr=16000)
        except Exception as exc:
            logger.warning("Failed to load audio: %s", exc)
            return None
    logger.warning("Sample has no audio_array or audio_path")
    return None


def _inputs_for_wav2vec(model, inputs, device: str):
    """Move features to ``device`` and match ``input_values`` dtype to model weights.

    The processor emits float32 waveforms; a half-precision model on CUDA
    otherwise raises *Input type (float) and bias type (c10::Half)*.
    """
    inputs = inputs.to(device)
    if "input_values" not in inputs:
        return inputs
    param_dtype = next(model.parameters()).dtype
    if inputs["input_values"].dtype != param_dtype:
        inputs["input_values"] = inputs["input_values"].to(dtype=param_dtype)
    return inputs


class Wav2Vec2ASRProvider(ASRProvider):
    """ASR with HuggingFace ``Wav2Vec2ForCTC`` (incl. MMS fine-tunes).

    Batching matches :class:`WhisperASRProvider`: chunk by ``batch_size``,
    pad with the processor, one forward pass per chunk.

    ``target_lang`` for MMS is derived from the evaluator language
    (e.g. ``mos_Latn`` → ``mos``).
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        self._model_id = model_id
        self._device = device

    def _load(self, language: Optional[str]):
        from goai_bench.core.device import resolve_device
        from goai_bench.core.model_cache import get_cached, put_cached
        from goai_bench.core.exceptions import ModelLoadError
        from goai_bench.utils.hf_utils import get_hf_token

        target = _target_lang_from_hf_language(language)
        lang_key = target or ""
        cache_key = f"asr_w2v:{self._model_id}:{lang_key}"
        cached = get_cached(cache_key)
        if cached:
            return cached

        resolved = resolve_device(self._device)
        logger.info(
            "Loading Wav2Vec2 ASR '%s' on %s (target_lang=%s)",
            self._model_id, resolved, target or "—",
        )
        try:
            from transformers import AutoProcessor, Wav2Vec2ForCTC
            import torch

            token = get_hf_token()
            tok_kw: Dict[str, Any] = {}
            if token:
                tok_kw["token"] = token

            processor = AutoProcessor.from_pretrained(self._model_id, **tok_kw)
            dtype = torch.float16 if resolved == "cuda" else torch.float32
            load_kw: Dict[str, Any] = {
                **tok_kw,
                "torch_dtype": dtype,
                "ignore_mismatched_sizes": True,
            }
            if target:
                load_kw["target_lang"] = target
            try:
                model = Wav2Vec2ForCTC.from_pretrained(self._model_id, **load_kw)
            except TypeError:
                load_kw.pop("target_lang", None)
                model = Wav2Vec2ForCTC.from_pretrained(self._model_id, **load_kw)
            model = model.to(resolved)
            model.eval()
        except Exception as exc:
            raise ModelLoadError(self._model_id, str(exc)) from exc

        triple = (model, processor, resolved)
        put_cached(cache_key, triple)
        return triple

    def transcribe(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        from goai_bench.utils.audio_utils import resample_audio

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        if sampling_rate != 16000:
            audio = resample_audio(audio, sampling_rate, 16000)

        model, processor, dev = self._load(language)
        import torch

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = _inputs_for_wav2vec(model, inputs, dev)
        try:
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]
            return processor.decode(pred_ids)
        except Exception as exc:
            logger.warning("Wav2Vec2 transcription failed: %s", exc)
            return ""

    def transcribe_batch(
        self,
        data: list,
        language: Optional[str] = None,
        batch_size: int = 8,
    ) -> List[str]:
        from tqdm import tqdm
        import torch

        results: List[str] = [""] * len(data)
        pending: List[tuple[int, np.ndarray]] = []

        for i, item in enumerate(data):
            arr = _prepare_audio_item(item)
            if arr is not None and len(arr) >= 160:
                pending.append((i, arr))

        if not pending:
            return results

        model, processor, dev = self._load(language)

        for start in tqdm(
            range(0, len(pending), batch_size),
            desc="Transcribing",
        ):
            chunk = pending[start : start + batch_size]
            indices = [c[0] for c in chunk]
            arrays = [c[1] for c in chunk]
            try:
                inputs = processor(
                    arrays,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = _inputs_for_wav2vec(model, inputs, dev)
                with torch.no_grad():
                    logits = model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=-1)
                texts = processor.batch_decode(pred_ids)
                for idx, text in zip(indices, texts):
                    results[idx] = text or ""
            except Exception as exc:
                logger.warning(
                    "Wav2Vec2 batch failed (%s), falling back per item", exc,
                )
                for idx, arr in zip(indices, arrays):
                    try:
                        results[idx] = self.transcribe(arr, 16000, language)
                    except Exception:
                        results[idx] = ""

        return results

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="wav2vec2_asr",
            model_id=self._model_id,
        )

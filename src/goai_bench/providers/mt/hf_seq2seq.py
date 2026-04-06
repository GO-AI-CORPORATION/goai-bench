"""HuggingFace seq2seq MT provider (NLLB, mBART, M2M100, opus-mt)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from goai_bench.core.exceptions import ModelLoadError
from goai_bench.providers.base import MTProvider, ProviderInfo

logger = logging.getLogger(__name__)

_NLLB_CODES = {
    "fra_Latn", "eng_Latn", "mos_Latn", "dyu_Latn", "fuv_Latn",
}


class HFSeq2SeqMTProvider(MTProvider):
    """MT provider for HuggingFace seq2seq models.

    Handles NLLB language tokens, forced BOS, and beam search
    automatically based on the model architecture.

    Args:
        model_id: HuggingFace model ID or local path.
        device: Target device (``cpu``, ``cuda``, ``mps``, ``auto``).
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        from goai_bench.core.device import resolve_device
        from goai_bench.core.model_cache import get_cached, put_cached

        self._model_id = model_id
        self._is_nllb = "nllb" in model_id.lower()

        cache_key = f"mt:{model_id}"
        cached = get_cached(cache_key)
        if cached:
            self._model, self._tokenizer = cached
            return

        resolved = resolve_device(device)
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            self._model = self._model.to(resolved)
            self._model.eval()
        except Exception as exc:
            raise ModelLoadError(model_id, str(exc)) from exc

        put_cached(cache_key, (self._model, self._tokenizer))

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text.

        Args:
            text: Source text.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            Translated string.
        """
        return self.translate_batch([text], source_lang, target_lang)[0]

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 16,
    ) -> List[str]:
        """Translate a batch of texts with beam search.

        Args:
            texts: Source texts.
            source_lang: Source language code.
            target_lang: Target language code.
            batch_size: Inference batch size.

        Returns:
            List of translated strings.
        """
        import torch

        if self._is_nllb:
            self._validate_nllb_lang(source_lang)
            self._validate_nllb_lang(target_lang)
            self._tokenizer.src_lang = source_lang

        all_translations: List[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self._model.device)

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": 256,
                "max_length": None,
                "num_beams": 4,
            }
            if self._is_nllb:
                tgt_id = self._tokenizer.convert_tokens_to_ids(target_lang)
                gen_kwargs["forced_bos_token_id"] = tgt_id

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            decoded = self._tokenizer.batch_decode(
                outputs, skip_special_tokens=True,
            )
            all_translations.extend(decoded)

        return all_translations

    def info(self) -> ProviderInfo:
        """Return provider metadata.

        Returns:
            ProviderInfo with model details.
        """
        return ProviderInfo(
            name="hf_seq2seq",
            supported_languages=sorted(_NLLB_CODES) if self._is_nllb else [],
            model_id=self._model_id,
        )

    @staticmethod
    def _validate_nllb_lang(lang: str) -> None:
        """Warn if lang is not in the known NLLB codes."""
        if lang not in _NLLB_CODES:
            logger.warning(
                "Language '%s' is not in the known NLLB codes %s. "
                "The tokenizer may still support it — proceeding anyway.",
                lang, sorted(_NLLB_CODES),
            )

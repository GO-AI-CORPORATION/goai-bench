"""Abstract base classes for all benchmark providers.

Any model or API can be benchmarked by implementing these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ProviderInfo:
    """Metadata about a provider's capabilities.

    Args:
        name: Human-readable provider name.
        supported_languages: List of supported HF language codes.
        model_id: Underlying model identifier.
    """

    name: str
    supported_languages: list[str] = field(default_factory=list)
    model_id: str = ""


class MTProvider(ABC):
    """Abstract interface for machine translation models or APIs."""

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text.

        Args:
            text: Source text.
            source_lang: Source language code (e.g. ``fra_Latn``).
            target_lang: Target language code (e.g. ``mos_Latn``).

        Returns:
            Translated text.
        """
        ...

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 16,
    ) -> List[str]:
        """Translate a batch of texts. Default: sequential calls.

        Args:
            texts: Source texts.
            source_lang: Source language code.
            target_lang: Target language code.
            batch_size: Batch size hint for GPU providers.

        Returns:
            List of translated strings.
        """
        return [self.translate(t, source_lang, target_lang) for t in texts]

    @abstractmethod
    def info(self) -> ProviderInfo:
        """Return provider metadata."""
        ...


class ASRProvider(ABC):
    """Abstract interface for speech recognition models or APIs."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe a single audio sample.

        Args:
            audio: Float32 mono audio array.
            sampling_rate: Audio sample rate.
            language: Optional language hint.

        Returns:
            Transcribed text.
        """
        ...

    def transcribe_batch(
        self,
        data: list,
        language: Optional[str] = None,
        batch_size: int = 8,
    ) -> List[str]:
        """Transcribe a batch of audio samples. Default: sequential calls.

        Args:
            data: List of dicts with ``audio_array``/``audio_path`` keys.
            language: Optional language hint.
            batch_size: Hint for GPU batching (HF provider); ignored here.

        Returns:
            List of transcription strings.
        """
        results = []
        for item in data:
            if "audio_array" in item:
                results.append(
                    self.transcribe(
                        item["audio_array"],
                        item.get("sampling_rate", 16000),
                        language,
                    )
                )
            elif "audio_path" in item:
                from goai_bench.utils.audio_utils import load_audio

                audio = load_audio(item["audio_path"], target_sr=16000)
                results.append(self.transcribe(audio, 16000, language))
            else:
                results.append("")
        return results

    @abstractmethod
    def info(self) -> ProviderInfo:
        """Return provider metadata."""
        ...


class TTSProvider(ABC):
    """Abstract interface for text-to-speech models or APIs."""

    @abstractmethod
    def synthesize(self, text: str, language: Optional[str] = None) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Input text.
            language: Optional language hint.

        Returns:
            Float32 audio array.
        """
        ...

    @abstractmethod
    def info(self) -> ProviderInfo:
        """Return provider metadata."""
        ...

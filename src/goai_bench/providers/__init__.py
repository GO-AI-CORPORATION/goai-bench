"""Provider interfaces and registry for model-agnostic benchmarking."""

from goai_bench.providers.base import (
    ASRProvider,
    MTProvider,
    ProviderInfo,
    TTSProvider,
)

__all__ = ["MTProvider", "ASRProvider", "TTSProvider", "ProviderInfo"]

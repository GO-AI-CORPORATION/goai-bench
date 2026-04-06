"""Task-agnostic evaluation dispatcher.

Provides a single ``run_evaluation()`` entry point reusable by
``run_benchmark.py``, ``run_baselines.py``, and the Python API.
"""

from __future__ import annotations

from typing import Any, Dict, List


def run_evaluation(
    task: str,
    provider: Any,
    data: List[Dict[str, Any]],
    language: str,
    source_lang: str = "fra_Latn",
    target_lang: str = "mos_Latn",
    batch_size: int = 8,
    compute_comet: bool = False,
    loopback_asr_model: str = "openai/whisper-large-v3",
) -> Any:
    """Run evaluation for a given task and return the result dataclass.

    Args:
        task: One of ``mt``, ``asr``, ``tts``.
        provider: A concrete provider instance (MTProvider / ASRProvider / TTSProvider).
        data: List of sample dicts as produced by :class:`DataLoader`.
        language: HF language code (e.g. ``mos_Latn``).
        source_lang: Source language code (MT only).
        target_lang: Target language code (MT only).
        batch_size: Batch size for inference.
        compute_comet: Whether to compute COMET (MT only).
        loopback_asr_model: ASR model for TTS loopback evaluation.

    Returns:
        ``MTResult``, ``ASRResult``, or ``TTSResult``.
    """
    if task == "mt":
        from goai_bench.tasks.mt import MTEvaluator

        evaluator = MTEvaluator(provider, source_lang, target_lang)
        return evaluator.evaluate(data, batch_size, compute_comet=compute_comet)

    if task == "asr":
        from goai_bench.tasks.asr import ASREvaluator

        evaluator = ASREvaluator(provider, language, batch_size=batch_size)
        return evaluator.evaluate(data)

    if task == "tts":
        from goai_bench.tasks.tts import TTSEvaluator

        evaluator = TTSEvaluator(
            provider,
            language,
            loopback_asr_model,
            synthesis_batch_size=max(batch_size, 1),
        )
        return evaluator.evaluate(data)

    raise ValueError(f"Unknown task: {task!r}")

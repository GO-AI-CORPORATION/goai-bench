"""Automatic Speech Recognition evaluator."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from goai_bench.providers.base import ASRProvider

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """Result container for ASR evaluation.

    Args:
        model_id: Model identifier string.
        language: HF language code.
        overall_wer: Corpus-level Word Error Rate.
        overall_cer: Corpus-level Character Error Rate.
        overall_mer: Corpus-level Match Error Rate.
        per_domain: Per-domain metric breakdown.
        n_samples: Total samples evaluated.
        n_samples_per_domain: Sample count per domain.
        hypotheses: All model transcriptions (normalized).
        references: All reference texts (normalized).
        domains: Domain label for each sample.
        timestamp: ISO timestamp.
        duration_seconds: Wall-clock time in seconds.
        all_metrics: Flat dict of all computed metrics.
    """

    model_id: str
    language: str
    overall_wer: float
    overall_cer: float
    overall_mer: float
    per_domain: Dict[str, Dict[str, float]]
    n_samples: int
    n_samples_per_domain: Dict[str, int]
    hypotheses: List[str]
    references: List[str]
    domains: List[str]
    timestamp: str
    duration_seconds: float
    all_metrics: Dict[str, Any] = field(default_factory=dict)


class ASREvaluator:
    """Evaluate any ASR provider on a dataset.

    The evaluator is model-agnostic: it receives an ``ASRProvider``
    instance and delegates all transcription to it.

    Args:
        provider: An object implementing ``ASRProvider``.
        language: HF language code.
        batch_size: Pipeline batch size for HF ASR (GPU throughput).
    """

    def __init__(
        self, provider: ASRProvider, language: str, batch_size: int = 8,
    ) -> None:
        self.provider = provider
        self.language = language
        self.batch_size = batch_size
        self.model_id = provider.info().model_id or provider.info().name

    def transcribe_batch(
        self, data: List[Dict[str, Any]],
    ) -> List[str]:
        """Transcribe all samples via the provider (batched when supported)."""
        return self.provider.transcribe_batch(
            data, self.language, batch_size=self.batch_size,
        )

    def evaluate(self, data: List[Dict[str, Any]]) -> ASRResult:
        """Run full ASR evaluation.

        Args:
            data: List of dicts from ``DataLoader.load_asr_data()``
                or ``DataLoader.load_from_config()``.

        Returns:
            ASRResult dataclass.
        """
        from goai_bench.metrics.asr_metrics import (
            compute_cer, compute_mer, compute_wer,
        )
        from goai_bench.utils.text_utils import normalize_text

        np.random.seed(42)
        start = time.time()

        raw_refs = [d["transcript"] for d in data]
        domains = [d.get("domain", "general") for d in data]

        logger.info(
            "Evaluating ASR: %s, %s, %d samples",
            self.model_id, self.language, len(data),
        )

        raw_hyps = self.transcribe_batch(data)

        references = [normalize_text(r) for r in raw_refs]
        hypotheses = [normalize_text(h) for h in raw_hyps]

        overall_wer = compute_wer(hypotheses, references) or 0.0
        overall_cer = compute_cer(hypotheses, references) or 0.0
        overall_mer = compute_mer(hypotheses, references) or 0.0

        per_domain: Dict[str, Dict[str, float]] = {}
        n_per_domain: Dict[str, int] = defaultdict(int)
        domain_groups: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {"hyps": [], "refs": []},
        )
        for h, r, d in zip(hypotheses, references, domains):
            domain_groups[d]["hyps"].append(h)
            domain_groups[d]["refs"].append(r)
            n_per_domain[d] += 1

        for dom, grp in domain_groups.items():
            per_domain[dom] = {
                "wer": compute_wer(grp["hyps"], grp["refs"]) or 0.0,
                "cer": compute_cer(grp["hyps"], grp["refs"]) or 0.0,
            }

        duration = time.time() - start

        return ASRResult(
            model_id=self.model_id,
            language=self.language,
            overall_wer=overall_wer,
            overall_cer=overall_cer,
            overall_mer=overall_mer,
            per_domain=dict(per_domain),
            n_samples=len(data),
            n_samples_per_domain=dict(n_per_domain),
            hypotheses=hypotheses,
            references=references,
            domains=domains,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            all_metrics={
                "wer": overall_wer, "cer": overall_cer, "mer": overall_mer,
            },
        )

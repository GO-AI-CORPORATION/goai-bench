"""Machine Translation evaluator."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from goai_bench.providers.base import MTProvider

logger = logging.getLogger(__name__)


@dataclass
class MTResult:
    """Result container for MT evaluation.

    Args:
        model_id: Model identifier string.
        source_lang: Source language code.
        target_lang: Target language code.
        overall_chrf: Corpus-level chrF++ score.
        overall_bleu: Corpus-level BLEU score.
        overall_ter: Corpus-level TER score.
        overall_comet: CometKiwi score or None.
        per_domain: Per-domain metric breakdown.
        n_samples: Total number of samples evaluated.
        n_samples_per_domain: Sample count per domain.
        hypotheses: All model outputs.
        references: All reference texts.
        sources: All source texts.
        domains: Domain label for each sample.
        timestamp: ISO timestamp of the evaluation.
        duration_seconds: Wall-clock time in seconds.
        all_metrics: Flat dict of all computed metrics.
    """

    model_id: str
    source_lang: str
    target_lang: str
    overall_chrf: float
    overall_bleu: float
    overall_ter: float
    overall_comet: Optional[float]
    per_domain: Dict[str, Dict[str, float]]
    n_samples: int
    n_samples_per_domain: Dict[str, int]
    hypotheses: List[str]
    references: List[str]
    sources: List[str]
    domains: List[str]
    timestamp: str
    duration_seconds: float
    all_metrics: Dict[str, Any] = field(default_factory=dict)


class MTEvaluator:
    """Evaluate any MT provider on a dataset.

    The evaluator is model-agnostic: it receives an ``MTProvider``
    instance and delegates all translation to it.

    Args:
        provider: An object implementing ``MTProvider``.
        source_lang: Source language code.
        target_lang: Target language code.
    """

    def __init__(
        self,
        provider: MTProvider,
        source_lang: str,
        target_lang: str,
    ) -> None:
        self.provider = provider
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model_id = provider.info().model_id or provider.info().name

    def translate_batch(
        self, texts: List[str], batch_size: int = 16,
    ) -> List[str]:
        """Translate a batch of texts via the provider.

        Args:
            texts: Source texts.
            batch_size: Batch size hint.

        Returns:
            List of translated strings.
        """
        from tqdm import tqdm

        all_translations: List[str] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i: i + batch_size]
            translated = self.provider.translate_batch(
                batch, self.source_lang, self.target_lang, batch_size,
            )
            all_translations.extend(translated)
        return all_translations

    def evaluate(
        self,
        data: List[Dict[str, Any]],
        batch_size: int = 16,
        compute_comet: bool = True,
    ) -> MTResult:
        """Run full MT evaluation.

        Args:
            data: List of dicts from ``DataLoader.load_mt_data()``.
            batch_size: Inference batch size.
            compute_comet: Whether to compute CometKiwi.

        Returns:
            MTResult dataclass with all metrics.
        """
        import numpy as np
        import torch
        from goai_bench.metrics.mt_metrics import (
            compute_bleu, compute_chrf, compute_comet_kiwi, compute_ter,
        )

        torch.manual_seed(42)
        np.random.seed(42)

        start = time.time()
        sources = [d["source"] for d in data]
        references = [d["reference"] for d in data]
        domains = [d.get("domain", "general") for d in data]

        logger.info(
            "Evaluating MT: %s, %s->%s, %d samples",
            self.model_id, self.source_lang, self.target_lang, len(sources),
        )

        hypotheses = self.translate_batch(sources, batch_size)

        overall_chrf = compute_chrf(hypotheses, references) or 0.0
        overall_bleu = compute_bleu(hypotheses, references) or 0.0
        overall_ter = compute_ter(hypotheses, references) or 0.0
        overall_comet = None
        if compute_comet:
            overall_comet = compute_comet_kiwi(sources, hypotheses)

        per_domain: Dict[str, Dict[str, float]] = {}
        n_per_domain: Dict[str, int] = defaultdict(int)

        domain_groups: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {"hyps": [], "refs": [], "srcs": []},
        )
        for h, r, s, d in zip(hypotheses, references, sources, domains):
            domain_groups[d]["hyps"].append(h)
            domain_groups[d]["refs"].append(r)
            domain_groups[d]["srcs"].append(s)
            n_per_domain[d] += 1

        for dom, group in domain_groups.items():
            per_domain[dom] = {
                "chrf": compute_chrf(group["hyps"], group["refs"]) or 0.0,
                "bleu": compute_bleu(group["hyps"], group["refs"]) or 0.0,
            }

        duration = time.time() - start

        return MTResult(
            model_id=self.model_id,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            overall_chrf=overall_chrf,
            overall_bleu=overall_bleu,
            overall_ter=overall_ter,
            overall_comet=overall_comet,
            per_domain=dict(per_domain),
            n_samples=len(sources),
            n_samples_per_domain=dict(n_per_domain),
            hypotheses=hypotheses,
            references=references,
            sources=sources,
            domains=domains,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            all_metrics={
                "chrf": overall_chrf,
                "bleu": overall_bleu,
                "ter": overall_ter,
                "comet_kiwi": overall_comet,
            },
        )

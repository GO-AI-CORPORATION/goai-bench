"""Text-to-Speech evaluator -- naturalness (UTMOS) + intelligibility (loopback ASR)."""
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from goai_bench.providers.base import TTSProvider

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result container for TTS evaluation."""

    model_id: str
    language: str
    overall_utmos: Optional[float]
    overall_loopback_wer: float
    overall_loopback_cer: float
    per_domain_utmos: Dict[str, Optional[float]]
    per_domain_loopback_wer: Dict[str, float]
    per_domain_loopback_cer: Dict[str, float]
    n_samples: int
    n_samples_per_domain: Dict[str, int]
    loopback_transcripts: List[str]
    input_texts: List[str]
    domains: List[str]
    timestamp: str
    duration_seconds: float
    all_metrics: Dict[str, Any] = field(default_factory=dict)


class TTSEvaluator:
    """Evaluate any TTS provider via naturalness and intelligibility.

    Automatically uses ``provider.synthesize_batch()`` when the provider
    exposes it (e.g. ``MMSTTSProvider``), falling back to the
    sequential loop otherwise.

    Args:
        provider:            An object implementing ``TTSProvider``.
        language:            HF language code.
        loopback_asr_model:  ASR model ID for intelligibility evaluation.
        synthesis_batch_size: Number of texts per batch call when the
                              provider does NOT implement
                              ``synthesize_batch()`` itself.
    """

    def __init__(
        self,
        provider: TTSProvider,
        language: str,
        loopback_asr_model: str = "openai/whisper-large-v3",
        synthesis_batch_size: int = 32,
    ) -> None:
        self.provider = provider
        self.language = language
        self.loopback_asr_model = loopback_asr_model
        self.model_id = provider.info().model_id or provider.info().name
        self._synthesis_batch_size = synthesis_batch_size

        self._provider_has_batch = callable(getattr(provider, "synthesize_batch", None))
        logger.info(
            "TTSEvaluator ready: provider_has_batch=%s  synthesis_batch_size=%d",
            self._provider_has_batch,
            synthesis_batch_size,
        )

    # ------------------------------------------------------------------
    # Synthesis dispatch
    # ------------------------------------------------------------------

    def synthesize_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Synthesize audio for a list of texts.

        Prefers ``provider.synthesize_batch()`` (single batched HTTP call
        per chunk, one ``model.generate()`` on the server) over the old
        one-by-one loop.  Falls back gracefully if the provider doesn't
        implement it.
        """
        if self._provider_has_batch:
            logger.info(
                "Using provider.synthesize_batch() for %d texts", len(texts)
            )
            from tqdm import tqdm

            # Batch synthesis with progress tracking.
            results: List[np.ndarray] = []
            batch_size = self._synthesis_batch_size

            with tqdm(total=len(texts), desc="Synthesizing (batch)") as pbar:
                for start in range(0, len(texts), batch_size):
                    chunk = texts[start : start + batch_size]
                    chunk_results = self.provider.synthesize_batch(
                        chunk, language=self.language
                    )
                    results.extend(
                        np.asarray(a, dtype=np.float32) for a in chunk_results
                    )
                    pbar.update(len(chunk))

            return results

        # ------------------------------------------------------------------
        # Legacy fallback: one item at a time (original behaviour)
        # ------------------------------------------------------------------
        logger.info(
            "Provider has no synthesize_batch(); falling back to sequential synthesis"
        )
        from tqdm import tqdm

        audio_samples: List[np.ndarray] = []
        for text in tqdm(texts, desc="Synthesizing"):
            audio = self.provider.synthesize(text, self.language)
            audio_samples.append(np.asarray(audio, dtype=np.float32))
        return audio_samples

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, data: List[Dict[str, Any]]) -> TTSResult:
        """Run full TTS evaluation.

        Args:
            data: From ``DataLoader.load_tts_data()``.

        Returns:
            TTSResult dataclass.
        """
        from goai_bench.metrics.asr_metrics import compute_cer, compute_wer
        from goai_bench.metrics.tts_metrics import compute_loopback_wer, compute_utmos
        from goai_bench.utils.text_utils import normalize_text

        np.random.seed(42)
        start = time.time()

        texts = [d["text"] for d in data]
        domains = [d.get("domain", "general") for d in data]

        logger.info(
            "Evaluating TTS: %s, %s, %d samples",
            self.model_id, self.language, len(texts),
        )

        audio_samples = self.synthesize_batch(texts)

        utmos_scores = compute_utmos(audio_samples)
        overall_utmos = float(np.mean(utmos_scores)) if utmos_scores else None

        lb_result = compute_loopback_wer(
            audio_samples,
            texts,
            loopback_model_id=self.loopback_asr_model,
            language=self.language,
        )
        overall_lb_wer = lb_result["wer"] if lb_result else 1.0
        overall_lb_cer = lb_result["cer"] if lb_result else 1.0
        lb_transcripts = lb_result["transcripts"] if lb_result else [""] * len(texts)

        per_domain_utmos: Dict[str, Optional[float]] = {}
        per_domain_lb_wer: Dict[str, float] = {}
        per_domain_lb_cer: Dict[str, float] = {}
        n_per_domain: Dict[str, int] = defaultdict(int)
        domain_groups: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {"utmos": [], "texts": [], "transcripts": []}
        )

        for i, dom in enumerate(domains):
            n_per_domain[dom] += 1
            if utmos_scores:
                domain_groups[dom]["utmos"].append(utmos_scores[i])
            domain_groups[dom]["texts"].append(texts[i])
            domain_groups[dom]["transcripts"].append(lb_transcripts[i])

        for dom, grp in domain_groups.items():
            per_domain_utmos[dom] = (
                float(np.mean(grp["utmos"])) if grp["utmos"] else None
            )
            norm_texts = [normalize_text(t) for t in grp["texts"]]
            norm_trans = [normalize_text(t) for t in grp["transcripts"]]
            per_domain_lb_wer[dom] = compute_wer(norm_trans, norm_texts) or 1.0
            per_domain_lb_cer[dom] = compute_cer(norm_trans, norm_texts) or 1.0

        duration = time.time() - start

        return TTSResult(
            model_id=self.model_id,
            language=self.language,
            overall_utmos=overall_utmos,
            overall_loopback_wer=overall_lb_wer,
            overall_loopback_cer=overall_lb_cer,
            per_domain_utmos=per_domain_utmos,
            per_domain_loopback_wer=per_domain_lb_wer,
            per_domain_loopback_cer=per_domain_lb_cer,
            n_samples=len(data),
            n_samples_per_domain=dict(n_per_domain),
            loopback_transcripts=lb_transcripts,
            input_texts=texts,
            domains=domains,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            all_metrics={
                "utmos": overall_utmos,
                "loopback_wer": overall_lb_wer,
                "loopback_cer": overall_lb_cer,
            },
        )
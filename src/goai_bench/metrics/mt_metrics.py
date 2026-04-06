"""Machine translation metrics: chrF++, BLEU, TER, CometKiwi, bootstrap CI."""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _sacrebleu_ref_lists(references: List[str]) -> List[List[str]]:
    """Format references for sacrebleu corpus_* (one variant list per sentence)."""
    return [[r] for r in references]


def compute_chrf(
    hypotheses: List[str],
    references: List[str],
    word_order: int = 2,
) -> Optional[float]:
    """Compute chrF++ score using sacrebleu.

    Args:
        hypotheses: Model outputs.
        references: Reference translations.
        word_order: 2 for chrF++ (default), 0 for chrF.

    Returns:
        chrF++ score (0–100 scale), or ``None`` if sacrebleu is missing.
    """
    try:
        import sacrebleu
        score = sacrebleu.corpus_chrf(
            hypotheses, _sacrebleu_ref_lists(references), word_order=word_order,
        )
        return float(score.score)
    except ImportError:
        logger.warning("sacrebleu not installed — skipping chrF++ computation.")
        return None
    except Exception as exc:
        logger.warning("chrF++ computation failed: %s", exc)
        return None


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
) -> Optional[float]:
    """Compute corpus-level BLEU with standard 13a tokenization.

    Returns:
        BLEU score (0–100 scale), or ``None`` on failure.
    """
    try:
        import sacrebleu
        score = sacrebleu.corpus_bleu(
            hypotheses, _sacrebleu_ref_lists(references),
        )
        return float(score.score)
    except ImportError:
        logger.warning("sacrebleu not installed — skipping BLEU computation.")
        return None
    except Exception as exc:
        logger.warning("BLEU computation failed: %s", exc)
        return None


def compute_ter(
    hypotheses: List[str],
    references: List[str],
) -> Optional[float]:
    """Compute Translation Edit Rate using sacrebleu.

    Returns:
        TER score (0–1+ scale), or ``None`` on failure.
    """
    try:
        import sacrebleu
        score = sacrebleu.corpus_ter(
            hypotheses, _sacrebleu_ref_lists(references),
        )
        return float(score.score) / 100.0
    except ImportError:
        logger.warning("sacrebleu not installed — skipping TER computation.")
        return None
    except Exception as exc:
        logger.warning("TER computation failed: %s", exc)
        return None


def compute_comet_kiwi(
    sources: List[str],
    hypotheses: List[str],
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
) -> Optional[float]:
    """Compute reference-free CometKiwi score.

    Args:
        sources: Source texts.
        hypotheses: Model outputs.
        model_name: COMET model identifier.

    Returns:
        Mean COMET score, or ``None`` if comet is unavailable.
    """
    try:
        from comet import download_model, load_from_checkpoint
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        data = [
            {"src": s, "mt": h}
            for s, h in zip(sources, hypotheses)
        ]
        output = model.predict(data, batch_size=16, gpus=0)
        scores = output.scores if hasattr(output, "scores") else output[0]
        return float(np.mean(scores))
    except ImportError:
        logger.warning(
            "unbabel-comet not installed — skipping CometKiwi. "
            "Install with: pip install unbabel-comet"
        )
        return None
    except Exception as exc:
        logger.warning("CometKiwi computation failed: %s", exc)
        return None


def compute_bootstrap_ci(
    hypotheses: List[str],
    references: List[str],
    metric_fn: Callable[[List[str], List[str]], Optional[float]],
    n_iterations: int = 1000,
    confidence: float = 0.95,
) -> Optional[Tuple[float, float]]:
    """Bootstrap confidence interval for any metric function.

    Args:
        hypotheses: Model outputs.
        references: Reference texts.
        metric_fn: A function ``(hyps, refs) -> float``.
        n_iterations: Number of bootstrap iterations.
        confidence: Confidence level (e.g. 0.95).

    Returns:
        Tuple ``(lower_bound, upper_bound)``, or ``None`` on failure.
    """
    n = len(hypotheses)
    if n == 0:
        return None
    rng = np.random.RandomState(42)
    scores: List[float] = []
    for _ in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        hyps_sample = [hypotheses[i] for i in indices]
        refs_sample = [references[i] for i in indices]
        s = metric_fn(hyps_sample, refs_sample)
        if s is not None:
            scores.append(s)

    if not scores:
        return None

    alpha = 1.0 - confidence
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lower, upper

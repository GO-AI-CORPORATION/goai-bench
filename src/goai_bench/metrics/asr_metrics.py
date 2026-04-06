"""ASR metrics: WER, CER, MER."""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def compute_wer(
    hypotheses: List[str], references: List[str],
) -> Optional[float]:
    """Compute Word Error Rate.

    Returns:
        WER as a float in [0, 1+], or ``None`` on failure.
    """
    try:
        import jiwer
        return float(jiwer.wer(references, hypotheses))
    except ImportError:
        logger.warning("jiwer not installed — skipping WER computation.")
        return None
    except Exception as exc:
        logger.warning("WER computation failed: %s", exc)
        return None


def compute_cer(
    hypotheses: List[str], references: List[str],
) -> Optional[float]:
    """Compute Character Error Rate.

    Returns:
        CER as a float in [0, 1+], or ``None`` on failure.
    """
    try:
        import jiwer
        return float(jiwer.cer(references, hypotheses))
    except ImportError:
        logger.warning("jiwer not installed — skipping CER computation.")
        return None
    except Exception as exc:
        logger.warning("CER computation failed: %s", exc)
        return None


def compute_mer(
    hypotheses: List[str], references: List[str],
) -> Optional[float]:
    """Compute Match Error Rate.

    Returns:
        MER as a float in [0, 1+], or ``None`` on failure.
    """
    try:
        import jiwer
        return float(jiwer.mer(references, hypotheses))
    except ImportError:
        logger.warning("jiwer not installed — skipping MER computation.")
        return None
    except Exception as exc:
        logger.warning("MER computation failed: %s", exc)
        return None

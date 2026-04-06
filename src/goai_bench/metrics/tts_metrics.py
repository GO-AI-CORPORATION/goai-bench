"""TTS metrics: UTMOS naturalness, loopback WER, MCD."""

import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# One UTMOS model per process — avoids reloading EfficientNet + checkpoint every TTS group.
_utmos_model: Any = None


def compute_utmos(
    audio_samples: List[np.ndarray], sr: int = 16000,
) -> Optional[List[float]]:
    """Score audio naturalness with UTMOSv2.

    Args:
        audio_samples: List of float32 audio arrays.
        sr: Sample rate.

    Returns:
        List of UTMOS scores (scale ~1–5), or ``None`` if unavailable.
    """
    try:
        # UTMOSv2 requires separate install; API differs across versions.
        import utmosv2  # noqa: F401
        logger.info("UTMOSv2 available — scoring %d samples", len(audio_samples))

        scores: List[float] = []
        if hasattr(utmosv2, "create_predictor"):
            predictor = utmosv2.create_predictor()
            for audio in audio_samples:
                if audio.size == 0:
                    scores.append(0.0)
                    continue
                scores.append(float(predictor.predict(audio, sr)))
        else:
            global _utmos_model
            if _utmos_model is None:
                _utmos_model = utmosv2.create_model(pretrained=True)
            model = _utmos_model
            for audio in audio_samples:
                if audio.size == 0:
                    scores.append(0.0)
                    continue
                raw = model.predict(
                    data=np.asarray(audio, dtype=np.float32),
                    sr=sr,
                    verbose=False,
                    batch_size=1,
                    num_workers=0,
                )
                scores.append(float(np.asarray(raw).reshape(-1)[0]))
        return scores
    except ImportError:
        logger.warning(
            "UTMOSv2 not installed — skipping naturalness scoring. "
            "Install with: pip install git+https://github.com/"
            "sarulab-speech/UTMOSv2.git"
        )
        return None
    except Exception as exc:
        logger.warning("UTMOS computation failed: %s", exc)
        return None


def compute_loopback_wer(
    audio_samples: List[np.ndarray],
    reference_texts: List[str],
    loopback_model_id: str = "openai/whisper-large-v3",
    language: Optional[str] = None,
    sr: int = 16000,
) -> Optional[dict]:
    """Evaluate intelligibility via ASR loopback.

    Transcribes synthesized audio with an ASR model and computes
    WER/CER against the original input text.

    Args:
        audio_samples: Synthesized audio arrays.
        reference_texts: Original TTS input texts.
        loopback_model_id: ASR model for loopback transcription.
        language: Language hint for ASR.
        sr: Sample rate.

    Returns:
        Dict with ``wer``, ``cer``, ``transcripts``, or ``None``.
    """
    try:
        from transformers import pipeline as hf_pipeline
        import jiwer
    except ImportError:
        logger.warning(
            "transformers or jiwer not installed — skipping loopback WER."
        )
        return None

    from goai_bench.core.model_cache import get_cached, put_cached
    from goai_bench.utils.text_utils import whisper_loopback_generate_kwargs

    cache_key = f"loopback_asr:{loopback_model_id}"
    pipe = get_cached(cache_key)
    if pipe is None:
        try:
            pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=loopback_model_id,
            )
            put_cached(cache_key, pipe)
        except Exception as exc:
            logger.warning("Could not load loopback ASR model: %s", exc)
            return None

    gen_kw = whisper_loopback_generate_kwargs(language)

    transcripts: List[str] = []
    for audio in audio_samples:
        try:
            if audio.size == 0:
                transcripts.append("")
                continue
            result = pipe(
                {"raw": audio, "sampling_rate": sr},
                generate_kwargs=gen_kw,
            )
            transcripts.append(result.get("text", ""))
        except Exception as exc:
            logger.debug("Loopback ASR failed for one sample: %s", exc)
            transcripts.append("")

    # Normalize
    from goai_bench.utils.text_utils import normalize_text
    norm_refs = [normalize_text(t) for t in reference_texts]
    norm_hyps = [normalize_text(t) for t in transcripts]

    wer = float(jiwer.wer(norm_refs, norm_hyps))
    cer = float(jiwer.cer(norm_refs, norm_hyps))

    return {"wer": wer, "cer": cer, "transcripts": transcripts}


def compute_mcd(
    synthesized: List[np.ndarray],
    reference: List[np.ndarray],
    sr: int = 16000,
) -> Optional[float]:
    """Compute Mel Cepstral Distortion between audio pairs.

    Returns:
        Mean MCD score, or ``None`` if pymcd is unavailable.
    """
    try:
        from pymcd.mcd import Calculate
        calc = Calculate()
        scores = []
        for synth, ref in zip(synthesized, reference):
            mcd = calc.mcd(synth, ref, sr)
            scores.append(float(mcd))
        return float(np.mean(scores)) if scores else None
    except ImportError:
        logger.warning("pymcd not installed — skipping MCD computation.")
        return None
    except Exception as exc:
        logger.warning("MCD computation failed: %s", exc)
        return None

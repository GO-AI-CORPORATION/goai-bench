"""Audio loading, resampling, and format conversion utilities."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_audio(
    path: str,
    target_sr: int = 16000,
) -> np.ndarray:
    """Load an audio file and convert to mono 16 kHz PCM float32.

    Args:
        path: Path to the audio file.
        target_sr: Target sample rate (default 16000).

    Returns:
        1-D numpy array of float32 samples.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If the audio cannot be loaded.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        import librosa
        audio, sr = librosa.load(str(p), sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        pass

    try:
        import soundfile as sf
        audio, sr = sf.read(str(p), dtype="float32")
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Resample if needed
        if sr != target_sr:
            audio = _resample_simple(audio, sr, target_sr)
        return audio.astype(np.float32)
    except ImportError:
        pass

    raise RuntimeError(
        f"Cannot load audio: neither librosa nor soundfile is installed. "
        f"Install one with: pip install librosa or pip install soundfile"
    )


def resample_audio(
    audio: np.ndarray, orig_sr: int, target_sr: int,
) -> np.ndarray:
    """Resample audio to a target sample rate.

    Args:
        audio: Input audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio as float32 numpy array.
    """
    return _resample_simple(audio, orig_sr, target_sr)


def _resample_simple(
    audio: np.ndarray, orig_sr: int, target_sr: int,
) -> np.ndarray:
    """Simple linear resampling (fallback when librosa unavailable)."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_out = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def get_audio_duration(path: str) -> Optional[float]:
    """Get audio duration in seconds.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds, or ``None`` if unavailable.
    """
    try:
        import librosa
        return float(librosa.get_duration(path=path))
    except Exception:
        pass
    try:
        import soundfile as sf
        info = sf.info(path)
        return float(info.duration)
    except Exception:
        return None

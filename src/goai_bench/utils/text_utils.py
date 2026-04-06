"""Text normalization, tokenization, and Whisper language helpers."""

import re
import unicodedata
from typing import Any, Dict, Optional

# Whisper language-token index used for ``forced_decoder_ids``.
WHISPER_LANG_TO_ID: dict[str, int] = {
    "en": 0, "zh": 1, "de": 2, "es": 3, "ru": 4, "ko": 5, "fr": 6, "ja": 7,
    "pt": 8, "tr": 9, "pl": 10, "ca": 11, "nl": 12, "ar": 13, "sv": 14,
    "it": 15, "id": 16, "hi": 17, "fi": 18, "vi": 19, "he": 20, "uk": 21,
    "el": 22, "ms": 23, "cs": 24, "ro": 25, "da": 26, "hu": 27, "ta": 28,
    "no": 29, "th": 30, "ur": 31, "hr": 32, "bg": 33, "lt": 34, "la": 35,
    "mi": 36, "ml": 37, "cy": 38, "sk": 39, "te": 40, "fa": 41, "lv": 42,
    "bn": 43, "sr": 44, "az": 45, "sl": 46, "kn": 47, "et": 48, "mk": 49,
    "br": 50, "eu": 51, "is": 52, "hy": 53, "ne": 54, "mn": 55, "bs": 56,
    "kk": 57, "sq": 58, "sw": 59, "gl": 60, "mr": 61, "pa": 62, "si": 63,
    "km": 64, "sn": 65, "yo": 66, "so": 67, "af": 68, "oc": 69, "ka": 70,
    "be": 71, "tg": 72, "sd": 73, "gu": 74, "am": 75, "yi": 76, "lo": 77,
    "uz": 78, "fo": 79, "ht": 80, "ps": 81, "tk": 82, "nn": 83, "mt": 84,
    "sa": 85, "lb": 86, "my": 87, "bo": 88, "tl": 89, "mg": 90, "as": 91,
    "tt": 92, "haw": 93, "ln": 94, "ha": 95, "ba": 96, "jw": 97, "su": 98,
}

# ISO 639-3 (HF ``xxx_Latn`` prefix) -> Whisper 2-letter keys.
# Languages absent here (e.g. ``mos``, ``dyu``) use Whisper auto-detect.
_HF_PREFIX_TO_WHISPER: dict[str, str] = {
    "fra": "fr", "eng": "en", "spa": "es", "por": "pt", "deu": "de",
    "ita": "it", "nld": "nl", "pol": "pl", "rus": "ru", "zho": "zh",
    "cmn": "zh", "jpn": "ja", "kor": "ko", "ara": "ar", "hin": "hi",
    "swe": "sv", "nob": "no", "nno": "nn", "dan": "da", "fin": "fi",
    "tur": "tr", "vie": "vi", "ces": "cs", "ukr": "uk", "ron": "ro",
    "ell": "el", "heb": "he", "msa": "ms", "ind": "id", "eus": "eu",
    "glg": "gl", "slk": "sk", "bul": "bg", "hrv": "hr", "srp": "sr",
    "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu",
    "pan": "pa", "kan": "kn", "slv": "sl", "lav": "lv", "lit": "lt",
    "est": "et", "mkd": "mk", "aze": "az", "bos": "bs", "cat": "ca",
    "cym": "cy", "gle": "ga", "mal": "ml", "swa": "sw",
}


def normalize_text(text: str) -> str:
    """Normalize text for metric computation.

    Applies: Unicode NFC normalization, lowercasing,
    punctuation stripping, whitespace collapsing.

    Args:
        text: Input text string.

    Returns:
        Normalized text.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_punctuation(text: str) -> str:
    """Remove all punctuation from text.

    Args:
        text: Input text.

    Returns:
        Text with punctuation removed.
    """
    return re.sub(r"[^\w\s]", "", text)


def tokenize_words(text: str) -> list:
    """Simple whitespace tokenizer.

    Args:
        text: Input text.

    Returns:
        List of word tokens.
    """
    return text.split()


def whisper_loopback_generate_kwargs(language: Optional[str]) -> Dict[str, Any]:
    """Build ``generate_kwargs`` for HF Whisper in TTS loopback / ASR.

    For unsupported ISO 639-3 prefixes (Moore, Dioula, ...),
    returns ``{}`` so Whisper uses automatic language detection.
    """
    if not language:
        return {}
    prefix = language.split("_")[0].lower()
    if prefix in WHISPER_LANG_TO_ID:
        code = prefix
    else:
        code = _HF_PREFIX_TO_WHISPER.get(prefix, prefix)
    if code in WHISPER_LANG_TO_ID:
        return {"forced_decoder_ids": [[2, WHISPER_LANG_TO_ID[code]]]}
    return {}


def normalize_language_code(code: str) -> str:
    """Normalize a language code to the HF/NLLB format.

    Handles common variants: ``fr`` → ``fra_Latn``, ``en`` → ``eng_Latn``.

    Args:
        code: Language code in any common format.

    Returns:
        Normalized HF-style code.
    """
    _ALIASES = {
        "fr": "fra_Latn", "fra": "fra_Latn", "french": "fra_Latn",
        "en": "eng_Latn", "eng": "eng_Latn", "english": "eng_Latn",
        "mos": "mos_Latn", "moore": "mos_Latn", "mooré": "mos_Latn",
        "dyu": "dyu_Latn", "dioula": "dyu_Latn", "dyula": "dyu_Latn",
        "fuv": "fuv_Latn", "fuf": "fuv_Latn", "fulfulde": "fuv_Latn",
        "fula": "fuv_Latn", "pulaar": "fuv_Latn",
        "gur": "gur_Latn", "gourmantchema": "gur_Latn",
    }
    normalized = code.strip().lower()
    return _ALIASES.get(normalized, code)

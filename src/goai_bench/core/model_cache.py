"""In-memory model cache — avoids reloading models across benchmark groups."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Keyed by a descriptive string like "mt:facebook/nllb-200-distilled-600M"
_MODEL_CACHE: Dict[str, Any] = {}


def get_cached(key: str) -> Any:
    """Return a cached model/pipeline, or ``None`` if not cached."""
    if key in _MODEL_CACHE:
        logger.info("Using cached model: %s", key)
        return _MODEL_CACHE[key]
    return None


def put_cached(key: str, value: Any) -> None:
    """Store a model/pipeline in the cache."""
    _MODEL_CACHE[key] = value


def clear_cache() -> None:
    """Clear all cached models (useful for testing)."""
    _MODEL_CACHE.clear()

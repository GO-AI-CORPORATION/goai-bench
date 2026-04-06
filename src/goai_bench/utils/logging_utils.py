"""Structured logging utilities."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Configure structured logging for the benchmark.

    Args:
        level: Logging level (``DEBUG``, ``INFO``, ``WARNING``).
        log_file: Optional file path for log output.
        verbose: If ``True``, sets level to ``DEBUG``.
    """
    if verbose:
        level = "DEBUG"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Reduce noise from third-party libraries — even in verbose mode,
    # we only want DEBUG output from our own goai_bench loggers.
    _noisy_libs = (
        "transformers", "datasets", "urllib3", "filelock",
        "httpcore", "httpx", "httpcore.connection", "httpcore.http11",
        "fsspec", "fsspec.local", "huggingface_hub",
        "huggingface_hub.utils", "huggingface_hub.file_download",
        "accelerate", "torch", "sentencepiece",
        "numba", "llvmlite",
    )
    for lib in _noisy_libs:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_library_versions() -> dict:
    """Return versions of key libraries for reproducibility.

    Returns:
        Dict mapping library name to version string.
    """
    versions = {}
    for lib in ("torch", "transformers", "sacrebleu", "jiwer",
                "datasets", "numpy", "pandas"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"
    return versions

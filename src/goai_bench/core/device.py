"""Device resolution utility — shared by all providers."""

import logging

logger = logging.getLogger(__name__)


def resolve_device(device: str = "auto") -> str:
    """Resolve ``'auto'`` to the best available device.

    Logs diagnostic information about GPU availability to help
    users identify when a CPU-only PyTorch build is installed.

    Args:
        device: One of ``cpu``, ``cuda``, ``mps``, ``auto``.

    Returns:
        Resolved device string.
    """
    if device != "auto":
        logger.info("Using explicit device: %s", device)
        return device
    try:
        import torch

        logger.info(
            "PyTorch %s | CUDA built: %s | CUDA available: %s",
            torch.__version__,
            torch.version.cuda or "no",
            torch.cuda.is_available(),
        )
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("Using GPU: %s", gpu_name)
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple MPS device")
            return "mps"
        logger.warning(
            "No GPU detected. If you have a GPU, install the CUDA version "
            "of PyTorch: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121"
        )
    except ImportError:
        pass
    return "cpu"

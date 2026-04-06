"""HuggingFace Hub utilities (token resolution, etc.)."""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_hf_token() -> Optional[str]:
    """Retrieve HF_TOKEN from environment or ``.env`` file at repo root."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                        return line.split("=", 1)[1].strip()
        except Exception as exc:
            logger.warning("Could not read .env file: %s", exc)
    return None

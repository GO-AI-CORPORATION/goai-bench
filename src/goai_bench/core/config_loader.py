"""YAML configuration loader and validator."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


class ConfigLoader:
    """Loads and validates YAML configuration files.

    Args:
        config_dir: Path to the configs/ directory. Defaults to the repo root.
    """

    def __init__(self, config_dir: Optional[str] = None) -> None:
        self.config_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load_languages(self) -> Dict[str, Dict[str, Any]]:
        """Load language metadata from languages.yaml."""
        data = self._load_yaml("languages.yaml")
        return data.get("languages", {})

    def load_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Load task definitions from tasks.yaml."""
        data = self._load_yaml("tasks.yaml")
        return data.get("tasks", {})

    def load_domains(self) -> Dict[str, Dict[str, Any]]:
        """Load domain definitions from domains.yaml."""
        data = self._load_yaml("domains.yaml")
        return data.get("domains", {})

    def load_dataset_config(self, language: str) -> Dict[str, Any]:
        """Load dataset configuration for a specific language.

        Args:
            language: HF language code (e.g. ``mos_Latn``).

        Returns:
            Parsed YAML dict for the language's dataset config.
        """
        path = self.config_dir / "datasets" / f"{language}.yaml"
        if not path.exists():
            logger.warning("No dataset config found for %s at %s", language, path)
            return {}
        return self._load_yaml(str(path), absolute=True)

    def get_dataset_source(
        self,
        language: str,
        task: str,
        source_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific dataset source config for a language/task.

        Args:
            language: HF language code (e.g. ``mos_Latn``).
            task: Task name (e.g. ``mt``, ``asr``, ``tts``).
            source_id: Optional source ID. If not provided, returns
                the first source for the task.

        Returns:
            Source config dict, or ``None`` if not found.
        """
        ds_cfg = self.load_dataset_config(language)
        sources = ds_cfg.get("tasks", {}).get(task, {}).get("sources", [])
        if not sources:
            return None
        if source_id:
            for src in sources:
                if src.get("id") == source_id:
                    return src
            logger.warning(
                "Source '%s' not found for %s/%s", source_id, language, task,
            )
            return None
        return sources[0]

    def get_benchmark_groups(
        self,
        language: str,
        task: str,
        source_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Get benchmark groups defined for a language/task source.

        Args:
            language: HF language code.
            task: Task name.
            source_id: Optional source ID.

        Returns:
            Dict mapping group name to group config (with ``splits`` list).
            Empty dict if no groups are defined.
        """
        source = self.get_dataset_source(language, task, source_id)
        if not source:
            return {}
        return source.get("benchmark_groups", {})

    def get_available_languages(self) -> List[str]:
        """Return list of language HF codes that have config files."""
        ds_dir = self.config_dir / "datasets"
        if not ds_dir.exists():
            return []
        return sorted(
            p.stem for p in ds_dir.glob("*.yaml")
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_yaml(self, filename: str, absolute: bool = False) -> Dict[str, Any]:
        """Load a YAML file and return its contents as a dict."""
        path = Path(filename) if absolute else self.config_dir / filename
        if not path.exists():
            logger.warning("Config file not found: %s", path)
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            try:
                return yaml.safe_load(fh) or {}
            except yaml.YAMLError as exc:
                logger.error("Error parsing %s: %s", path, exc)
                return {}

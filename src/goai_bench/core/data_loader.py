"""Unified dataset loading — HuggingFace Hub only."""

import logging
from typing import Any, Dict, List, Optional

from goai_bench.utils.hf_utils import get_hf_token

logger = logging.getLogger(__name__)


class DataLoader:
    """Load evaluation data from HuggingFace Hub.

    All ``load_*`` methods return a list of dicts. They never crash on
    missing data — instead they log a warning and return an empty list.
    """

    def __init__(self) -> None:
        self._hf_token = get_hf_token()

    # ==================================================================
    # Config-driven entry point
    # ==================================================================

    def load_from_config(
        self,
        source_cfg: Dict[str, Any],
        task: str,
        split: Optional[str] = None,
        domain: str = "all",
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load data using a source configuration dict from YAML.

        Args:
            source_cfg: Source dict from ``configs/datasets/<lang>.yaml``.
            task: One of ``mt``, ``asr``, ``tts``.
            split: Override split from config.
            domain: Domain filter or ``"all"``.
            max_samples: Cap number of returned samples.

        Returns:
            List of dicts with task-specific keys.
        """
        if task == "mt":
            dataset_id = source_cfg["hf_dataset"]
            resolved_split = self._resolve_split(source_cfg, split)
            src_col = source_cfg.get("source_column", "source")
            tgt_col = source_cfg.get("target_column", "target")
            # FLORES+ configs use sentence_{lang} columns. If the user overrides
            # source_lang/target_lang (e.g. mos→fra) but YAML still lists the
            # opposite pair, metrics collapse (wrong language vs reference).
            src_lang = source_cfg.get("source_lang")
            tgt_lang = source_cfg.get("target_lang")
            if (
                src_lang and tgt_lang
                and str(src_col).startswith("sentence_")
                and str(tgt_col).startswith("sentence_")
            ):
                aligned_src = f"sentence_{src_lang}"
                aligned_tgt = f"sentence_{tgt_lang}"
                if src_col != aligned_src or tgt_col != aligned_tgt:
                    logger.info(
                        "Aligning MT columns to source_lang/target_lang: "
                        "%s, %s (was %s, %s)",
                        aligned_src, aligned_tgt, src_col, tgt_col,
                    )
                src_col, tgt_col = aligned_src, aligned_tgt
            domain_col = source_cfg.get("domain_column")
            return self._load_mt_hf(
                dataset_id, resolved_split, src_col, tgt_col,
                domain, max_samples, domain_col=domain_col,
            )
        if task == "asr":
            return self._load_asr_hf(source_cfg, split, max_samples)
        if task == "tts":
            return self._load_tts_hf(source_cfg, split, domain, max_samples)

        logger.warning("Unsupported task '%s'", task)
        return []

    # ==================================================================
    # Benchmark group loading
    # ==================================================================

    def load_benchmark_group(
        self,
        source_cfg: Dict[str, Any],
        task: str,
        group_name: str,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load data for a benchmark group, merging multiple splits if needed.

        Args:
            source_cfg: Source dict from YAML config.
            task: One of ``mt``, ``asr``, ``tts``.
            group_name: Benchmark group name (e.g. ``general``, ``health``).
            max_samples: Cap number of returned samples.

        Returns:
            List of sample dicts with ``domain`` set to ``group_name``.
        """
        groups = source_cfg.get("benchmark_groups", {})
        group = groups.get(group_name)
        if not group:
            logger.warning("Benchmark group '%s' not found in config", group_name)
            return []

        splits = group.get("splits", [])
        all_data: List[Dict[str, Any]] = []
        for split in splits:
            data = self.load_from_config(source_cfg, task, split=split)
            all_data.extend(data)

        for item in all_data:
            item["domain"] = group_name

        if max_samples and len(all_data) > max_samples:
            all_data = all_data[:max_samples]

        logger.info(
            "Loaded %d samples for benchmark group '%s' (%s, %d splits)",
            len(all_data), group_name, task, len(splits),
        )
        return all_data

    # ==================================================================
    # Public loading methods (prefix-based API)
    # ==================================================================

    def load_mt_data(
        self,
        source: str,
        language: str,
        domain: str = "all",
        source_lang: str = "fra_Latn",
        target_lang: str = "mos_Latn",
        max_samples: Optional[int] = None,
        split: str = "devtest",
        source_column: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load machine translation evaluation data from HuggingFace.

        Args:
            source: ``"hf:<dataset_id>"``.
            language: HF language code.
            domain: Domain filter or ``"all"``.
            source_lang: Source language code.
            target_lang: Target language code.
            max_samples: Cap number of returned samples.
            split: HF dataset split name.
            source_column: Override source text column name.
            target_column: Override target text column name.

        Returns:
            List of dicts with keys ``source``, ``reference``,
            ``domain``, ``id``.
        """
        if source.startswith("hf:"):
            src_col = source_column or f"sentence_{source_lang}"
            tgt_col = target_column or f"sentence_{target_lang}"
            return self._load_mt_hf(
                source[3:], split, src_col, tgt_col,
                domain, max_samples,
            )
        logger.warning("Unknown source prefix in '%s'. Use hf:<dataset_id>.", source)
        return []

    # ==================================================================
    # HuggingFace loading — unified helpers
    # ==================================================================

    def _load_hf_dataset(self, dataset_id: str, split: str) -> Any:
        """Load a single HF dataset split with token support.

        Args:
            dataset_id: HuggingFace dataset identifier.
            split: Dataset split name.

        Returns:
            HF dataset object, or ``None`` on failure.
        """
        try:
            from datasets import load_dataset

            return load_dataset(
                dataset_id,
                split=split,
                trust_remote_code=False,
                token=self._hf_token,
            )
        except Exception as exc:
            logger.warning(
                "Could not load HF dataset '%s' split '%s': %s",
                dataset_id, split, exc,
            )
            return None

    def _resolve_split(
        self, source_cfg: Dict[str, Any], split_override: Optional[str],
    ) -> str:
        """Determine which split to use from config or override.

        Args:
            source_cfg: Source configuration dict.
            split_override: Explicit split override.

        Returns:
            Resolved split name.
        """
        if split_override:
            return split_override
        if "hf_split" in source_cfg:
            return source_cfg["hf_split"]
        splits = source_cfg.get("hf_splits", [])
        if splits:
            return splits[0]
        return "test"

    def _load_mt_hf(
        self,
        dataset_id: str,
        split: str,
        source_column: str,
        target_column: str,
        domain: str,
        max_samples: Optional[int],
        domain_col: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load MT data from a HuggingFace dataset.

        Args:
            dataset_id: HF dataset identifier.
            split: Dataset split.
            source_column: Column for source text.
            target_column: Column for target text.
            domain: Domain filter or ``"all"``.
            max_samples: Cap samples.
            domain_col: Optional column for per-row domain labels.

        Returns:
            List of MT sample dicts.
        """
        ds = self._load_hf_dataset(dataset_id, split)
        if ds is None:
            return []

        if source_column not in ds.column_names:
            logger.warning(
                "Column '%s' not in dataset columns %s",
                source_column, ds.column_names,
            )
            return []
        if target_column not in ds.column_names:
            logger.warning(
                "Column '%s' not in dataset columns %s",
                target_column, ds.column_names,
            )
            return []

        results: List[Dict[str, Any]] = []
        for i, row in enumerate(ds):
            src = row.get(source_column, "")
            tgt = row.get(target_column, "")
            if not src or not tgt:
                continue

            row_domain = "general"
            if domain_col and domain_col in ds.column_names:
                row_domain = row.get(domain_col, "general") or "general"
            elif split not in ("test", "devtest", "train", "validation"):
                row_domain = split
            elif domain != "all":
                row_domain = domain

            if domain != "all" and row_domain != domain:
                continue

            results.append({
                "source": src,
                "reference": tgt,
                "domain": row_domain,
                "id": f"{dataset_id}_{split}_{i}",
            })
            if max_samples and len(results) >= max_samples:
                break

        logger.info(
            "Loaded %d MT samples from HF %s/%s",
            len(results), dataset_id, split,
        )
        return results

    def _load_asr_hf(
        self,
        source_cfg: Dict[str, Any],
        split: Optional[str],
        max_samples: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Load ASR data from a HuggingFace dataset.

        Args:
            source_cfg: Source configuration dict.
            split: Optional split override.
            max_samples: Cap samples.

        Returns:
            List of ASR sample dicts with ``audio_array``.
        """
        dataset_id = source_cfg["hf_dataset"]
        resolved_split = self._resolve_split(source_cfg, split)
        audio_col = source_cfg.get("audio_column", "audio")
        text_col = source_cfg.get("text_column", "text")

        ds = self._load_hf_dataset(dataset_id, resolved_split)
        if ds is None:
            return []

        for col, label in [(audio_col, "Audio"), (text_col, "Text")]:
            if col not in ds.column_names:
                logger.warning(
                    "%s column '%s' not in dataset columns %s",
                    label, col, ds.column_names,
                )
                return []

        results: List[Dict[str, Any]] = []
        for i, row in enumerate(ds):
            audio_data = row.get(audio_col)
            text = row.get(text_col, "")
            if not audio_data or not text:
                continue
            results.append({
                "audio_array": audio_data["array"],
                "sampling_rate": audio_data.get("sampling_rate", 16000),
                "transcript": text,
                "domain": row.get("domain", "general"),
                "speaker_id": str(row.get("speaker_id", "")),
                "duration": float(row.get("duration", 0)),
                "id": f"{dataset_id}_{resolved_split}_{i}",
            })
            if max_samples and len(results) >= max_samples:
                break

        logger.info(
            "Loaded %d ASR samples from HF %s/%s",
            len(results), dataset_id, resolved_split,
        )
        return results

    def _load_tts_hf(
        self,
        source_cfg: Dict[str, Any],
        split: Optional[str],
        domain: str,
        max_samples: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Load TTS text prompts from a HuggingFace dataset.

        Args:
            source_cfg: Source configuration dict.
            split: Optional split override.
            domain: Domain filter or ``"all"``.
            max_samples: Cap samples.

        Returns:
            List of TTS sample dicts with ``text``, ``domain``.
        """
        dataset_id = source_cfg["hf_dataset"]
        resolved_split = self._resolve_split(source_cfg, split)
        text_col = source_cfg.get("text_column", "text")
        domain_col = source_cfg.get("domain_column")

        ds = self._load_hf_dataset(dataset_id, resolved_split)
        if ds is None:
            return []

        if text_col not in ds.column_names:
            logger.warning(
                "Text column '%s' not in dataset columns %s",
                text_col, ds.column_names,
            )
            return []

        results: List[Dict[str, Any]] = []
        for i, row in enumerate(ds):
            text = row.get(text_col, "")
            if not text:
                continue
            row_domain = "general"
            if domain_col and domain_col in ds.column_names:
                row_domain = row.get(domain_col, "general") or "general"
            elif resolved_split not in ("test", "devtest", "train", "validation"):
                row_domain = resolved_split

            if domain != "all" and row_domain != domain:
                continue

            results.append({
                "text": text,
                "domain": row_domain,
                "sentence_id": str(
                    row.get("id", f"{dataset_id}_{resolved_split}_{i}"),
                ),
            })
            if max_samples and len(results) >= max_samples:
                break

        logger.info(
            "Loaded %d TTS samples from HF %s/%s",
            len(results), dataset_id, resolved_split,
        )
        return results

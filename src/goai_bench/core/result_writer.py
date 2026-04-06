"""Result serialization, leaderboard management, and export."""

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import filelock
import pandas as pd

from goai_bench.visualization.leaderboard import (
    TASK_PRIMARY_METRICS,
    normalize_score,
)

logger = logging.getLogger(__name__)


def _compute_composite(result_dict: Dict[str, Any]) -> float:
    """Compute a composite score normalized to [0, 1]."""
    task = result_dict.get("task", "")
    primary_score = _extract_primary_score_static(result_dict)
    metric, _ = TASK_PRIMARY_METRICS.get(task, ("score", "higher"))
    return normalize_score(primary_score, metric)


def _extract_primary_score_static(d: Dict[str, Any]) -> float:
    """Extract primary score from a result dict (standalone helper)."""
    task = d.get("task", "")
    mapping = {"mt": "overall_chrf", "asr": "overall_wer", "tts": "overall_utmos"}
    key = mapping.get(task, "primary_score")
    return float(d.get(key, d.get("primary_score", 0.0)) or 0.0)


def _fmt(val: Any) -> str:
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def result_output_base(
    output_dir: str,
    benchmark_language: str,
    task: str,
    result_dict: Dict[str, Any],
) -> Path:
    """Result root: ``<out>/<lang>/<task>`` or ``<out>/<lang>/mt/<src>_<tgt>``."""
    p = Path(output_dir) / benchmark_language / task
    if task == "mt":
        src = result_dict.get("source_lang", "")
        tgt = result_dict.get("target_lang", "")
        if src and tgt:
            p = p / f"{src}_{tgt}"
    return p


def safe_model_dir_slug(model_id: str) -> str:
    """Filesystem-safe directory name for a model id (HF id or API URL).

    Args:
        model_id: HuggingFace id or ``http://host:port`` style API base.

    Returns:
        Slug without path-invalid characters (Windows-safe).
    """
    raw = (model_id or "model").strip()
    tail = raw.split("/")[-1]
    for ch in (":", "\\", "?", "*", "|", "<", ">", '"'):
        tail = tail.replace(ch, "_")
    return tail or "model"


class ResultWriter:
    """Save evaluation results and manage the leaderboard JSON store.

    Args:
        leaderboard_path: Path to the master ``leaderboard.json``.
    """

    def __init__(
        self, leaderboard_path: str = "leaderboard/leaderboard.json",
    ) -> None:
        self.leaderboard_path = Path(leaderboard_path)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def result_to_dict(self, result: Any) -> Dict[str, Any]:
        """Convert any ``*Result`` dataclass to a serializable dict.

        Adds ``task`` (inferred from type name) and ``composite_score``.
        """
        if is_dataclass(result) and not isinstance(result, type):
            d = asdict(result)
        else:
            d = dict(result) if hasattr(result, "__iter__") else {}

        # Infer task from class name
        cls_name = type(result).__name__.lower()
        for task in ("mt", "asr", "tts"):
            if task in cls_name:
                d["task"] = task
                break

        d.setdefault("task", "unknown")
        d["composite_score"] = _compute_composite(d)
        return d

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_result(
        self,
        result: Any,
        output_dir: str,
        group_name: Optional[str] = None,
        benchmark_language: Optional[str] = None,
    ) -> str:
        """Save result as JSON and CSV.

        With ``group_name``, layout is::

            <out>/<benchmark_language>/<task>/[<src>_<tgt>/]<model>/<group>.json

        MT includes ``<src>_<tgt>`` so reverse directions do not collide.

        Args:
            result: A ``*Result`` dataclass (e.g. ``MTResult``).
            output_dir: Root output directory.
            group_name: Benchmark group (e.g. ``general``, ``health``).
            benchmark_language: Value of CLI ``--language`` (required with group_name).

        Returns:
            Path to the saved JSON file.
        """
        d = self.result_to_dict(result)

        model_slug = safe_model_dir_slug(d.get("model_id", "model"))
        task = d.get("task", "unknown")
        bl = benchmark_language or d.get("language")
        if not bl:
            bl = d.get("target_lang") or d.get("source_lang") or "unknown"

        if group_name:
            base = result_output_base(output_dir, bl, task, d)
            out = base / model_slug
            filename = f"{group_name}.json"
        else:
            out = Path(output_dir)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            lang_tag = bl
            filename = f"{task}_{model_slug}_{lang_tag}_{ts}.json"

        out.mkdir(parents=True, exist_ok=True)
        filepath = out / filename

        compact = {
            k: v for k, v in d.items()
            if k not in ("hypotheses", "references", "sources",
                         "predictions", "loopback_transcripts",
                         "input_texts", "confidences", "domains")
        }
        compact["benchmark_language"] = bl
        if task == "mt":
            compact["mt_pair"] = (
                f"{d.get('source_lang', '')}_{d.get('target_lang', '')}"
            )
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(compact, fh, indent=2, ensure_ascii=False, default=str)

        logger.info("Result saved to %s", filepath)

        csv_path = filepath.with_suffix(".csv")
        try:
            pd.DataFrame([compact]).to_csv(csv_path, index=False)
        except Exception as exc:
            logger.warning("Could not write CSV summary: %s", exc)

        return str(filepath)

    # ------------------------------------------------------------------
    # Summary & Comparison
    # ------------------------------------------------------------------

    _METRIC_KEYS = {
        "mt": ["chrf", "bleu", "ter", "comet_kiwi"],
        "asr": ["wer", "cer", "mer"],
        "tts": ["utmos", "loopback_wer", "loopback_cer"],
    }

    def save_summary(
        self,
        task: str,
        model_id: str,
        language: str,
        group_results: Dict[str, Any],
        output_dir: str,
    ) -> str:
        """Save a summary JSON aggregating all benchmark group results.

        Writes ``summary.json`` alongside the per-group files.

        Args:
            task: Task name.
            model_id: Model identifier.
            language: HF language code.
            group_results: Mapping of group_name to result dataclass.
            output_dir: Root output directory.

        Returns:
            Path to the saved summary file.
        """
        first = next(iter(group_results.values()))
        fd = self.result_to_dict(first)
        model_slug = safe_model_dir_slug(model_id)
        base = result_output_base(output_dir, language, task, fd)
        out = base / model_slug
        out.mkdir(parents=True, exist_ok=True)

        metric_keys = self._METRIC_KEYS.get(task, [])
        groups_data = {}
        total_samples = 0

        for group_name, result in group_results.items():
            d = self.result_to_dict(result)
            metrics = d.get("all_metrics", {})
            entry = {"n_samples": d.get("n_samples", 0)}
            for mk in metric_keys:
                entry[mk] = metrics.get(mk)
            groups_data[group_name] = entry
            total_samples += entry["n_samples"]

        averages = {}
        for mk in metric_keys:
            vals = [
                g[mk] for g in groups_data.values()
                if g[mk] is not None
            ]
            averages[mk] = sum(vals) / len(vals) if vals else None

        summary = {
            "model_id": model_id,
            "task": task,
            "benchmark_language": language,
            "total_samples": total_samples,
            "groups": groups_data,
            "groups_evaluated": sorted(group_results.keys()),
            "average": averages,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if task == "mt":
            summary["mt_pair"] = (
                f"{fd.get('source_lang', '')}_{fd.get('target_lang', '')}"
            )

        filepath = out / "summary.json"
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

        logger.info("Summary saved to %s", filepath)
        return str(filepath)

    def generate_comparison(
        self,
        results_dir: str,
        language: str,
        task: str,
        mt_pair: Optional[str] = None,
    ) -> str:
        """Write ``comparison.json`` and ``comparison.md`` under the task tree.

        Layout: ``<results>/<language>/<task>/`` or for MT
        ``<results>/<language>/mt/<source>_<target>/``.

        Args:
            results_dir: Root results directory.
            language: CLI ``--language`` (benchmark language).
            task: Task name.
            mt_pair: For MT, ``source_target`` subfolder; if None, all MT pairs.

        Returns:
            Path to the last comparison markdown written, or ``""``.
        """
        if task == "mt":
            mt_root = Path(results_dir) / language / "mt"
            if mt_pair:
                subdirs = [mt_root / mt_pair] if (mt_root / mt_pair).is_dir() else []
            else:
                subdirs = sorted(
                    [p for p in mt_root.iterdir() if p.is_dir()],
                )
            last = ""
            for td in subdirs:
                path = self._generate_comparison_in_dir(
                    td, language, task, td.name,
                )
                if path:
                    last = path
            return last

        task_dir = Path(results_dir) / language / task
        return self._generate_comparison_in_dir(task_dir, language, task, "")

    def _generate_comparison_in_dir(
        self,
        task_dir: Path,
        benchmark_language: str,
        task: str,
        mt_pair: str,
    ) -> str:
        """Build comparison files inside ``task_dir`` (contains model subdirs)."""
        if not task_dir.is_dir():
            logger.warning("No results at %s", task_dir)
            return ""

        metric_keys = self._METRIC_KEYS.get(task, [])
        primary_metric = self._primary_metric_for(task)
        lower_is_better = task in ("asr",)

        models_data: List[Dict[str, Any]] = []

        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            summary_path = model_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                models_data.append(data)
                continue

            groups = {}
            for jf in sorted(model_dir.glob("*.json")):
                if jf.name in ("summary.json", "comparison.json"):
                    continue
                with open(jf, "r", encoding="utf-8") as fh:
                    gdata = json.load(fh)
                group_name = jf.stem
                metrics = gdata.get("all_metrics", {})
                entry = {"n_samples": gdata.get("n_samples", 0)}
                for mk in metric_keys:
                    entry[mk] = metrics.get(mk)
                groups[group_name] = entry

            if not groups:
                continue

            averages = {}
            for mk in metric_keys:
                vals = [
                    g[mk] for g in groups.values() if g[mk] is not None
                ]
                averages[mk] = sum(vals) / len(vals) if vals else None

            total = sum(g["n_samples"] for g in groups.values())
            models_data.append({
                "model_id": model_dir.name,
                "task": task,
                "benchmark_language": benchmark_language,
                "mt_pair": mt_pair or None,
                "total_samples": total,
                "groups": groups,
                "average": averages,
            })

        if not models_data:
            logger.warning("No model results found in %s", task_dir)
            return ""

        models_data.sort(
            key=lambda m: m.get("average", {}).get(primary_metric) or 0,
            reverse=not lower_is_better,
        )

        comparison_json = task_dir / "comparison.json"
        with open(comparison_json, "w", encoding="utf-8") as fh:
            json.dump(models_data, fh, indent=2, ensure_ascii=False, default=str)

        md_path = task_dir / "comparison.md"
        md_lines = self._build_comparison_md(
            task, benchmark_language, models_data, metric_keys, primary_metric,
            mt_pair,
        )
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        logger.info("Comparison saved to %s", md_path)
        return str(md_path)

    @staticmethod
    def _build_comparison_md(
        task: str,
        benchmark_language: str,
        models_data: List[Dict[str, Any]],
        metric_keys: List[str],
        primary_metric: str,
        mt_pair: str = "",
    ) -> List[str]:
        """Build markdown lines for the comparison table."""
        all_groups = []
        for m in models_data:
            for g in m.get("groups", {}):
                if g not in all_groups:
                    all_groups.append(g)

        lines = [
            f"# {task.upper()} model comparison\n",
            f"*CLI language (--language): {benchmark_language}*",
            f"*Task: {task}*",
        ]
        if task == "mt" and mt_pair:
            lines.append(f"*MT direction (source_target): {mt_pair}*")
        lines.append(
            f"*Benchmark groups (columns): {', '.join(all_groups) or 'n/a'}*",
        )
        lines.append(f"*Primary metric: {primary_metric}*\n")

        for mk in metric_keys:
            lines.append(f"\n## {mk.upper()}\n")

            header = "| Rank | Model |"
            sep = "|------|-------|"
            for g in all_groups:
                header += f" {g} |"
                sep += "------|"
            header += " Average |"
            sep += "---------|"
            lines.extend([header, sep])

            for rank, m in enumerate(models_data, 1):
                model_name = m.get("model_id", "?")
                row = f"| {rank} | {model_name} |"
                for g in all_groups:
                    val = (m.get("groups", {}).get(g, {}) or {}).get(mk)
                    row += f" {_fmt(val)} |"
                avg = (m.get("average", {}) or {}).get(mk)
                row += f" **{_fmt(avg)}** |"
                lines.append(row)

        lines.append(
            f"\n*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
        )
        return lines

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def append_to_leaderboard(self, result: Any) -> None:
        """Thread-safe append to leaderboard.json using file locking.

        Deduplicates by (model_id, language, task, domain) — keeps the
        better score.
        """
        d = self.result_to_dict(result)
        entry = self._make_entry(d)

        self.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = str(self.leaderboard_path) + ".lock"

        with filelock.FileLock(lock_path, timeout=30):
            data = self._read_leaderboard()
            existing_idx = self._find_existing(data["entries"], entry)

            if existing_idx is not None:
                old = data["entries"][existing_idx]
                if self._is_better(entry, old):
                    data["entries"][existing_idx] = entry
            else:
                data["entries"].append(entry)

            data["last_updated"] = entry["timestamp"]
            self._write_leaderboard(data)

        logger.info("Leaderboard updated: %s", self.leaderboard_path)

    def export_markdown_leaderboard(
        self, output_path: str = "leaderboard/LEADERBOARD.md",
    ) -> None:
        """Generate a markdown file with tables per task per language."""
        data = self._read_leaderboard()
        entries = data.get("entries", [])
        if not entries:
            Path(output_path).write_text(
                "# Leaderboard\n\nNo results yet.\n", encoding="utf-8",
            )
            return

        df = pd.DataFrame(entries)
        lines = ["# GO AI Bench — Leaderboard\n"]
        lines.append(
            f"*Last updated: {data.get('last_updated', 'N/A')}*\n"
        )

        for task in sorted(df["task"].unique()):
            lines.append(f"\n## {task.upper()}\n")
            task_df = df[df["task"] == task]
            for lang in sorted(task_df["language"].unique()):
                lang_df = task_df[task_df["language"] == lang].copy()
                asc = task in ("asr",)
                lang_df = lang_df.sort_values(
                    "primary_score", ascending=asc,
                )
                lines.append(f"\n### {lang}\n")
                lines.append(
                    "| Rank | Model | Score | Metric | Date |"
                )
                lines.append(
                    "|------|-------|-------|--------|------|"
                )
                for rank, (_, row) in enumerate(lang_df.iterrows(), 1):
                    lines.append(
                        f"| {rank} | {row['model_id']} | "
                        f"{row['primary_score']:.2f} | "
                        f"{row['primary_metric']} | "
                        f"{row.get('timestamp', '')[:10]} |"
                    )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown leaderboard exported to %s", output_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_entry(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create a leaderboard entry dict from a result dict."""
        return {
            "rank": None,
            "model_id": d.get("model_id", ""),
            "language": d.get(
                "language",
                d.get("target_lang", d.get("source_lang", "")),
            ),
            "task": d.get("task", ""),
            "domain": "all",
            "primary_metric": self._primary_metric_for(d.get("task", "")),
            "primary_score": self._extract_primary_score(d),
            "all_metrics": d.get("all_metrics", self._build_all_metrics(d)),
            "per_domain_scores": d.get(
                "per_domain_scores", d.get("per_domain", {}),
            ),
            "n_samples": d.get("n_samples", 0),
            "timestamp": d.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat(),
            ),
            "dataset_id": d.get("dataset_id", ""),
            "is_baseline": d.get("is_baseline", False),
            "composite_score": d.get("composite_score", 0.0),
        }

    @staticmethod
    def _primary_metric_for(task: str) -> str:
        return {
            "mt": "chrf", "asr": "wer", "tts": "utmos",
        }.get(task, "score")

    @staticmethod
    def _extract_primary_score(d: Dict[str, Any]) -> float:
        task = d.get("task", "")
        mapping = {
            "mt": "overall_chrf",
            "asr": "overall_wer",
            "tts": "overall_utmos",
        }
        key = mapping.get(task, "primary_score")
        return float(d.get(key, d.get("primary_score", 0.0)) or 0.0)

    @staticmethod
    def _build_all_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}
        for k, v in d.items():
            if k.startswith("overall_") and v is not None:
                metrics[k.replace("overall_", "")] = v
        return metrics

    def _read_leaderboard(self) -> Dict[str, Any]:
        if not self.leaderboard_path.exists():
            return {
                "last_updated": "",
                "entries": [],
            }
        try:
            with open(self.leaderboard_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "Corrupted leaderboard at %s, reinitializing: %s",
                self.leaderboard_path, exc,
            )
            return {"last_updated": "", "entries": []}

    def _write_leaderboard(self, data: Dict[str, Any]) -> None:
        with open(self.leaderboard_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _find_existing(
        entries: list, new_entry: Dict[str, Any],
    ) -> Optional[int]:
        for i, e in enumerate(entries):
            if (
                e.get("model_id") == new_entry.get("model_id")
                and e.get("language") == new_entry.get("language")
                and e.get("task") == new_entry.get("task")
                and e.get("domain") == new_entry.get("domain")
            ):
                return i
        return None

    @staticmethod
    def _is_better(
        new_entry: Dict[str, Any], old_entry: Dict[str, Any],
    ) -> bool:
        new_n = new_entry.get("n_samples", 0)
        old_n = old_entry.get("n_samples", 0)
        if new_n > old_n * 2:
            return True
        task = new_entry.get("task", "")
        new_s = new_entry.get("primary_score", 0.0)
        old_s = old_entry.get("primary_score", 0.0)
        if task in ("asr",):
            return new_s < old_s
        return new_s > old_s

"""Chart generation functions for all evaluation tasks.

All charts use matplotlib with a consistent domain color palette.
Every function returns a ``matplotlib.figure.Figure`` and optionally
saves to file.
"""

import logging
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Domain → colour mapping
PALETTE: Dict[str, str] = {
    "health": "#2196F3",
    "education": "#4CAF50",
    "child_protection": "#FF9800",
    "environment": "#8BC34A",
    "general": "#9E9E9E",
}

_MODEL_COLORS = [
    "#1a6b72", "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#e67e22", "#1abc9c", "#c0392b", "#2980b9",
]


def _safe_figure(func):
    """Decorator: catch errors and return an empty figure."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.warning("Chart generation failed (%s): %s", func.__name__, exc)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Chart unavailable", ha="center", va="center",
                    fontsize=14, color="#999")
            ax.set_axis_off()
            return fig
    return wrapper


@_safe_figure
def plot_mt_domain_comparison(
    result: Any, save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart: domains × chrF++ and BLEU side by side.

    Args:
        result: ``MTResult`` dataclass.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    domains = sorted(result.per_domain.keys())
    chrf_scores = [result.per_domain[d].get("chrf", 0) for d in domains]
    bleu_scores = [result.per_domain[d].get("bleu", 0) for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, chrf_scores, width, label="chrF++",
           color="#1a6b72", alpha=0.85)
    ax.bar(x + width / 2, bleu_scores, width, label="BLEU",
           color="#e74c3c", alpha=0.85)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Score")
    ax.set_title(
        f"MT Domain Comparison — {result.model_id}\n"
        f"{result.source_lang} → {result.target_lang}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@_safe_figure
def plot_model_comparison(
    results: List[Any],
    metric: str,
    task: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart comparing models on a single metric.

    Args:
        results: List of ``*Result`` dataclasses.
        metric: Metric key (e.g. ``"overall_chrf"``).
        task: Task name for title.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure.
    """
    model_names = [r.model_id for r in results]
    scores = [getattr(r, metric, 0.0) or 0.0 for r in results]

    # Sort descending (or ascending for WER/CER)
    lower_better = metric in ("overall_wer", "overall_cer", "overall_mer",
                              "overall_ter")
    paired = sorted(zip(scores, model_names), reverse=not lower_better)
    scores, model_names = zip(*paired) if paired else ([], [])

    fig, ax = plt.subplots(figsize=(10, max(4, len(model_names) * 0.6)))
    colors = [_MODEL_COLORS[i % len(_MODEL_COLORS)]
              for i in range(len(model_names))]
    ax.barh(range(len(model_names)), scores, color=colors, alpha=0.85)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel(metric.replace("overall_", "").upper())
    ax.set_title(f"Model Comparison — {task.upper()} ({metric})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@_safe_figure
def plot_radar(
    results: List[Dict[str, Any]],
    metric_label: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Radar/spider chart with one axis per domain, one line per model.

    Args:
        results: List of dicts ``{"model": str, "scores": {domain: float}}``.
        metric_label: Label for the metric being plotted.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure.
    """
    if not results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    domains = sorted(results[0]["scores"].keys())
    n = len(domains)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for i, entry in enumerate(results):
        values = [entry["scores"].get(d, 0) for d in domains]
        values += values[:1]
        color = _MODEL_COLORS[i % len(_MODEL_COLORS)]
        ax.plot(angles, values, "-o", color=color, label=entry["model"],
                linewidth=2, markersize=5)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_title(f"Domain Comparison — {metric_label}", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@_safe_figure
def plot_leaderboard_table(
    leaderboard_df: Any,
    task: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Styled matplotlib table with rank, model, scores.

    Args:
        leaderboard_df: pandas DataFrame with leaderboard data.
        task: Task name for title.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure.
    """
    import pandas as pd
    df = leaderboard_df.head(20)
    cols = ["model_id", "language", "primary_score"]
    available = [c for c in cols if c in df.columns]
    if not available:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    display_df = df[available].copy()
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))

    fig, ax = plt.subplots(
        figsize=(12, max(3, len(display_df) * 0.4 + 1)),
    )
    ax.set_axis_off()
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Gold highlight for rank 1
    if len(display_df) > 0:
        for j in range(len(display_df.columns)):
            table[1, j].set_facecolor("#FFF8E1")

    ax.set_title(f"Leaderboard — {task.upper()}", fontsize=14, pad=20)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@_safe_figure
def plot_asr_wer_cer_comparison(
    result: Any, save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side WER and CER bars grouped by domain.

    Args:
        result: ``ASRResult`` dataclass.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure.
    """
    domains = sorted(result.per_domain.keys())
    wer_scores = [result.per_domain[d].get("wer", 0) for d in domains]
    cer_scores = [result.per_domain[d].get("cer", 0) for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, wer_scores, width, label="WER",
           color="#e74c3c", alpha=0.85)
    ax.bar(x + width / 2, cer_scores, width, label="CER",
           color="#3498db", alpha=0.85)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Error Rate")
    ax.set_title(f"ASR — {result.model_id} ({result.language})")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@_safe_figure
def plot_tts_combined(
    result: Any, save_path: Optional[str] = None,
) -> plt.Figure:
    """Two subplots: UTMOS per domain + loopback WER per domain.

    Args:
        result: ``TTSResult`` dataclass.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure.
    """
    domains = sorted(result.per_domain_utmos.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # UTMOS
    utmos_vals = [result.per_domain_utmos.get(d) or 0 for d in domains]
    colors = [PALETTE.get(d, "#9E9E9E") for d in domains]
    ax1.bar(domains, utmos_vals, color=colors, alpha=0.85)
    ax1.set_ylabel("UTMOS Score")
    ax1.set_title("Naturalness (UTMOS)")
    ax1.set_ylim(0, 5)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis="x", rotation=30)

    # Loopback WER
    wer_vals = [result.per_domain_loopback_wer.get(d, 1.0) for d in domains]
    ax2.bar(domains, wer_vals, color=colors, alpha=0.85)
    ax2.set_ylabel("Loopback WER")
    ax2.set_title("Intelligibility (Loopback WER)")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=30)

    fig.suptitle(f"TTS — {result.model_id} ({result.language})", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig



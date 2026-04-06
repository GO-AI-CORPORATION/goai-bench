"""Leaderboard table generation and ranking logic."""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

TASK_PRIMARY_METRICS = {
    "mt": ("chrf", "higher"),
    "asr": ("wer", "lower"),
    "tts": ("utmos", "higher"),
}

COMPOSITE_WEIGHTS = {
    "mt": 0.40,
    "asr": 0.40,
    "tts": 0.20,
}


def normalize_score(score: float, metric: str) -> float:
    """Normalize a metric score to [0, 1] where 1 = best.

    Args:
        score: Raw metric value.
        metric: Metric name.

    Returns:
        Normalized score in [0, 1].
    """
    if metric in ("wer", "cer", "mer", "ter"):
        return max(0.0, 1.0 - score)
    if metric == "chrf":
        return score / 100.0
    if metric == "bleu":
        return score / 100.0
    if metric == "utmos":
        return score / 5.0
    return float(score)


def compute_composite_score(row: pd.Series) -> float:
    """Compute weighted composite score for a leaderboard entry.

    Used for cross-task ranking in the general leaderboard.

    Args:
        row: pandas Series with ``task`` and ``primary_score``.

    Returns:
        Composite score in [0, 1].
    """
    task = row.get("task", "")
    score = row.get("primary_score", 0.0)
    metric, _ = TASK_PRIMARY_METRICS.get(task, ("score", "higher"))
    return normalize_score(score, metric)


def compute_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank column to a DataFrame of leaderboard entries.

    Ranks within each ``(language, task)`` group based on the
    primary metric and its direction (higher/lower is better).

    Args:
        df: DataFrame with columns ``language``, ``task``,
            ``primary_score``.

    Returns:
        DataFrame with ``rank`` column added, sorted by
        ``(task, language, rank)``.
    """
    if df.empty:
        return df

    df = df.copy()
    df["rank"] = None
    df["composite_score"] = df.apply(compute_composite_score, axis=1)

    for (lang, task), group in df.groupby(["language", "task"]):
        _, direction = TASK_PRIMARY_METRICS.get(
            task, ("primary_score", "higher"),
        )
        ascending = direction == "lower"
        sorted_idx = group["primary_score"].sort_values(
            ascending=ascending,
        ).index
        for rank_pos, idx in enumerate(sorted_idx, 1):
            df.at[idx, "rank"] = rank_pos

    return df.sort_values(["task", "language", "rank"])


def build_leaderboard_table(
    entries: List[Dict[str, Any]],
    task_filter: Optional[str] = None,
    language_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Build a filtered and ranked leaderboard DataFrame.

    Args:
        entries: List of leaderboard entry dicts.
        task_filter: Filter to a specific task.
        language_filter: Filter to a specific language.

    Returns:
        Ranked DataFrame.
    """
    if not entries:
        return pd.DataFrame()

    df = pd.DataFrame(entries)
    if task_filter:
        df = df[df["task"] == task_filter]
    if language_filter:
        df = df[df["language"] == language_filter]

    if df.empty:
        return df

    return compute_rankings(df)

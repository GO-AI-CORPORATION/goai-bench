"""Rich console display helpers for benchmark results."""

from __future__ import annotations

from typing import Any, Dict


def print_result_table(task: str, result: Any, console: Any) -> None:
    """Print a single result metrics table."""
    from rich.table import Table

    t = Table(title="Results")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    for k, v in getattr(result, "all_metrics", {}).items():
        if v is not None:
            t.add_row(k.upper(), f"{v:.4f}")
    console.print(t)


def print_summary_table(
    task: str, all_results: Dict[str, Any], console: Any,
) -> None:
    """Print a summary table across all benchmark groups."""
    from rich.table import Table

    if task == "mt":
        t = Table(title="MT Benchmark Summary")
        t.add_column("Group", style="bold")
        t.add_column("Samples", justify="right")
        t.add_column("chrF++", justify="right")
        t.add_column("BLEU", justify="right")
        t.add_column("TER", justify="right")
        for grp, r in all_results.items():
            t.add_row(
                grp, str(r.n_samples),
                f"{r.overall_chrf:.1f}", f"{r.overall_bleu:.1f}",
                f"{r.overall_ter:.3f}",
            )
    elif task == "asr":
        t = Table(title="ASR Benchmark Summary")
        t.add_column("Group", style="bold")
        t.add_column("Samples", justify="right")
        t.add_column("WER", justify="right")
        t.add_column("CER", justify="right")
        for grp, r in all_results.items():
            t.add_row(
                grp, str(r.n_samples),
                f"{r.overall_wer:.3f}", f"{r.overall_cer:.3f}",
            )
    elif task == "tts":
        t = Table(title="TTS Benchmark Summary")
        t.add_column("Group", style="bold")
        t.add_column("Samples", justify="right")
        t.add_column("UTMOS", justify="right")
        t.add_column("LB-WER", justify="right")
        for grp, r in all_results.items():
            utmos = f"{r.overall_utmos:.2f}" if r.overall_utmos else "N/A"
            t.add_row(
                grp, str(r.n_samples),
                utmos, f"{r.overall_loopback_wer:.3f}",
            )
    else:
        return

    console.print(t)

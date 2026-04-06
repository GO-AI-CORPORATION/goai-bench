#!/usr/bin/env python3
"""Generate model comparison tables from existing benchmark results.

Reads all model result directories for a given language/task and produces
a comparison file (JSON + Markdown) ranking models by primary metric.

Usage::

    # Compare all MT models evaluated on Moore
    python scripts/compare_results.py --language mos_Latn --task mt

    # Compare all ASR models on Dioula
    python scripts/compare_results.py --language dyu_Latn --task asr

    # Compare all tasks for a language
    python scripts/compare_results.py --language mos_Latn --task all

    # Custom results directory
    python scripts/compare_results.py --language mos_Latn --task mt --results-dir my_results
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@click.command()
@click.option("--language", required=True, help="HF language code.")
@click.option("--task", required=True,
              type=click.Choice(["mt", "asr", "tts", "all"]),
              help="Task to compare, or 'all'.")
@click.option("--results-dir", default="results",
              help="Root results directory.")
def main(language: str, task: str, results_dir: str) -> None:
    """Generate comparison tables from benchmark results."""
    from rich.console import Console

    console = Console()
    console.print("\n[bold cyan]=== GO AI Bench — Model Comparison ===[/]\n")

    from goai_bench.core.result_writer import ResultWriter
    writer = ResultWriter()

    tasks = ["mt", "asr", "tts"] if task == "all" else [task]

    for t in tasks:
        check = Path(results_dir) / language / t
        if t == "mt":
            check = Path(results_dir) / language / "mt"
        if not check.exists():
            console.print(f"[dim]No results for {language}/{t}, skipping.[/]")
            continue

        console.print(f"[cyan]Generating comparison for {t.upper()} / {language}...[/]")
        md_path = writer.generate_comparison(results_dir, language, t)

        if md_path:
            console.print(f"  [green]Written:[/] {md_path}")
            json_path = md_path.replace(".md", ".json")
            console.print(f"  [green]Written:[/] {json_path}")

            md_content = Path(md_path).read_text(encoding="utf-8")
            console.print(f"\n{md_content}")
        else:
            console.print(f"  [yellow]No model results found.[/]")

    console.print()


if __name__ == "__main__":
    main()

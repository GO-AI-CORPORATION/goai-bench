#!/usr/bin/env python3
"""Evaluate all pre-configured baseline models across languages and tasks.

For each (language, task, model), runs all benchmark groups defined in
the YAML config, saving results in the clean hierarchical structure.
"""

import logging
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from goai_bench.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

BASELINES = {
    "mt": [
        "facebook/nllb-200-3.3B",
        "facebook/nllb-200-distilled-1.3B",
        "facebook/nllb-200-distilled-600M",
    ],
    "asr": [
        "openai/whisper-large-v3",
        "openai/whisper-medium",
        "facebook/mms-1b-all",
    ],
    "tts": {
        "mos_Latn": ["facebook/mms-tts-mos"],
        "dyu_Latn": ["facebook/mms-tts-dyu"],
    },
}

LANGUAGES = ["mos_Latn", "dyu_Latn"]


@click.command()
@click.option("--max-samples", default=None, type=int,
              help="Max samples per benchmark group.")
@click.option("--output-dir", default="results",
              help="Root output directory.")
@click.option("--no-comet", is_flag=True, help="Skip COMET (MT).")
@click.option("--verbose", is_flag=True)
def main(max_samples: int, output_dir: str, no_comet: bool, verbose: bool) -> None:
    """Run all baseline evaluations across benchmark groups."""
    setup_logging(verbose=verbose)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold cyan]=== Baseline Evaluation Sweep ===[/]\n")

    from goai_bench.core.config_loader import ConfigLoader
    from goai_bench.core.data_loader import DataLoader
    from goai_bench.core.result_writer import ResultWriter
    from goai_bench.providers.factory import (
        create_asr_provider,
        create_mt_provider,
        create_tts_provider,
    )

    cfg = ConfigLoader()
    data_loader = DataLoader()
    writer = ResultWriter()

    completed = []
    skipped = []

    for task, task_cfg in BASELINES.items():
        for lang in LANGUAGES:
            if isinstance(task_cfg, dict):
                models = task_cfg.get(lang, [])
            else:
                models = task_cfg
            for model_id in models:
                source_cfg = cfg.get_dataset_source(lang, task)
                if not source_cfg:
                    skipped.append((lang, task, model_id, "No config"))
                    continue

                groups = cfg.get_benchmark_groups(lang, task)
                if not groups:
                    skipped.append((lang, task, model_id, "No benchmark groups"))
                    continue

                console.print(
                    f"[cyan]Loading:[/] {model_id} | {task.upper()} | {lang}"
                )
                try:
                    if task == "mt":
                        provider = create_mt_provider(model_id)
                    elif task == "asr":
                        provider = create_asr_provider(model_id)
                    elif task == "tts":
                        provider = create_tts_provider(model_id)
                    else:
                        continue
                except Exception as exc:
                    skipped.append((lang, task, model_id, str(exc)[:80]))
                    console.print(f"  [red]Model load failed: {exc}[/]")
                    continue

                src_lang = source_cfg.get("source_lang", "fra_Latn")
                tgt_lang = source_cfg.get("target_lang", lang)

                group_results = {}
                for group_name in groups:
                    console.print(f"  [dim]Group: {group_name}[/]")
                    try:
                        data = data_loader.load_benchmark_group(
                            source_cfg, task, group_name, max_samples,
                        )
                        if not data:
                            skipped.append((
                                lang, task, model_id,
                                f"No data for group '{group_name}'",
                            ))
                            continue

                        from goai_bench.core.evaluator import run_evaluation

                        result = run_evaluation(
                            task=task,
                            provider=provider,
                            data=data,
                            language=lang,
                            source_lang=src_lang,
                            target_lang=tgt_lang,
                            compute_comet=not no_comet,
                        )

                        writer.save_result(
                            result, output_dir, group_name=group_name,
                            benchmark_language=lang,
                        )
                        writer.append_to_leaderboard(result)
                        group_results[group_name] = result
                        completed.append(
                            (lang, task, model_id, group_name),
                        )
                        console.print(f"    [green]Done[/]")

                    except Exception as exc:
                        skipped.append((
                            lang, task, model_id,
                            f"{group_name}: {str(exc)[:60]}",
                        ))
                        console.print(f"    [red]{exc}[/]")

                if group_results:
                    writer.save_summary(
                        task, model_id, lang, group_results, output_dir,
                    )

    console.print(f"\n[bold]Completed: {len(completed)}[/]")
    if completed:
        t = Table(title="Completed Evaluations")
        t.add_column("Language")
        t.add_column("Task")
        t.add_column("Model")
        t.add_column("Group")
        for lang, task, model_id, grp in completed:
            t.add_row(lang, task, model_id, grp)
        console.print(t)

    if skipped:
        console.print(f"\n[bold yellow]Skipped: {len(skipped)}[/]")
        t = Table(title="Skipped Evaluations")
        t.add_column("Language")
        t.add_column("Task")
        t.add_column("Model")
        t.add_column("Reason")
        for lang, task, model_id, reason in skipped:
            t.add_row(lang, task, model_id, reason)
        console.print(t)

    console.print("\n[cyan]Generating model comparisons...[/]")
    for lang in LANGUAGES:
        for task in BASELINES:
            comp = writer.generate_comparison(output_dir, lang, task)
            if comp:
                console.print(f"  [green]{comp}[/]")

    writer.export_markdown_leaderboard()


if __name__ == "__main__":
    main()

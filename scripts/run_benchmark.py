#!/usr/bin/env python3
"""Main CLI entrypoint for GO AI Bench.

Without ``--split`` or ``--group``, evaluates every **benchmark group** defined
in ``configs/datasets/<language>.yaml`` for the chosen ``--task``. The model is
loaded once and reused across groups.

For full CLI options, recipes, output layout, and companion scripts, see
``docs/en/benchmarking.md`` and ``docs/fr/benchmarking.md`` in the repository.
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from goai_bench import __version__
from goai_bench.utils.logging_utils import (
    get_library_versions,
    setup_logging,
)


@click.command()
@click.option("--task", required=True,
              type=click.Choice(["mt", "asr", "tts"]),
              help="Evaluation task.")
@click.option("--language", required=True, help="HF language code.")
@click.option("--model", required=True,
              help="HuggingFace model ID or local path.")
@click.option("--source-lang", default=None, help="Source language (MT only).")
@click.option("--target-lang", default=None, help="Target language (MT only).")
@click.option("--source-column", default=None,
              help="Override source text column name (MT).")
@click.option("--target-column", default=None,
              help="Override target text column name (MT).")
@click.option("--dataset", default=None,
              help="Explicit HuggingFace dataset: hf:<id>.")
@click.option("--split", default=None,
              help="Single HF split to evaluate (bypasses benchmark groups).")
@click.option("--group", default=None,
              help="Single benchmark group to evaluate (e.g. general, health).")
@click.option("--output", default="results", help="Root output directory.")
@click.option(
    "--batch-size",
    default=8,
    type=int,
    help="Batch size (MT, ASR GPU batching, TTS synthesis chunk size).",
)
@click.option("--max-samples", default=None, type=int,
              help="Limit evaluation samples per group.")
@click.option("--device", default="auto",
              type=click.Choice(["cpu", "cuda", "mps", "auto"]))
@click.option("--submit-to-leaderboard", is_flag=True,
              help="Append result to leaderboard.")
@click.option("--loopback-asr-model", default="openai/whisper-large-v3",
              help="ASR model for TTS loopback (TTS only).")
@click.option("--no-comet", is_flag=True, help="Skip COMET computation (MT).")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.version_option(version=__version__)
def main(
    task: str,
    language: str,
    model: str,
    source_lang: str,
    target_lang: str,
    source_column: str,
    target_column: str,
    dataset: str,
    split: str,
    group: str,
    output: str,
    batch_size: int,
    max_samples: int,
    device: str,
    submit_to_leaderboard: bool,
    loopback_asr_model: str,
    no_comet: bool,
    verbose: bool,
) -> None:
    """Run benchmark evaluation for an NLP model."""
    setup_logging(verbose=verbose)

    import numpy as np
    import torch
    from rich.console import Console

    console = Console()
    torch.manual_seed(42)
    np.random.seed(42)

    console.print("\n[bold cyan]=== GO AI Bench ===[/]\n")

    from goai_bench.core.config_loader import ConfigLoader
    from goai_bench.core.data_loader import DataLoader
    from goai_bench.core.evaluator import run_evaluation
    from goai_bench.core.result_writer import ResultWriter
    from goai_bench.providers.factory import (
        create_asr_provider,
        create_mt_provider,
        create_tts_provider,
    )
    from goai_bench.utils.display import print_result_table, print_summary_table

    cfg = ConfigLoader()
    loader = DataLoader()
    writer = ResultWriter()

    source_cfg = cfg.get_dataset_source(language, task)
    if source_cfg:
        if task == "mt" and source_column:
            source_cfg = {**source_cfg, "source_column": source_column}
        if task == "mt" and target_column:
            source_cfg = {**source_cfg, "target_column": target_column}
        if task == "mt" and source_lang:
            source_cfg = {**source_cfg, "source_lang": source_lang}
        if task == "mt" and target_lang:
            source_cfg = {**source_cfg, "target_lang": target_lang}

    src_lang = source_lang or (source_cfg or {}).get("source_lang", "fra_Latn")
    tgt_lang = target_lang or (source_cfg or {}).get("target_lang", language)

    console.print(f"[cyan]Loading model:[/] {model}")
    if task == "mt":
        provider = create_mt_provider(model, device)
    elif task == "asr":
        provider = create_asr_provider(model, device)
    elif task == "tts":
        provider = create_tts_provider(model, device)
    else:
        console.print(f"[red]Unknown task: {task}[/]")
        sys.exit(1)

    console.print("[green]Model loaded.[/]\n")

    eval_kwargs = dict(
        task=task, provider=provider, language=language,
        source_lang=src_lang, target_lang=tgt_lang,
        batch_size=batch_size, compute_comet=not no_comet,
        loopback_asr_model=loopback_asr_model,
    )

    if dataset:
        _run_single_dataset(
            task, language, source_cfg, dataset, split, src_lang, tgt_lang,
            source_column, target_column,
            max_samples, output, loader, writer, submit_to_leaderboard,
            console, eval_kwargs,
        )
    elif split:
        _run_single_split(
            language, split, source_cfg, max_samples, output,
            loader, writer, submit_to_leaderboard, console, eval_kwargs,
        )
    else:
        groups = cfg.get_benchmark_groups(language, task)
        if group:
            if group not in groups:
                console.print(
                    f"[red]Group '{group}' not found. "
                    f"Available: {list(groups.keys())}[/]"
                )
                sys.exit(1)
            groups = {group: groups[group]}

        if not groups:
            console.print(
                f"[red]No benchmark groups configured for {language}/{task}.[/]\n"
                f"[dim]Add benchmark_groups in configs/datasets/{language}.yaml[/]"
            )
            sys.exit(1)

        _run_benchmark_groups(
            task, language, groups, source_cfg, max_samples, output,
            loader, writer, submit_to_leaderboard, console, eval_kwargs,
        )

    if verbose:
        console.print("\n[dim]Library versions:[/]")
        for lib, ver in get_library_versions().items():
            console.print(f"  [dim]{lib}: {ver}[/]")


def _run_benchmark_groups(
    task, language, groups, source_cfg, max_samples, output,
    loader, writer, submit_to_leaderboard, console, eval_kwargs,
):
    """Evaluate across all benchmark groups."""
    from goai_bench.utils.display import print_summary_table

    console.print(
        f"[bold]Evaluating {len(groups)} benchmark group(s): "
        f"{list(groups.keys())}[/]\n"
    )

    all_results = {}
    for group_name, group_cfg in groups.items():
        splits = group_cfg.get("splits", [])
        desc = group_cfg.get("description", "")
        console.print(
            f"[cyan]--- Group: {group_name} ---[/]"
            f"  [dim]splits={splits}  {desc}[/]"
        )

        data = loader.load_benchmark_group(
            source_cfg, task, group_name, max_samples,
        )
        if not data:
            console.print("  [yellow]No data loaded, skipping.[/]\n")
            continue

        console.print(f"  [green]Loaded {len(data)} samples[/]")
        result = run_evaluation(data=data, **eval_kwargs)
        filepath = writer.save_result(
            result, output, group_name=group_name,
            benchmark_language=language,
        )
        console.print(f"  [green]Saved:[/] {filepath}")

        if submit_to_leaderboard:
            writer.append_to_leaderboard(result)

        all_results[group_name] = result
        console.print()

    if all_results:
        print_summary_table(task, all_results, console)

        model_id = next(iter(all_results.values())).model_id
        summary_path = writer.save_summary(
            task, model_id, language, all_results, output,
        )
        console.print(f"[green]Summary saved:[/] {summary_path}")

        mt_pair = None
        if task == "mt":
            r0 = next(iter(all_results.values()))
            mt_pair = f"{r0.source_lang}_{r0.target_lang}"
        comp_path = writer.generate_comparison(
            output, language, task, mt_pair=mt_pair,
        )
        if comp_path:
            console.print(f"[green]Comparison updated:[/] {comp_path}")

    if submit_to_leaderboard:
        writer.export_markdown_leaderboard()
        console.print("[green]Leaderboard updated.[/]")


def _run_single_split(
    language, split, source_cfg, max_samples, output,
    loader, writer, submit_to_leaderboard, console, eval_kwargs,
):
    """Evaluate a single explicit split."""
    from goai_bench.core.exceptions import EmptyDatasetError
    from goai_bench.utils.display import print_result_table

    if not source_cfg:
        console.print(f"[red]No dataset config for {language}/{eval_kwargs['task']}.[/]")
        sys.exit(1)

    console.print(f"[cyan]Loading split:[/] {split}")
    data = loader.load_from_config(
        source_cfg, eval_kwargs["task"], split=split, max_samples=max_samples,
    )
    if not data:
        raise EmptyDatasetError(language, eval_kwargs["task"], split)

    console.print(f"[green]Loaded {len(data)} samples[/]\n")
    result = run_evaluation(data=data, **eval_kwargs)
    filepath = writer.save_result(
        result, output, group_name=split, benchmark_language=language,
    )
    console.print(f"\n[green]Saved:[/] {filepath}")
    print_result_table(eval_kwargs["task"], result, console)

    if submit_to_leaderboard:
        writer.append_to_leaderboard(result)
        writer.export_markdown_leaderboard()


def _run_single_dataset(
    task, language, source_cfg, dataset, split, src_lang, tgt_lang,
    source_column, target_column,
    max_samples, output, loader, writer, submit_to_leaderboard,
    console, eval_kwargs,
):
    """Evaluate with an explicit HuggingFace dataset (``hf:<id>``).

    Merges the CLI dataset id with the language/task YAML source so ASR/TTS
    keep ``audio_column`` / ``text_column`` (and MT keeps domain columns).
    """
    from goai_bench.core.exceptions import EmptyDatasetError
    from goai_bench.utils.display import print_result_table

    hf_id = dataset.replace("hf:", "") if dataset.startswith("hf:") else dataset
    console.print(f"[cyan]Loading data from:[/] {hf_id}")

    resolved_split = split or "test"

    if task == "mt":
        if source_cfg:
            merged = {**source_cfg, "hf_dataset": hf_id}
            if source_column:
                merged["source_column"] = source_column
            if target_column:
                merged["target_column"] = target_column
        else:
            merged = {
                "hf_dataset": hf_id,
                "source_column": source_column or f"sentence_{src_lang}",
                "target_column": target_column or f"sentence_{tgt_lang}",
                "source_lang": src_lang,
                "target_lang": tgt_lang,
            }
    elif task in ("asr", "tts"):
        if not source_cfg:
            console.print(
                f"[red]No dataset config for {language}/{task}.[/]\n"
                f"[dim]Define the task in configs/datasets/{language}.yaml "
                f"(column names), or omit --dataset to use that config only.[/]"
            )
            sys.exit(1)
        merged = {**source_cfg, "hf_dataset": hf_id}
    else:
        merged = {"hf_dataset": hf_id}

    data = loader.load_from_config(
        merged, task, split=resolved_split, max_samples=max_samples,
    )

    if not data:
        raise EmptyDatasetError(
            eval_kwargs.get("language", ""), eval_kwargs["task"], "",
        )
    console.print(f"[green]Loaded {len(data)} samples[/]\n")

    result = run_evaluation(data=data, **eval_kwargs)
    filepath = writer.save_result(
        result, output, benchmark_language=eval_kwargs.get("language"),
    )
    console.print(f"\n[green]Saved:[/] {filepath}")
    print_result_table(eval_kwargs["task"], result, console)

    if submit_to_leaderboard:
        writer.append_to_leaderboard(result)
        writer.export_markdown_leaderboard()


if __name__ == "__main__":
    main()

# Running benchmarks

This page describes how to run evaluations with **GO AI Bench** after `pip install -e .` from the repository root.

> [!IMPORTANT]
> Set `HF_TOKEN` if any dataset or model is gated. See the [main README](../../README.md).

## Default behavior

If you **omit** `--split` and `--dataset`, the CLI evaluates **every benchmark group** defined in `configs/datasets/<language>.yaml` for the chosen `--task`. The model is loaded **once** and reused for all groups.

## Modes of operation

| Mode | Flags | Behavior |
|------|-------|----------|
| All groups | (none of the below) | Loops groups from YAML; writes one JSON per group + `summary.json` + comparison files. |
| Single group | `--group health` | Same as above but only that group. |
| Single split | `--split test` | Ignores benchmark groups; loads one HF split via YAML column config. |
| Explicit dataset | `--dataset hf:org/name` | Overrides `hf_dataset` while keeping YAML **column** settings (MT/ASR/TTS). Requires YAML for ASR/TTS. |

## Output layout

Under `--output` (default `results/`):

- `results/<language>/<task>/...` for ASR/TTS.
- MT adds a direction folder: `results/<language>/mt/<source>_<target>/<model_slug>/`.

Files include per-group JSON, `summary.json`, and `comparison.json` / `comparison.md` when applicable.

## Full CLI reference (`run_benchmark.py`)

| Option | Description |
|--------|-------------|
| `--task` | Required. `mt`, `asr`, or `tts`. |
| `--language` | Required. HF language code (e.g. `mos_Latn`, `dyu_Latn`). Must match `configs/datasets/<language>.yaml`. |
| `--model` | Required. Hugging Face model id or local path understood by the factory. |
| `--source-lang` | MT: source NLLB-style code (default from YAML). |
| `--target-lang` | MT: target code (default from YAML or `--language`). |
| `--source-column` | MT: override source text column. |
| `--target-column` | MT: override target text column. |
| `--dataset` | Optional. `hf:<dataset_id>` to swap Hub dataset id (see [datasets.md](datasets.md)). |
| `--split` | Optional. Single HF split name; bypasses benchmark groups. |
| `--group` | Optional. Single benchmark group name. |
| `--output` | Root results directory (default `results`). |
| `--batch-size` | MT batching, ASR pipeline batching, TTS synthesis chunk size. |
| `--max-samples` | Cap samples per group or per single-split run. |
| `--device` | `cpu`, `cuda`, `mps`, or `auto`. |
| `--submit-to-leaderboard` | Append to `leaderboard/leaderboard.json` and refresh markdown export. |
| `--loopback-asr-model` | TTS: ASR model id for loopback WER (default Whisper large v3). |
| `--no-comet` | MT: skip COMET (faster, fewer deps). |
| `--verbose` | Debug logging and library versions. |
| `--version` | Print package version. |

Invocation:

```bash
python scripts/run_benchmark.py [OPTIONS]
# or, after install:
goai-bench [OPTIONS]
```

## Recipe: MT all groups (French → Mooré)

```bash
python scripts/run_benchmark.py \
  --task mt \
  --language mos_Latn \
  --model facebook/nllb-200-distilled-600M \
  --no-comet \
  --verbose
```

## Recipe: MT one domain

```bash
python scripts/run_benchmark.py \
  --task mt --language mos_Latn \
  --model facebook/nllb-200-distilled-600M \
  --group health \
  --no-comet
```

## Recipe: MT reverse direction (Mooré → French)

```bash
python scripts/run_benchmark.py \
  --task mt --language mos_Latn \
  --model facebook/nllb-200-distilled-600M \
  --source-lang mos_Latn --target-lang fra_Latn \
  --source-column sentence_mos_Latn \
  --target-column sentence_fra_Latn \
  --no-comet
```

## Recipe: ASR (all groups)

```bash
python scripts/run_benchmark.py \
  --task asr --language mos_Latn \
  --model openai/whisper-large-v3 \
  --verbose
```

## Recipe: TTS (all groups)

```bash
python scripts/run_benchmark.py \
  --task tts --language mos_Latn \
  --model facebook/mms-tts-mos \
  --verbose
```

## Recipe: explicit split (no groups)

```bash
python scripts/run_benchmark.py \
  --task mt --language mos_Latn \
  --model facebook/nllb-200-distilled-600M \
  --split test \
  --no-comet
```

## Recipe: explicit Hub dataset (same columns as YAML)

```bash
python scripts/run_benchmark.py \
  --task asr --language mos_Latn \
  --model openai/whisper-large-v3 \
  --dataset hf:your-org/your-asr-dataset \
  --split test \
  --max-samples 100
```

## `compare_results.py`

Regenerates ranking tables from **existing** JSON under `results/`:

```bash
python scripts/compare_results.py --language mos_Latn --task mt
python scripts/compare_results.py --language dyu_Latn --task asr
python scripts/compare_results.py --language mos_Latn --task all
```

Options: `--results-dir` to point to a non-default root.

## `run_baselines.py`

Runs a **hard-coded** list of baseline models across `mos_Latn` and `dyu_Latn`. Useful for internal sweeps; see script source for the exact model list.

> [!NOTE]
> Aligning this script with `configs/tasks.yaml` `baseline_models` is a planned improvement (see [architecture.md](architecture.md)).

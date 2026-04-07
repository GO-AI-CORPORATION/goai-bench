# Dataset configuration

Benchmark data is defined in **`configs/datasets/<language>.yaml`**. Each task (`mt`, `asr`, `tts`) has one or more **sources**; the first source is used by the CLI unless you extend the loader.

> [!IMPORTANT]
> Column names in your YAML **must match** the Hugging Face dataset schema. If they differ, either rename columns in the dataset card or map them when building the Hub dataset.

## Structure of a source block

Common keys:

| Key | Used by | Description |
|-----|---------|-------------|
| `id` | Human | Stable identifier for the source (documentation only). |
| `hf_dataset` | All | Hugging Face dataset repo id, e.g. `goaicorp/MooreMTBench`. |
| `hf_config` | All | Optional dataset config (subset) name, e.g. `flores`. |
| `hf_revision` | All | Optional tag, branch or commit to pin, e.g. `v1.0`. |
| `benchmark_groups` | All | Named groups; each lists `splits` and optional `description`. A group with `disabled: true` is skipped with a warning. |
| `source_column`, `target_column` | MT | Text columns for source and reference. |
| `source_lang`, `target_lang` | MT | NLLB-style codes (e.g. `fra_Latn`, `mos_Latn`). |
| `language_columns` | MT | Optional map from language code to column (`fra_Latn: fra`), so `--source-lang` / `--target-lang` can flip the direction. |
| `domain_column` | MT, TTS | Optional column with domain labels (`health`, …). |
| `audio_column`, `text_column` | ASR | Audio feature column (dict with `array` / `sampling_rate`) and transcript. |
| `text_column` | TTS | Column used as synthesis prompt. |

Optional split defaults: `hf_split` or first entry of `hf_splits` (see `DataLoader._resolve_split`).

## Default benchmark datasets (with Hub links)

> [!IMPORTANT]
> MooreMTBench and DioulaMTBench are gated with manual approval. Request access on each dataset page, then set `HF_TOKEN` to a token of an account that has access. Without it, MT and TTS load no data.

### Mooré (`mos_Latn`)

| Task | Dataset | Hub |
|------|---------|-----|
| MT | `goaicorp/MooreMTBench` (config `flores`, tag `v1.0`) | [Dataset card](https://huggingface.co/datasets/goaicorp/MooreMTBench) |
| ASR | `goaicorp/GOAI-MooreSpeechCorpora` | [Dataset card](https://huggingface.co/datasets/goaicorp/GOAI-MooreSpeechCorpora) |
| TTS (prompts) | `goaicorp/MooreMTBench` | Same as MT (uses `mos` as prompts) |

**Benchmark groups** (MT/TTS): `general` = FLORES+ splits `dev` (997) and `devtest` (1,012). The `health`, `education`, `child_protection` and `environment` groups are disabled for now: their data is being fixed. **ASR**: `general` on split `test`.

### Dioula (`dyu_Latn`)

| Task | Dataset | Hub |
|------|---------|-----|
| MT | `goaicorp/DioulaMTBench` (config `flores`, tag `v1.0`) | [Dataset card](https://huggingface.co/datasets/goaicorp/DioulaMTBench) |
| ASR | `goaicorp/goai-dioula-speech` | [Dataset card](https://huggingface.co/datasets/goaicorp/goai-dioula-speech) |
| TTS | `goaicorp/DioulaMTBench` | Same as MT (`dyu` for prompts) |

**Benchmark groups** (MT/TTS): `general` = FLORES+ `dev` and `devtest`, as for Mooré; the four domain groups are disabled for now. **ASR** default group uses split `dev` (see `configs/datasets/dyu_Latn.yaml`).

## Overriding the dataset from the CLI (`--dataset`)

You can pass `--dataset hf:<dataset_id>` to evaluate another Hub dataset **while reusing column names from the YAML** for that language and task. This keeps ASR/TTS column mappings correct.

> [!WARNING]
> If there is **no** YAML entry for the language/task, `--dataset` with **ASR** or **TTS** will exit with an error. For **MT**, a minimal config can be built from `--source-lang` / `--target-lang` and default `sentence_*` columns.

## Best practices before publishing on Hugging Face

1. **Schema**: Use clear column names (`audio` + `text` for ASR; parallel sentences for MT).
2. **Splits**: Use conventional names (`train`, `validation`, `test`) where possible; domain splits (`health`, …) are supported if they exist as split names.
3. **Audio**: Prefer the `datasets.Audio` feature so rows expose `{"array": ..., "sampling_rate": ...}`.
4. **License and card**: Document language, collection method, and known biases in the dataset README.
5. **Gated data**: If the dataset is private or gated, document that evaluators need `HF_TOKEN` (see main README).
6. **Versioning**: Pin a dataset revision with `hf_revision` in YAML for reproducible benchmarks (the default configs pin `v1.0`).

> [!TIP]
> After pushing to the Hub, run a smoke test: `load_dataset("org/name", split="test")` in a notebook, then a single `run_benchmark.py` command with `--max-samples 2`.

# Dataset configuration

Benchmark data is defined in **`configs/datasets/<language>.yaml`**. Each task (`mt`, `asr`, `tts`) has one or more **sources**; the first source is used by the CLI unless you extend the loader.

> [!IMPORTANT]
> Column names in your YAML **must match** the Hugging Face dataset schema. If they differ, either rename columns in the dataset card or map them when building the Hub dataset.

## Structure of a source block

Common keys:

| Key | Used by | Description |
|-----|---------|-------------|
| `id` | Human | Stable identifier for the source (documentation only). |
| `hf_dataset` | All | Hugging Face dataset repo id, e.g. `goaicorp/flores-plus-fra-mos`. |
| `benchmark_groups` | All | Named groups; each lists `splits` and optional `description`. |
| `source_column`, `target_column` | MT | Text columns for source and reference. |
| `source_lang`, `target_lang` | MT | NLLB-style codes (e.g. `fra_Latn`, `mos_Latn`). |
| `domain_column` | MT, TTS | Optional column with domain labels (`health`, …). |
| `audio_column`, `text_column` | ASR | Audio feature column (dict with `array` / `sampling_rate`) and transcript. |
| `text_column` | TTS | Column used as synthesis prompt. |

Optional split defaults: `hf_split` or first entry of `hf_splits` (see `DataLoader._resolve_split`).

## Default benchmark datasets (with Hub links)

### Mooré (`mos_Latn`)

| Task | Dataset | Hub |
|------|---------|-----|
| MT | `goaicorp/flores-plus-fra-mos` | [Dataset card](https://huggingface.co/datasets/goaicorp/flores-plus-fra-mos) |
| ASR | `goaicorp/GOAI-MooreSpeechCorpora` | [Dataset card](https://huggingface.co/datasets/goaicorp/GOAI-MooreSpeechCorpora) |
| TTS (prompts) | `goaicorp/flores-plus-fra-mos` | Same as MT (uses `sentence_mos_Latn` as prompts) |

**Benchmark groups** (MT/TTS): `general` (splits `test`, `devtest`), `health`, `education`, `child_protection`, `environment`. **ASR**: `general` on split `test`.

### Dioula (`dyu_Latn`)

| Task | Dataset | Hub |
|------|---------|-----|
| MT | `goaicorp/flores-plus-fra-dyu` | [Dataset card](https://huggingface.co/datasets/goaicorp/flores-plus-fra-dyu) |
| ASR | `goaicorp/goai-dioula-speech` | [Dataset card](https://huggingface.co/datasets/goaicorp/goai-dioula-speech) |
| TTS | `goaicorp/flores-plus-fra-dyu` | Same as MT (`sentence_dyu_Latn` for prompts) |

**ASR** default group uses split `dev` (see `configs/datasets/dyu_Latn.yaml`).

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
6. **Versioning**: Pin a dataset revision in YAML if you need reproducible benchmarks (`datasets` supports `revision=`; extending GO AI Bench to read a `revision` field from YAML is a possible enhancement).

> [!TIP]
> After pushing to the Hub, run a smoke test: `load_dataset("org/name", split="test")` in a notebook, then a single `run_benchmark.py` command with `--max-samples 2`.

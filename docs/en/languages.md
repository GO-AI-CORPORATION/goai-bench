# Adding a new language

To register a new benchmark language you touch **metadata**, **dataset config**, and optionally **tasks** defaults.

> [!NOTE]
> GO AI Bench uses **Hugging Face style** language tags such as `mos_Latn` (ISO 639-3 + script). Align with NLLB / FLORES conventions when you work with MT.

## 1. Register metadata — `configs/languages.yaml`

Add an entry under `languages:` with at least:

- `name`, `iso_639_3`, `hf_code`, `script`, `family`, `region`, `resource_level`

Optional fields (examples in the file): `flores_code`, `nllb_code`, `notes`.

## 2. Create dataset config — `configs/datasets/<hf_code>.yaml`

Copy `mos_Latn.yaml` or `dyu_Latn.yaml` as a template and adjust:

- Top-level `language:` to your HF code.
- `hf_dataset` ids for `mt`, `asr`, `tts`.
- Column names and `source_lang` / `target_lang` for MT.
- `benchmark_groups` and split names that exist on your Hub datasets.

> [!IMPORTANT]
> The CLI `--language` argument must match the **filename** (without `.yaml`), e.g. `gur_Latn` for `configs/datasets/gur_Latn.yaml`.

## 3. Optional — `configs/tasks.yaml`

`tasks.yaml` describes task-level metadata and suggested `baseline_models`. Update lists if you want documentation and baseline sweeps to mention models relevant to the new language.

## 4. Provider-side language lists

- **NLLB MT**: Unknown language codes log a **warning** but still run; extend `_NLLB_CODES` in `providers/mt/hf_seq2seq.py` if you want stricter checks for your deployment.
- **Whisper ASR**: Low-resource ISO codes may fall back to automatic language detection (see `utils/text_utils.py`).

## 5. Verify

```bash
python -m pytest tests/test_config_loader.py -v
python scripts/run_benchmark.py --task mt --language <your_code> --model facebook/nllb-200-distilled-600M --group general --max-samples 2 --no-comet
```

> [!CAUTION]
> If your dataset splits differ from the template (e.g. no `health` split), remove or rename benchmark groups in YAML; otherwise the loader will return empty groups.

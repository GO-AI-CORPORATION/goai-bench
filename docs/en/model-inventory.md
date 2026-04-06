# Model inventory (Mooré and Dioula)

This inventory lists **open or public Hub models** referenced in GO AI Corporation’s benchmarking artifacts (`benchmarking-report/results_final.json` and the LaTeX report). It is **not** an exhaustive list of all models that exist for these languages.

> [!NOTE]
> Some entries are **fine-tuned** by GO AI (`goaicorp/...`); others are **off-the-shelf** checkpoints evaluated in zero-shot or transfer settings. WER/chrF numbers are in the report files, not duplicated here.

> [!WARNING]
> The JSON may list the same architecture under multiple benchmark languages (e.g. a model name containing `mos` may appear in a Dioula row if reused in an experiment). Always verify the **Hub card** and license before production use.

## Mooré (`mos_Latn`)

### Machine translation

| Model | Link | Notes |
|-------|------|-------|
| `facebook/nllb-200-distilled-600M` | [Hub](https://huggingface.co/facebook/nllb-200-distilled-600M) | Baseline NLLB distilled |
| `facebook/nllb-200-1.3B` | [Hub](https://huggingface.co/facebook/nllb-200-1.3B) | Larger NLLB |
| `facebook/nllb-200-3.3B` | [Hub](https://huggingface.co/facebook/nllb-200-3.3B) | Largest baseline in report |
| `goaicorp/mos-nllb-600M-fr2mos` | [Hub](https://huggingface.co/goaicorp/mos-nllb-600M-fr2mos) | Fine-tuned (FR→Mooré) |
| `goaicorp/mos-nllb-600M-mos2fr` | [Hub](https://huggingface.co/goaicorp/mos-nllb-600M-mos2fr) | Fine-tuned (Mooré→FR) |
| `goaicorp/mos-nllb-3.3B-fr2mos` | [Hub](https://huggingface.co/goaicorp/mos-nllb-3.3B-fr2mos) | Fine-tuned (FR→Mooré) |
| `goaicorp/mos-nllb-3.3B-mos2fr` | [Hub](https://huggingface.co/goaicorp/mos-nllb-3.3B-mos2fr) | Fine-tuned (Mooré→FR) |

### Automatic speech recognition

| Model | Link | Notes |
|-------|------|-------|
| `openai/whisper-base` | [Hub](https://huggingface.co/openai/whisper-base) | Small Whisper |
| `openai/whisper-small` | [Hub](https://huggingface.co/openai/whisper-small) | Medium-small Whisper |
| `openai/whisper-large-v3-turbo` | [Hub](https://huggingface.co/openai/whisper-large-v3-turbo) | Turbo variant |
| `facebook/mms-1b-all` | [Hub](https://huggingface.co/facebook/mms-1b-all) | MMS ASR (multilingual) |
| `goaicorp/mos-whisper-small` | [Hub](https://huggingface.co/goaicorp/mos-whisper-small) | Fine-tuned Whisper-class |
| `goaicorp/mos-whisper-large-v3-turbo` | [Hub](https://huggingface.co/goaicorp/mos-whisper-large-v3-turbo) | Fine-tuned |
| `goaicorp/mos-wav2vec2-large-mms-1b` | [Hub](https://huggingface.co/goaicorp/mos-wav2vec2-large-mms-1b) | Wav2Vec2 / MMS family |

### Text-to-speech

| Model | Link | Notes |
|-------|------|-------|
| `facebook/mms-tts-mos` | [Hub](https://huggingface.co/facebook/mms-tts-mos) | Meta MMS-TTS for Mooré |
| `goaicorp/mos-mms-tts-v2` | [Hub](https://huggingface.co/goaicorp/mos-mms-tts-v2) | GO AI TTS checkpoint |
| `goaicorp/Spark-TTS-mos` | [Hub](https://huggingface.co/goaicorp/Spark-TTS-mos) | Appears in report JSON as evaluated TTS |

## Dioula (`dyu_Latn`)

### Machine translation

| Model | Link | Notes |
|-------|------|-------|
| `facebook/nllb-200-distilled-600M` | [Hub](https://huggingface.co/facebook/nllb-200-distilled-600M) | Baseline |
| `facebook/nllb-200-1.3B` | [Hub](https://huggingface.co/facebook/nllb-200-1.3B) | Larger baseline |

### Automatic speech recognition

| Model | Link | Notes |
|-------|------|-------|
| `openai/whisper-base` | [Hub](https://huggingface.co/openai/whisper-base) | |
| `openai/whisper-large-v3-turbo` | [Hub](https://huggingface.co/openai/whisper-large-v3-turbo) | |
| `facebook/mms-1b-all` | [Hub](https://huggingface.co/facebook/mms-1b-all) | |
| `goaicorp/mos-whisper-small` | [Hub](https://huggingface.co/goaicorp/mos-whisper-small) | Listed in `results_final.json` for `dyu_Latn`; verify intended pairing |

### Text-to-speech

No separate TTS rows in the aggregated `results_final.json` excerpt used for this table; use **`facebook/mms-tts-dyu`** as the standard open MMS-TTS checkpoint for Dioula: [Hub](https://huggingface.co/facebook/mms-tts-dyu).

> [!TIP]
> For full metric tables and methodology, read `benchmarking-report/report_final.tex` and `benchmarking-report/RESULTS.md` in the repository.

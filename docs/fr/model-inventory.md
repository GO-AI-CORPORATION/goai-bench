# Inventaire des modèles (Mooré et Dioula)

Cet inventaire recense des modèles **ouverts ou publics sur le Hub** mentionnés dans les artefacts de benchmark de GO AI Corporation (`benchmarking-report/results_final.json` et le rapport LaTeX). Ce n’est **pas** une liste exhaustive de tous les modèles possibles pour ces langues.

> [!NOTE]
> Certaines entrées sont **fine-tunées** par GO AI (`goaicorp/...`) ; d’autres sont des checkpoints **génériques** évalués en zéro-shot ou transfert. Les WER/chrF sont dans les fichiers de rapport, non recopiés ici.

> [!WARNING]
> Le JSON peut lister la même architecture sous plusieurs langues de benchmark (ex. un nom contenant `mos` peut apparaître pour une ligne Dioula si le modèle a été réutilisé). Vérifiez toujours la **fiche Hub** et la licence avant un usage en production.

## Mooré (`mos_Latn`)

### Traduction automatique

| Modèle | Lien | Notes |
|--------|------|-------|
| `facebook/nllb-200-distilled-600M` | [Hub](https://huggingface.co/facebook/nllb-200-distilled-600M) | Baseline NLLB distillé |
| `facebook/nllb-200-1.3B` | [Hub](https://huggingface.co/facebook/nllb-200-1.3B) | NLLB plus grand |
| `facebook/nllb-200-3.3B` | [Hub](https://huggingface.co/facebook/nllb-200-3.3B) | Plus grosse baseline du rapport |
| `goaicorp/mos-nllb-600M-fr2mos` | [Hub](https://huggingface.co/goaicorp/mos-nllb-600M-fr2mos) | Fine-tuné (FR→Mooré) |
| `goaicorp/mos-nllb-600M-mos2fr` | [Hub](https://huggingface.co/goaicorp/mos-nllb-600M-mos2fr) | Fine-tuné (Mooré→FR) |
| `goaicorp/mos-nllb-3.3B-fr2mos` | [Hub](https://huggingface.co/goaicorp/mos-nllb-3.3B-fr2mos) | Fine-tuné (FR→Mooré) |
| `goaicorp/mos-nllb-3.3B-mos2fr` | [Hub](https://huggingface.co/goaicorp/mos-nllb-3.3B-mos2fr) | Fine-tuné (Mooré→FR) |

### Reconnaissance vocale automatique

| Modèle | Lien | Notes |
|--------|------|-------|
| `openai/whisper-base` | [Hub](https://huggingface.co/openai/whisper-base) | Petit Whisper |
| `openai/whisper-small` | [Hub](https://huggingface.co/openai/whisper-small) | Whisper moyen-petit |
| `openai/whisper-large-v3-turbo` | [Hub](https://huggingface.co/openai/whisper-large-v3-turbo) | Variante turbo |
| `facebook/mms-1b-all` | [Hub](https://huggingface.co/facebook/mms-1b-all) | MMS ASR multilingue |
| `goaicorp/mos-whisper-small` | [Hub](https://huggingface.co/goaicorp/mos-whisper-small) | Whisper fine-tuné |
| `goaicorp/mos-whisper-large-v3-turbo` | [Hub](https://huggingface.co/goaicorp/mos-whisper-large-v3-turbo) | Fine-tuné |
| `goaicorp/mos-wav2vec2-large-mms-1b` | [Hub](https://huggingface.co/goaicorp/mos-wav2vec2-large-mms-1b) | Famille Wav2Vec2 / MMS |

### Synthèse vocale

| Modèle | Lien | Notes |
|--------|------|-------|
| `facebook/mms-tts-mos` | [Hub](https://huggingface.co/facebook/mms-tts-mos) | MMS-TTS Meta pour le mooré |
| `goaicorp/mos-mms-tts-v2` | [Hub](https://huggingface.co/goaicorp/mos-mms-tts-v2) | Checkpoint TTS GO AI |
| `goaicorp/Spark-TTS-mos` | [Hub](https://huggingface.co/goaicorp/Spark-TTS-mos) | Présent dans le JSON du rapport comme TTS évalué |

## Dioula (`dyu_Latn`)

### Traduction automatique

| Modèle | Lien | Notes |
|--------|------|-------|
| `facebook/nllb-200-distilled-600M` | [Hub](https://huggingface.co/facebook/nllb-200-distilled-600M) | Baseline |
| `facebook/nllb-200-1.3B` | [Hub](https://huggingface.co/facebook/nllb-200-1.3B) | Baseline plus grande |

### Reconnaissance vocale automatique

| Modèle | Lien | Notes |
|--------|------|-------|
| `openai/whisper-base` | [Hub](https://huggingface.co/openai/whisper-base) | |
| `openai/whisper-large-v3-turbo` | [Hub](https://huggingface.co/openai/whisper-large-v3-turbo) | |
| `facebook/mms-1b-all` | [Hub](https://huggingface.co/facebook/mms-1b-all) | |
| `goaicorp/mos-whisper-small` | [Hub](https://huggingface.co/goaicorp/mos-whisper-small) | Listé dans `results_final.json` pour `dyu_Latn` ; vérifier l’appariement prévu |

### Synthèse vocale

Pas de lignes TTS distinctes dans l’extrait agrégé de `results_final.json` utilisé pour ce tableau ; le checkpoint MMS-TTS ouvert standard pour le dioula est **`facebook/mms-tts-dyu`** : [Hub](https://huggingface.co/facebook/mms-tts-dyu).

> [!TIP]
> Pour les tableaux de métriques complets et la méthodologie, consulter `benchmarking-report/report_final.tex` et `benchmarking-report/RESULTS.md` dans le dépôt.

# Configuration des jeux de données

Les benchmarks sont décrits dans **`configs/datasets/<langue>.yaml`**. Chaque tâche (`mt`, `asr`, `tts`) contient des **sources** ; la première est utilisée par défaut par la CLI.

> [!IMPORTANT]
> Les noms de colonnes du YAML **doivent correspondre** au schéma du dataset Hugging Face.

## Clés usuelles d’une source

| Clé | Tâche | Description |
|-----|-------|-------------|
| `id` | — | Identifiant lisible (documentation). |
| `hf_dataset` | Toutes | Id Hub, ex. `goaicorp/flores-plus-fra-mos`. |
| `benchmark_groups` | Toutes | Groupes nommés ; chaque groupe liste des `splits`. |
| `source_column`, `target_column` | MT | Colonnes texte parallèles. |
| `source_lang`, `target_lang` | MT | Codes style NLLB (`fra_Latn`, `mos_Latn`, …). |
| `domain_column` | MT, TTS | Colonne de domaine (`health`, …). |
| `audio_column`, `text_column` | ASR | Audio (dict `array` / `sampling_rate`) et transcription. |
| `text_column` | TTS | Texte lu par le modèle TTS. |

## Jeux par défaut (liens Hub)

### Mooré (`mos_Latn`)

| Tâche | Dataset | Fiche |
|-------|---------|--------|
| MT | `goaicorp/flores-plus-fra-mos` | [Hugging Face](https://huggingface.co/datasets/goaicorp/flores-plus-fra-mos) |
| ASR | `goaicorp/GOAI-MooreSpeechCorpora` | [Hugging Face](https://huggingface.co/datasets/goaicorp/GOAI-MooreSpeechCorpora) |
| TTS | `goaicorp/flores-plus-fra-mos` | Idem MT (`sentence_mos_Latn`) |

### Dioula (`dyu_Latn`)

| Tâche | Dataset | Fiche |
|-------|---------|--------|
| MT | `goaicorp/flores-plus-fra-dyu` | [Hugging Face](https://huggingface.co/datasets/goaicorp/flores-plus-fra-dyu) |
| ASR | `goaicorp/goai-dioula-speech` | [Hugging Face](https://huggingface.co/datasets/goaicorp/goai-dioula-speech) |
| TTS | `goaicorp/flores-plus-fra-dyu` | Idem MT (`sentence_dyu_Latn`) |

## Option CLI `--dataset`

`--dataset hf:<id>` remplace uniquement **`hf_dataset`** en conservant les colonnes du YAML pour la langue et la tâche.

> [!WARNING]
> Sans YAML pour la paire langue/tâche, **ASR** et **TTS** avec `--dataset` sont refusés. Le **MT** peut utiliser une config minimale dérivée des options de langue.

## Bonnes pratiques avant publication sur le Hub

1. Schéma clair (`audio` / `text` pour l’ASR).
2. Splits cohérents (`train`, `test`, ou splits par domaine).
3. Audio au format attendu par `datasets` (feature `Audio`).
4. Carte de dataset : licence, langue, limites.
5. Jeux privés : documenter le besoin de `HF_TOKEN`.
6. Reproductibilité : envisager d’épingler une révision Git du dataset (extension possible du loader).

> [!TIP]
> Après le push, tester `load_dataset` puis un run avec `--max-samples 2`.

# Configuration des jeux de données

Les benchmarks sont décrits dans **`configs/datasets/<langue>.yaml`**. Chaque tâche (`mt`, `asr`, `tts`) contient des **sources** ; la première est utilisée par défaut par la CLI.

> [!IMPORTANT]
> Les noms de colonnes du YAML **doivent correspondre** au schéma du dataset Hugging Face.

## Clés usuelles d’une source

| Clé | Tâche | Description |
|-----|-------|-------------|
| `id` | — | Identifiant lisible (documentation). |
| `hf_dataset` | Toutes | Id Hub, ex. `goaicorp/MooreMTBench`. |
| `hf_config` | Toutes | Nom de configuration (sous-ensemble) du dataset, ex. `flores`. |
| `hf_revision` | Toutes | Tag, branche ou commit à épingler, ex. `v1.0`. |
| `benchmark_groups` | Toutes | Groupes nommés ; chaque groupe liste des `splits`. Un groupe avec `disabled: true` est ignoré avec un avertissement. |
| `source_column`, `target_column` | MT | Colonnes texte parallèles. |
| `source_lang`, `target_lang` | MT | Codes style NLLB (`fra_Latn`, `mos_Latn`, …). |
| `language_columns` | MT | Correspondance code de langue → colonne (`fra_Latn: fra`), pour inverser la direction avec `--source-lang` / `--target-lang`. |
| `domain_column` | MT, TTS | Colonne de domaine (`health`, …). |
| `audio_column`, `text_column` | ASR | Audio (dict `array` / `sampling_rate`) et transcription. |
| `text_column` | TTS | Texte lu par le modèle TTS. |

## Jeux par défaut (liens Hub)

> [!IMPORTANT]
> MooreMTBench et DioulaMTBench sont à accès restreint, avec validation manuelle. Demandez l'accès sur la page de chaque dataset, puis définissez `HF_TOKEN` avec le jeton d'un compte qui y a accès. Sans cela, le MT et le TTS ne chargent aucune donnée.

### Mooré (`mos_Latn`)

| Tâche | Dataset | Fiche |
|-------|---------|--------|
| MT | `goaicorp/MooreMTBench` (config `flores`, tag `v1.0`) | [Hugging Face](https://huggingface.co/datasets/goaicorp/MooreMTBench) |
| ASR | `goaicorp/GOAI-MooreSpeechCorpora` | [Hugging Face](https://huggingface.co/datasets/goaicorp/GOAI-MooreSpeechCorpora) |
| TTS | `goaicorp/MooreMTBench` | Idem MT (`mos`) |

### Dioula (`dyu_Latn`)

| Tâche | Dataset | Fiche |
|-------|---------|--------|
| MT | `goaicorp/DioulaMTBench` (config `flores`, tag `v1.0`) | [Hugging Face](https://huggingface.co/datasets/goaicorp/DioulaMTBench) |
| ASR | `goaicorp/goai-dioula-speech` | [Hugging Face](https://huggingface.co/datasets/goaicorp/goai-dioula-speech) |
| TTS | `goaicorp/DioulaMTBench` | Idem MT (`dyu`) |

**Groupes** (MT/TTS, les deux langues) : `general` = FLORES+ `dev` (997) et `devtest` (1 012). Les groupes `health`, `education`, `child_protection` et `environment` sont désactivés pour l'instant : leurs données sont en cours de correction.

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
6. Reproductibilité : épingler une révision avec `hf_revision` dans le YAML (les configs par défaut épinglent `v1.0`).

> [!TIP]
> Après le push, tester `load_dataset` puis un run avec `--max-samples 2`.

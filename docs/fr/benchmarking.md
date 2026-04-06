# Lancer des benchmarks

Ce guide décrit comment exécuter des évaluations avec **GO AI Bench** après `pip install -e .` à la racine du dépôt.

> [!IMPORTANT]
> Définissez `HF_TOKEN` si un jeu ou un modèle est restreint. Voir le [README principal](../../README.md).

## Comportement par défaut

Sans `--split` ni `--dataset`, la CLI évalue **tous les groupes de benchmark** définis dans `configs/datasets/<langue>.yaml` pour la `--task` choisie. Le modèle est chargé **une fois** et réutilisé pour tous les groupes.

## Modes d’utilisation

| Mode | Options | Comportement |
|------|---------|--------------|
| Tous les groupes | (aucune des options ci-dessous) | Boucle sur les groupes du YAML ; un JSON par groupe + `summary.json` + comparaisons. |
| Un groupe | `--group health` | Idem mais un seul groupe. |
| Un split | `--split test` | Ignore les groupes ; charge un split HF via la config colonnes du YAML. |
| Dataset explicite | `--dataset hf:org/nom` | Remplace `hf_dataset` en gardant les **colonnes** du YAML (MT/ASR/TTS). YAML obligatoire pour ASR/TTS. |

## Sorties

Sous `--output` (défaut `results/`) :

- `results/<langue>/<tâche>/...` pour ASR/TTS.
- Le MT ajoute un dossier de direction : `results/<langue>/mt/<source>_<cible>/<slug_modèle>/`.

Fichiers : JSON par groupe, `summary.json`, et `comparison.json` / `comparison.md` le cas échéant.

## Référence CLI complète (`run_benchmark.py`)

| Option | Description |
|--------|-------------|
| `--task` | Obligatoire. `mt`, `asr` ou `tts`. |
| `--language` | Obligatoire. Code HF (ex. `mos_Latn`, `dyu_Latn`). Doit correspondre à `configs/datasets/<langue>.yaml`. |
| `--model` | Obligatoire. Id Hugging Face ou chemin local compris par la factory. |
| `--source-lang` | MT : code source style NLLB (défaut YAML). |
| `--target-lang` | MT : code cible (défaut YAML ou `--language`). |
| `--source-column` | MT : surcharge colonne texte source. |
| `--target-column` | MT : surcharge colonne texte cible. |
| `--dataset` | Optionnel. `hf:<id_dataset>` pour changer l’id Hub (voir [datasets.md](datasets.md)). |
| `--split` | Optionnel. Nom d’un split HF ; contourne les groupes. |
| `--group` | Optionnel. Un seul groupe de benchmark. |
| `--output` | Racine des résultats (défaut `results`). |
| `--batch-size` | Lots MT, lots pipeline ASR, taille de chunk TTS. |
| `--max-samples` | Plafond d’échantillons par groupe ou par run mono-split. |
| `--device` | `cpu`, `cuda`, `mps` ou `auto`. |
| `--submit-to-leaderboard` | Ajoute à `leaderboard/leaderboard.json` et rafraîchit l’export markdown. |
| `--loopback-asr-model` | TTS : id du modèle ASR pour le WER en boucle (défaut Whisper large v3). |
| `--no-comet` | MT : pas de COMET (plus rapide, moins de dépendances). |
| `--verbose` | Logs debug et versions des bibliothèques. |
| `--version` | Affiche la version du package. |

Invocation :

```bash
python scripts/run_benchmark.py [OPTIONS]
# ou, après installation :
goai-bench [OPTIONS]
```

## Recette : MT tous les groupes (français → mooré)

```bash
python scripts/run_benchmark.py \
  --task mt \
  --language mos_Latn \
  --model facebook/nllb-200-distilled-600M \
  --no-comet \
  --verbose
```

## Recette : MT un domaine

```bash
python scripts/run_benchmark.py \
  --task mt --language mos_Latn \
  --model facebook/nllb-200-distilled-600M \
  --group health \
  --no-comet
```

## Recette : MT sens inverse (mooré → français)

```bash
python scripts/run_benchmark.py \
  --task mt --language mos_Latn \
  --model facebook/nllb-200-distilled-600M \
  --source-lang mos_Latn --target-lang fra_Latn \
  --source-column sentence_mos_Latn \
  --target-column sentence_fra_Latn \
  --no-comet
```

## Recette : ASR (tous les groupes)

```bash
python scripts/run_benchmark.py \
  --task asr --language mos_Latn \
  --model openai/whisper-large-v3 \
  --verbose
```

## Recette : TTS (tous les groupes)

```bash
python scripts/run_benchmark.py \
  --task tts --language mos_Latn \
  --model facebook/mms-tts-mos \
  --verbose
```

## Recette : split explicite (sans groupes)

```bash
python scripts/run_benchmark.py \
  --task mt --language mos_Latn \
  --model facebook/nllb-200-distilled-600M \
  --split test \
  --no-comet
```

## Recette : dataset Hub explicite (mêmes colonnes que le YAML)

```bash
python scripts/run_benchmark.py \
  --task asr --language mos_Latn \
  --model openai/whisper-large-v3 \
  --dataset hf:votre-org/votre-dataset-asr \
  --split test \
  --max-samples 100
```

## `compare_results.py`

Régénère les tableaux de classement à partir des JSON **existants** sous `results/` :

```bash
python scripts/compare_results.py --language mos_Latn --task mt
python scripts/compare_results.py --language dyu_Latn --task asr
python scripts/compare_results.py --language mos_Latn --task all
```

Options : `--results-dir` pour une racine autre que `results/`.

## `run_baselines.py`

Exécute une liste **codée en dur** de modèles de référence sur `mos_Latn` et `dyu_Latn`. Utile pour des balayages internes ; voir le script pour la liste exacte.

> [!NOTE]
> L’alignement de ce script sur `baseline_models` dans `configs/tasks.yaml` est une amélioration prévue (voir [architecture.md](architecture.md)).

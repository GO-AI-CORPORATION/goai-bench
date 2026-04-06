# Ajouter une langue

Enregistrer une nouvelle langue implique les **métadonnées**, un **YAML de datasets** et éventuellement **`tasks.yaml`**.

> [!NOTE]
> Utilisez un code aligné sur HF / NLLB (ex. `xxx_Latn`).

## 1. `configs/languages.yaml`

Ajouter une entrée avec `name`, `iso_639_3`, `hf_code`, `script`, `family`, `region`, `resource_level`, etc.

## 2. `configs/datasets/<code>.yaml`

Copier `mos_Latn.yaml` ou `dyu_Latn.yaml`, adapter :

- `language:` en tête de fichier.
- `hf_dataset`, colonnes, `benchmark_groups` et noms de **splits** réels.

> [!IMPORTANT]
> L’argument CLI `--language` doit être le **nom de fichier sans `.yaml`**.

## 3. Optionnel : `configs/tasks.yaml`

Mettre à jour les `baseline_models` si vous documentez des modèles adaptés à cette langue.

## 4. Côté providers

- **NLLB** : codes hors liste connue → simple avertissement (voir `hf_seq2seq.py`).
- **Whisper** : langues peu couvertes → détection automatique possible.

## 5. Vérification

```bash
python -m pytest tests/test_config_loader.py -v
python scripts/run_benchmark.py --task mt --language <code> \
  --model facebook/nllb-200-distilled-600M --group general --max-samples 2 --no-comet
```

> [!CAUTION]
> Si un groupe référence un split absent du dataset Hub, le groupe sera vide : alignez les noms sur le Hub.

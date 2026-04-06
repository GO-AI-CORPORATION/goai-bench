# Ajouter un provider personnalisé

Les *providers* encapsulent les backends d’inférence derrière de petites interfaces pour que les **évaluateurs restent inchangés**.

> [!NOTE]
> L’enregistrement se fait dans [`factory.py`](../../src/goai_bench/providers/factory.py). Pour des essais ponctuels, vous pouvez aussi construire un provider en Python et l’appeler via `run_evaluation` (voir exemple en fin de page).

## Interfaces (`providers/base.py`)

### MT — `MTProvider`

- `translate(text, source_lang, target_lang) -> str`
- `translate_batch(texts, source_lang, target_lang, batch_size=...) -> list[str]` (surcharge optionnelle)
- `info() -> ProviderInfo`

### ASR — `ASRProvider`

- `transcribe(audio, sampling_rate=16000, language=None) -> str`
- `transcribe_batch(data, language=None, batch_size=8) -> list[str]` (optionnel ; chaque élément utilise `audio_array` ou `audio_path`)
- `info() -> ProviderInfo`

### TTS — `TTSProvider`

- `synthesize(text, language=None) -> np.ndarray`
- `synthesize_batch(texts, language=None)` (optionnel ; utilisé par `TTSEvaluator` s’il existe)
- `info() -> ProviderInfo`

## Exemple : provider MT minimal

```python
from goai_bench.providers.base import MTProvider, ProviderInfo

class MyMTProvider(MTProvider):
    def __init__(self, model_id: str, device: str = "auto") -> None:
        self._model_id = model_id
        # charger votre modèle ici

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        # appel API ou modèle local
        return translated_text

    def info(self) -> ProviderInfo:
        return ProviderInfo(name="my_mt", model_id=self._model_id)
```

Enregistrement dans `create_mt_provider` :

```python
def create_mt_provider(model_id: str, device: str = "auto") -> MTProvider:
    if model_id.startswith("myorg:"):
        from goai_bench.providers.mt.my_mt import MyMTProvider
        return MyMTProvider(model_id.split(":", 1)[1], device)
    ...
```

> [!TIP]
> Utilisez un **préfixe** (`myorg:`) ou une convention de nom pour éviter les collisions avec les ids Hugging Face.

## Exemple : squelette ASR

```python
import numpy as np
from goai_bench.providers.base import ASRProvider, ProviderInfo

class MyASRProvider(ASRProvider):
    def transcribe(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        language: str | None = None,
    ) -> str:
        # mono float32 ; resampler à 16 kHz si besoin
        return hypothesis_text

    def info(self) -> ProviderInfo:
        return ProviderInfo(name="my_asr", model_id="my/model")
```

## Exemple : squelette TTS

```python
import numpy as np
from goai_bench.providers.base import TTSProvider, ProviderInfo

class MyTTSProvider(TTSProvider):
    def synthesize(self, text: str, language: str | None = None) -> np.ndarray:
        # waveform float32 à 16 kHz de préférence
        return waveform

    def info(self) -> ProviderInfo:
        return ProviderInfo(name="my_tts", model_id="my/tts")
```

## Évaluation avec un provider personnalisé (Python)

Vous pouvez contourner la factory dans un notebook ou un script :

```python
from goai_bench.core.evaluator import run_evaluation

provider = MyMTProvider("dummy", device="cpu")
# data = liste de dicts depuis DataLoader
result = run_evaluation(
    task="mt",
    provider=provider,
    data=data,
    language="mos_Latn",
    source_lang="fra_Latn",
    target_lang="mos_Latn",
    batch_size=4,
    compute_comet=False,
)
```

> [!WARNING]
> Les providers personnalisés doivent produire des **types compatibles** (audio float32, chaînes UTF-8 valides) sinon les métriques ou l’écriture des résultats peuvent échouer.

## Cache et périphériques

Réutilisez [`resolve_device`](../../src/goai_bench/core/device.py) et [`get_cached` / `put_cached`](../../src/goai_bench/core/model_cache.py) comme les providers intégrés pour éviter de recharger les poids à chaque groupe.

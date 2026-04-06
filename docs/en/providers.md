# Adding a custom provider

Providers wrap inference backends behind small interfaces so **evaluators stay unchanged**.

> [!NOTE]
> Registration happens in [`factory.py`](../../src/goai_bench/providers/factory.py). For one-off experiments you can also construct a provider in Python and pass it to `run_evaluation` if you add a thin wrapper (see example at the end).

## Interfaces (`providers/base.py`)

### MT — `MTProvider`

- `translate(text, source_lang, target_lang) -> str`
- `translate_batch(texts, source_lang, target_lang, batch_size=...) -> list[str]` (optional override)
- `info() -> ProviderInfo`

### ASR — `ASRProvider`

- `transcribe(audio, sampling_rate=16000, language=None) -> str`
- `transcribe_batch(data, language=None, batch_size=8) -> list[str]` (optional override; `data` items use `audio_array` or `audio_path`)
- `info() -> ProviderInfo`

### TTS — `TTSProvider`

- `synthesize(text, language=None) -> np.ndarray`
- `synthesize_batch(texts, language=None)` (optional; used by `TTSEvaluator` when present)
- `info() -> ProviderInfo`

## Example: minimal custom MT provider

```python
from goai_bench.providers.base import MTProvider, ProviderInfo

class MyMTProvider(MTProvider):
    def __init__(self, model_id: str, device: str = "auto") -> None:
        self._model_id = model_id
        # load your model here

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        # call your API or local model
        return translated_text

    def info(self) -> ProviderInfo:
        return ProviderInfo(name="my_mt", model_id=self._model_id)
```

Register in `create_mt_provider`:

```python
def create_mt_provider(model_id: str, device: str = "auto") -> MTProvider:
    if model_id.startswith("myorg:"):
        from goai_bench.providers.mt.my_mt import MyMTProvider
        return MyMTProvider(model_id.split(":", 1)[1], device)
    ...
```

> [!TIP]
> Use a **prefix** (`myorg:`) or a filename convention to avoid colliding with Hugging Face hub ids.

## Example: ASR provider skeleton

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
        # mono float32 expected; resample to 16 kHz if needed
        return hypothesis_text

    def info(self) -> ProviderInfo:
        return ProviderInfo(name="my_asr", model_id="my/model")
```

## Example: TTS provider skeleton

```python
import numpy as np
from goai_bench.providers.base import TTSProvider, ProviderInfo

class MyTTSProvider(TTSProvider):
    def synthesize(self, text: str, language: str | None = None) -> np.ndarray:
        # return float32 waveform at 16 kHz when possible
        return waveform

    def info(self) -> ProviderInfo:
        return ProviderInfo(name="my_tts", model_id="my/tts")
```

## Running evaluation with a custom provider (Python)

You can bypass the factory inside a notebook or script:

```python
from goai_bench.core.evaluator import run_evaluation

provider = MyMTProvider("dummy", device="cpu")
# data = list of dicts from DataLoader
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
> Custom providers must produce **compatible dtypes** (float32 audio, valid UTF-8 strings) or metrics and saving may fail.

## Caching and devices

Reuse [`resolve_device`](../../src/goai_bench/core/device.py) and [`get_cached` / `put_cached`](../../src/goai_bench/core/model_cache.py) like the built-in providers to avoid reloading large weights on every benchmark group.

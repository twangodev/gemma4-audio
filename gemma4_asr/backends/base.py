from typing import Protocol, runtime_checkable

import numpy as np

from gemma4_asr.config import TranscriptionResult


@runtime_checkable
class InferenceBackend(Protocol):
    def load_model(
        self,
        model_id: str,
        quantization: str | None = None,
        device: str | None = None,
    ) -> None: ...

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
    ) -> TranscriptionResult: ...

    def cleanup(self) -> None: ...

from typing import Protocol, runtime_checkable

import numpy as np

from gemma4_audio.config import BatchItem, TranscriptionResult


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
        max_output_tokens: int = 512,
    ) -> TranscriptionResult: ...

    def transcribe_batch(
        self, items: list[BatchItem]
    ) -> list[TranscriptionResult]: ...

    def cleanup(self) -> None: ...


def loop_transcribe_batch(
    backend: InferenceBackend, items: list[BatchItem]
) -> list[TranscriptionResult]:
    """Serial fallback for backends without native batching.

    Calls ``transcribe`` once per item, preserving order. Used by the
    transformers and mlx backends, which have no continuous batching.
    """
    return [
        backend.transcribe(it.audio, it.sample_rate, it.prompt, it.max_output_tokens)
        for it in items
    ]

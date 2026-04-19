from typing import Protocol, runtime_checkable

from gemma4_audio.config import TranscribeRequest, TranscriptionResult


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
        batch: list[TranscribeRequest],
    ) -> list[TranscriptionResult]:
        """Transcribe a batch of audio clips.

        Returns one TranscriptionResult per request, in the same order.
        Backends that support true batching (vLLM) process the batch
        concurrently; backends that don't (Transformers, MLX) iterate
        internally. Callers treat both alike.
        """
        ...

    def cleanup(self) -> None: ...

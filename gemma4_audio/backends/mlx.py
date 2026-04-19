import time

from gemma4_audio.config import TranscribeRequest, TranscriptionResult


class MLXBackend:
    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def load_model(
        self,
        model_id: str,
        quantization: str | None = None,
        device: str | None = None,
    ) -> None:
        from mlx_vlm import load

        self._model, self._processor = load(model_id)

    def transcribe(
        self,
        batch: list[TranscribeRequest],
    ) -> list[TranscriptionResult]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load_model() before transcribe().")
        return [self._transcribe_one(req) for req in batch]

    def _transcribe_one(self, req: TranscribeRequest) -> TranscriptionResult:
        from mlx_vlm import generate

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": req.audio, "sample_rate": req.sample_rate},
                    {"type": "text", "text": req.prompt},
                ],
            }
        ]

        formatted = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = time.perf_counter()
        response = generate(
            self._model, self._processor, formatted, max_tokens=req.max_output_tokens
        )
        elapsed = time.perf_counter() - start

        return TranscriptionResult(
            text=response.strip(),
            elapsed_seconds=elapsed,
            tokens_generated=len(response.split()),
        )

    def cleanup(self) -> None:
        self._model = None
        self._processor = None

import time

import numpy as np

from gemma4_audio.config import TranscriptionResult


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
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
    ) -> TranscriptionResult:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load_model() before transcribe().")

        from mlx_vlm import generate

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio, "sample_rate": sample_rate},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        formatted = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = time.perf_counter()
        response = generate(
            self._model, self._processor, formatted, max_tokens=512
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

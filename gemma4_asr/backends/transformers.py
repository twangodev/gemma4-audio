import time

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from gemma4_asr.config import TranscriptionResult


class TransformersBackend:
    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def load_model(
        self,
        model_id: str,
        quantization: str | None = None,
        device: str | None = None,
    ) -> None:
        kwargs: dict = {"device_map": device or "auto"}

        if quantization is not None:
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"Unsupported quantization: {quantization}")

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            model_id, **kwargs
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
    ) -> TranscriptionResult:
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load_model() before transcribe().")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio, "sample_rate": sample_rate},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=512)
        elapsed = time.perf_counter() - start

        tokens_generated = outputs.shape[-1] - input_len
        response = self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        return TranscriptionResult(
            text=response.strip(),
            elapsed_seconds=elapsed,
            tokens_generated=int(tokens_generated),
        )

    def cleanup(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

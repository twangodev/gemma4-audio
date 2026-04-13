import time

import numpy as np

from gemma4_audio.config import TranscriptionResult


class VLLMBackend:
    def __init__(self) -> None:
        self._llm = None
        self._tokenizer = None
        self._sampling_params = None

    def load_model(
        self,
        model_id: str,
        quantization: str | None = None,
        device: str | None = None,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        kwargs: dict = {
            "model": model_id,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.90,
        }

        if quantization == "4bit":
            kwargs["quantization"] = "awq"
        elif quantization == "8bit":
            kwargs["quantization"] = "gptq"
        elif quantization is not None:
            raise ValueError(f"Unsupported quantization for vLLM: {quantization}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._llm = LLM(**kwargs)
        self._sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
    ) -> TranscriptionResult:
        if self._llm is None or self._tokenizer is None:
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

        text_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = time.perf_counter()
        outputs = self._llm.generate(text_prompt, self._sampling_params)
        elapsed = time.perf_counter() - start

        generated_text = outputs[0].outputs[0].text
        tokens_generated = len(outputs[0].outputs[0].token_ids)

        return TranscriptionResult(
            text=generated_text.strip(),
            elapsed_seconds=elapsed,
            tokens_generated=tokens_generated,
        )

    def cleanup(self) -> None:
        del self._llm
        self._llm = None
        self._tokenizer = None

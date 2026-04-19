import time

import numpy as np

from gemma4_audio.config import TranscriptionResult


class VLLMBackend:
    def __init__(self) -> None:
        self._llm = None
        self._processor = None

    def load_model(
        self,
        model_id: str,
        quantization: str | None = None,
        device: str | None = None,
    ) -> None:
        from transformers import AutoProcessor
        from vllm import LLM

        kwargs: dict = {
            "model": model_id,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.90,
            "limit_mm_per_prompt": {"audio": 1},
        }

        if quantization == "4bit":
            kwargs["quantization"] = "awq"
        elif quantization == "8bit":
            kwargs["quantization"] = "gptq"
        elif quantization is not None:
            raise ValueError(f"Unsupported quantization for vLLM: {quantization}")

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._llm = LLM(**kwargs)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
        max_output_tokens: int = 512,
    ) -> TranscriptionResult:
        if self._llm is None or self._processor is None:
            raise RuntimeError("Call load_model() before transcribe().")

        from vllm import SamplingParams

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_output_tokens)

        start = time.perf_counter()
        outputs = self._llm.generate(
            {
                "prompt": text_prompt,
                "multi_modal_data": {"audio": (audio, sample_rate)},
            },
            sampling_params=sampling_params,
        )
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
        self._processor = None

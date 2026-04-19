import time

from gemma4_audio.config import TranscribeRequest, TranscriptionResult


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
        batch: list[TranscribeRequest],
    ) -> list[TranscriptionResult]:
        if self._llm is None or self._processor is None:
            raise RuntimeError("Call load_model() before transcribe().")
        # Batched audio with variable-length clips crashes vLLM's Gemma-4
        # multimodal path — the HF processor pads each request's features
        # independently, so features arrive as a ragged list where the model
        # expects a stacked tensor. Fix is open upstream in
        # vllm-project/vllm#39459 (not yet merged as of writing). Iterate
        # per-request for now; the batched interface stays so swapping back
        # to a single generate() call is a one-liner once the fix ships.
        return [self._transcribe_one(req) for req in batch]

    def _transcribe_one(self, req: TranscribeRequest) -> TranscriptionResult:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=req.max_output_tokens
        )

        start = time.perf_counter()
        outputs = self._llm.generate(
            self._build_prompt(req), sampling_params=sampling_params
        )
        elapsed = time.perf_counter() - start

        return TranscriptionResult(
            text=outputs[0].outputs[0].text.strip(),
            elapsed_seconds=elapsed,
            tokens_generated=len(outputs[0].outputs[0].token_ids),
        )

    def _build_prompt(self, req: TranscribeRequest) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": req.prompt},
                ],
            }
        ]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": text_prompt,
            "multi_modal_data": {"audio": (req.audio, req.sample_rate)},
        }

    def cleanup(self) -> None:
        del self._llm
        self._llm = None
        self._processor = None

import time

import numpy as np

from gemma4_audio.config import BatchItem, TranscriptionResult

# NOTE: running the Gemma 4 12B (gemma4_unified) with batched audio requires a
# small fix to vLLM's gemma4_unified._process_audio_input (it assumes a single
# stacked tensor and crashes on the ragged-batch list). The fix lives in
# patches/vllm-gemma4-unified-batched-audio.patch and must be applied to the
# vLLM source the EngineCore subprocess imports (a runtime monkeypatch does not
# reach that subprocess). Remove once fixed upstream.


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

    def _build_prompt(self, prompt: str) -> str:
        # Text BEFORE audio, per the Gemma 4 model card's canonical audio
        # example. The unified 12B is unstable with audio-first ordering
        # (greedy decoding degenerates into a <|channel>thought loop); text
        # first gives clean, deterministic transcriptions.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio"},
                ],
            }
        ]
        return self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _sampling_params(max_output_tokens: int):
        from vllm import SamplingParams

        # Greedy for deterministic, reproducible ASR (matches the E2B/E4B
        # methodology). skip_special_tokens=False keeps the <|channel>/
        # <channel|> markers so parse_thinking_output can split off the
        # (empty) thought channel the 12B emits even with thinking disabled.
        return SamplingParams(
            temperature=0.0,
            max_tokens=max_output_tokens,
            skip_special_tokens=False,
        )

    @staticmethod
    def _request_latency(output, fallback: float) -> float:
        """Per-request end-to-end latency from vLLM metrics, else ``fallback``.

        Under batching this includes queue-wait; the headline speed metric is
        the corpus-level throughput_rtfx (total audio / wall clock).
        """
        metrics = getattr(output, "metrics", None)
        if metrics is not None:
            arrival = getattr(metrics, "arrival_time", None)
            finished = getattr(metrics, "finished_time", None)
            if arrival is not None and finished is not None:
                return float(finished - arrival)
        return fallback

    def transcribe_batch(
        self, items: list[BatchItem]
    ) -> list[TranscriptionResult]:
        if self._llm is None or self._processor is None:
            raise RuntimeError("Call load_model() before transcribe_batch().")
        if not items:
            return []

        from vllm.reasoning.gemma4_utils import parse_thinking_output

        requests = [
            {
                "prompt": self._build_prompt(it.prompt),
                "multi_modal_data": {"audio": (it.audio, it.sample_rate)},
            }
            for it in items
        ]
        sampling = [self._sampling_params(it.max_output_tokens) for it in items]

        start = time.perf_counter()
        # vLLM returns outputs in input order and continuous-batches internally.
        outputs = self._llm.generate(requests, sampling_params=sampling)
        elapsed = time.perf_counter() - start

        fallback = elapsed / len(outputs)
        results = []
        for output in outputs:
            answer = parse_thinking_output(output.outputs[0].text)["answer"]
            results.append(
                TranscriptionResult(
                    text=answer.strip(),
                    elapsed_seconds=self._request_latency(output, fallback),
                    tokens_generated=len(output.outputs[0].token_ids),
                )
            )
        return results

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str,
        max_output_tokens: int = 512,
    ) -> TranscriptionResult:
        return self.transcribe_batch(
            [BatchItem(audio, sample_rate, prompt, max_output_tokens)]
        )[0]

    def cleanup(self) -> None:
        del self._llm
        self._llm = None
        self._processor = None

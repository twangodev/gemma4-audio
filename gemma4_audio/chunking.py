from typing import Callable, Iterator

import numpy as np

from gemma4_audio.backends.base import InferenceBackend
from gemma4_audio.config import TranscriptionResult


def split_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_s: float,
    overlap_s: float = 0.0,
) -> Iterator[np.ndarray]:
    """Yield fixed-duration audio chunks.

    overlap_s reserved for future boundary-dedupe ("Strategy B"). With
    overlap_s=0.0 (current default) chunks are adjacent and non-overlapping.
    """
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be positive")
    if overlap_s < 0 or overlap_s >= chunk_duration_s:
        raise ValueError("overlap_s must be in [0, chunk_duration_s)")

    chunk_samples = int(chunk_duration_s * sample_rate)
    step_samples = int((chunk_duration_s - overlap_s) * sample_rate)

    i = 0
    while i < len(audio):
        yield audio[i : i + chunk_samples]
        i += step_samples


def stitch_hypotheses(hypotheses: list[str], overlap_s: float = 0.0) -> str:
    """Stitch chunk hypotheses into one transcription.

    Current behaviour (Strategy A): simple whitespace-joined concat of stripped
    chunks. overlap_s is accepted but ignored; Strategy B will use it to drive
    fuzzy word-level deduplication of the overlap region.
    """
    del overlap_s  # Strategy B hook
    return " ".join(h.strip() for h in hypotheses if h.strip())


def chunked_transcribe(
    backend: InferenceBackend,
    audio: np.ndarray,
    sample_rate: int,
    prompt: str,
    chunk_duration_s: float,
    max_output_tokens_fn: Callable[[float], int],
    overlap_s: float = 0.0,
) -> TranscriptionResult:
    """Transcribe long audio by chunking, then stitch.

    max_output_tokens_fn: given a chunk's duration in seconds, returns the
    token budget for that chunk (so the wrapper inherits the caller's
    auto-scaling policy instead of hardcoding one).
    """
    hypotheses: list[str] = []
    total_latency = 0.0
    total_tokens = 0

    for chunk in split_audio(audio, sample_rate, chunk_duration_s, overlap_s):
        chunk_duration = len(chunk) / sample_rate
        max_tokens = max_output_tokens_fn(chunk_duration)
        result = backend.transcribe(chunk, sample_rate, prompt, max_tokens)
        hypotheses.append(result.text)
        total_latency += result.elapsed_seconds
        total_tokens += result.tokens_generated

    return TranscriptionResult(
        text=stitch_hypotheses(hypotheses, overlap_s),
        elapsed_seconds=total_latency,
        tokens_generated=total_tokens,
    )

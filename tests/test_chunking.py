from unittest.mock import MagicMock

import numpy as np
import pytest

from gemma4_audio.chunking import (
    chunked_transcribe,
    split_audio,
    stitch_hypotheses,
)
from gemma4_audio.config import TranscriptionResult


SR = 16000


def test_split_audio_adjacent_chunks():
    audio = np.arange(10 * SR, dtype=np.float32)
    chunks = list(split_audio(audio, SR, chunk_duration_s=3.0))
    assert len(chunks) == 4
    assert len(chunks[0]) == 3 * SR
    assert len(chunks[1]) == 3 * SR
    assert len(chunks[2]) == 3 * SR
    # Last chunk is the 1-second remainder
    assert len(chunks[3]) == SR
    # No overlap: chunks concatenate back to original
    assert np.array_equal(np.concatenate(chunks), audio)


def test_split_audio_rejects_bad_params():
    audio = np.zeros(SR, dtype=np.float32)
    with pytest.raises(ValueError):
        list(split_audio(audio, SR, chunk_duration_s=0))
    with pytest.raises(ValueError):
        list(split_audio(audio, SR, chunk_duration_s=1.0, overlap_s=1.0))


def test_split_audio_overlap_hook():
    # Strategy B hook: overlap_s should advance by step < chunk_duration
    audio = np.arange(10 * SR, dtype=np.float32)
    chunks = list(split_audio(audio, SR, chunk_duration_s=3.0, overlap_s=1.0))
    # step=2s → chunks start at 0, 2, 4, 6, 8 → 5 chunks
    assert len(chunks) == 5


def test_stitch_hypotheses_joins_and_strips():
    assert stitch_hypotheses(["  hello ", "world  "]) == "hello world"
    assert stitch_hypotheses(["hello", "", "world"]) == "hello world"
    assert stitch_hypotheses([]) == ""


def test_chunked_transcribe_aggregates_latency_and_tokens():
    audio = np.zeros(5 * SR, dtype=np.float32)
    mock_backend = MagicMock()
    responses = iter(
        [
            TranscriptionResult("foo", 0.1, 2),
            TranscriptionResult("bar", 0.2, 3),
            TranscriptionResult("baz", 0.3, 4),
        ]
    )
    mock_backend.transcribe.side_effect = lambda *a, **k: next(responses)

    result = chunked_transcribe(
        mock_backend,
        audio,
        SR,
        prompt="p",
        chunk_duration_s=2.0,
        max_output_tokens_fn=lambda d: 512,
    )

    assert result.text == "foo bar baz"
    assert result.elapsed_seconds == pytest.approx(0.6)
    assert result.tokens_generated == 9
    assert mock_backend.transcribe.call_count == 3


def test_chunked_transcribe_passes_scaled_max_tokens():
    audio = np.zeros(4 * SR, dtype=np.float32)
    mock_backend = MagicMock()
    mock_backend.transcribe.side_effect = lambda *a, **k: TranscriptionResult(
        "", 0.0, 0
    )

    chunked_transcribe(
        mock_backend,
        audio,
        SR,
        prompt="p",
        chunk_duration_s=2.0,
        max_output_tokens_fn=lambda d: int(d * 10),
    )

    # Each 2s chunk → 20 max tokens
    for call in mock_backend.transcribe.call_args_list:
        args, _ = call
        assert args[3] == 20

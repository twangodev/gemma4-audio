import numpy as np
import pytest

from gemma4_audio.datasets.base import Sample


@pytest.fixture
def fake_samples() -> list[Sample]:
    """Small set of samples with known references for metric testing."""
    return [
        Sample(
            id="s1",
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            reference="the cat sat on the mat",
        ),
        Sample(
            id="s2",
            audio=np.zeros(32000, dtype=np.float32),
            sample_rate=16000,
            reference="hello world",
        ),
        Sample(
            id="s3",
            audio=np.zeros(48000, dtype=np.float32),
            sample_rate=16000,
            reference="one two three four five",
        ),
    ]


@pytest.fixture
def fake_audio_1s() -> np.ndarray:
    """1 second of silence at 16kHz."""
    return np.zeros(16000, dtype=np.float32)

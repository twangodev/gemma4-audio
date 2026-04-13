import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from gemma4_audio.cli import parse_args
from gemma4_audio.config import TranscriptionResult
from gemma4_audio.datasets.base import Sample
from gemma4_audio.eval import run_eval


def test_e2e_with_mock():
    """Full pipeline: CLI args -> eval -> JSON + CSV output."""
    samples = [
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
    ]

    mock_backend = MagicMock()
    responses = iter([
        TranscriptionResult("the cat sat on the mat", 0.1, 10),
        TranscriptionResult("hello world", 0.05, 5),
    ])
    mock_backend.transcribe.side_effect = lambda *a, **k: next(responses)

    mock_dataset = MagicMock()
    mock_dataset.name = "librispeech"
    mock_dataset.__iter__ = MagicMock(return_value=iter(samples))
    mock_dataset.__len__ = MagicMock(return_value=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = str(Path(tmpdir) / "results.json")
        csv_path = str(Path(tmpdir) / "results.csv")

        config = parse_args([
            "eval",
            "--model",
            "google/gemma-4-E4B-it",
            "--output-json",
            json_path,
            "--output-csv",
            csv_path,
            "--quiet",
        ])
        result = run_eval(config, backend=mock_backend, dataset=mock_dataset)

        # Verify metrics
        assert result.corpus_metrics.wer == 0.0
        assert result.corpus_metrics.num_samples == 2

        # Verify JSON output
        with open(json_path) as f:
            data = json.load(f)
        assert data["corpus_metrics"]["wer"] == 0.0
        assert len(data["sample_results"]) == 2

        # Verify CSV output
        csv_content = Path(csv_path).read_text()
        assert "s1" in csv_content
        assert "s2" in csv_content
        assert "the cat sat on the mat" in csv_content

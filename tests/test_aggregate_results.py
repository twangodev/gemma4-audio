import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "aggregate_results.py"


def _write_result(
    path: Path,
    *,
    model: str,
    dataset: str,
    split: str,
    wer: float,
    rtfx_mean: float,
    num_samples: int,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(
        json.dumps(
            {
                "config": {"model": model, "dataset": dataset, "split": split},
                "corpus_metrics": {
                    "wer": wer,
                    "rtfx": {"mean": rtfx_mean, "p50": 0.0, "p95": 0.0},
                    "num_samples": num_samples,
                },
                "sample_results": [],
            }
        )
    )


def _run(root: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_pivots_results_into_markdown_tables(tmp_path):
    _write_result(
        tmp_path / "r0",
        model="google/gemma-4-E2B-it",
        dataset="librispeech",
        split="test.clean",
        wer=0.0523,
        rtfx_mean=2.3,
        num_samples=10,
    )
    _write_result(
        tmp_path / "r1",
        model="google/gemma-4-E4B-it",
        dataset="librispeech",
        split="test.clean",
        wer=0.0411,
        rtfx_mean=1.8,
        num_samples=10,
    )
    _write_result(
        tmp_path / "r2",
        model="google/gemma-4-E2B-it",
        dataset="ami",
        split="test",
        wer=0.18,
        rtfx_mean=1.2,
        num_samples=5,
    )

    out = _run(tmp_path)

    assert "## ASR Benchmark Results" in out
    assert "gemma-4-E2B-it" in out
    assert "gemma-4-E4B-it" in out
    assert "5.23%" in out
    assert "4.11%" in out
    assert "2.30x" in out
    # Missing (model, dataset) cell renders as the em-dash placeholder.
    ami_row = next(line for line in out.splitlines() if line.startswith("| ami:test"))
    assert "—" in ami_row


def test_skips_missing_or_malformed_results(tmp_path):
    _write_result(
        tmp_path / "good",
        model="google/gemma-4-E2B-it",
        dataset="librispeech",
        split="test.clean",
        wer=0.05,
        rtfx_mean=2.0,
        num_samples=10,
    )
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "results.json").write_text("")
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "results.json").write_text(json.dumps({"config": {}}))

    out = _run(tmp_path)

    assert "librispeech:test.clean" in out
    assert "5.00%" in out


def test_reports_empty_state(tmp_path):
    out = _run(tmp_path)
    assert "No results found" in out

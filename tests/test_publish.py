import csv
import json

from gemma4_audio.cli import main
from gemma4_audio.publish import publish


def _write_run(root, slug, dataset, split):
    d = root / slug
    d.mkdir(parents=True)
    results = {
        "config": {
            "model": "google/gemma-4-12B-it",
            "dataset": dataset,
            "split": split,
        },
        "corpus_metrics": {"wer": 0.1, "throughput_rtfx": 50.0, "num_samples": 1},
        "sample_results": [
            {
                "id": "s1",
                "reference": "hello world",
                "hypothesis": "hello word",
                "wer": 0.5,
                "cer": 0.1,
                "mer": 0.1,
                "wil": 0.1,
                "substitutions": 1,
                "insertions": 0,
                "deletions": 0,
                "latency_s": 0.1,
                "rtfx": 10.0,
                "audio_duration_s": 1.0,
            }
        ],
    }
    (d / "results.json").write_text(json.dumps(results))
    with open(d / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "reference", "hypothesis", "wer", "latency_s"])
        w.writerow(["s1", "hello world", "hello word", "0.5", "0.1"])


def test_publish_strips_samples_and_drops_text_columns(tmp_path):
    raw = tmp_path / "eval_results"
    out = tmp_path / "out"
    _write_run(
        raw, "google_gemma-4-12B-it__librispeech_test.clean", "librispeech", "test.clean"
    )
    publish(raw, out)

    pub = out / "google_gemma-4-12B-it__librispeech_test.clean"
    data = json.loads((pub / "results.json").read_text())
    assert data["sample_results"] == []  # per-sample text stripped
    assert data["corpus_metrics"]["wer"] == 0.1  # aggregates preserved
    assert data["config"]["dataset"] == "librispeech"

    header = (pub / "results.csv").read_text().splitlines()[0]
    assert "reference" not in header and "hypothesis" not in header
    assert "wer" in header and "id" in header
    # rows are retained (just without the text columns)
    assert "s1" in (pub / "results.csv").read_text()


def test_publish_excludes_nonredistributable_datasets(tmp_path):
    raw = tmp_path / "eval_results"
    out = tmp_path / "out"
    _write_run(raw, "google_gemma-4-12B-it__tedlium_test", "tedlium", "test")
    _write_run(raw, "google_gemma-4-12B-it__spgispeech_test", "spgispeech", "test")
    _write_run(raw, "google_gemma-4-12B-it__ami_test", "ami", "test")
    publish(raw, out)

    assert (out / "google_gemma-4-12B-it__ami_test").exists()  # redistributable
    assert not (out / "google_gemma-4-12B-it__tedlium_test").exists()
    assert not (out / "google_gemma-4-12B-it__spgispeech_test").exists()


def test_publish_cli_subcommand(tmp_path):
    raw = tmp_path / "eval_results"
    out = tmp_path / "out"
    _write_run(raw, "google_gemma-4-E4B-it__ami_test", "ami", "test")
    main(["publish", "--results-dir", str(raw), "--out", str(out)])
    assert (out / "google_gemma-4-E4B-it__ami_test" / "results.json").exists()

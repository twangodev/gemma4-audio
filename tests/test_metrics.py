import pytest

from gemma4_asr.metrics import compute_corpus_metrics, compute_sample_metrics


def test_perfect_transcription():
    r = compute_sample_metrics("s1", "the cat sat on the mat", "the cat sat on the mat", 0.5, 2.0)
    assert r.wer == 0.0
    assert r.cer == 0.0
    assert r.substitutions == 0
    assert r.rtfx == pytest.approx(4.0)


def test_substitution():
    r = compute_sample_metrics("s2", "the cat sat", "the bat sat", 0.3, 1.0)
    assert r.wer == pytest.approx(1 / 3)
    assert r.substitutions == 1


def test_insertion_and_deletion():
    ins = compute_sample_metrics("s3", "the cat", "the big cat", 0.2, 1.0)
    assert ins.insertions == 1

    dele = compute_sample_metrics("s4", "the big cat", "the cat", 0.2, 1.0)
    assert dele.deletions == 1


def test_corpus_metrics():
    samples = [
        compute_sample_metrics("s1", "the cat sat", "the cat sat", 0.5, 2.0),
        compute_sample_metrics("s2", "hello world", "hello word", 0.3, 1.0),
    ]
    corpus = compute_corpus_metrics(samples, ["the cat sat", "hello world"], ["the cat sat", "hello word"])
    assert corpus.num_samples == 2
    assert corpus.wer > 0.0
    assert corpus.latency.mean == pytest.approx(0.4)
    assert corpus.audio_duration.total == pytest.approx(3.0)

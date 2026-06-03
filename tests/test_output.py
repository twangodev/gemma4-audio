from gemma4_audio.config import EvalConfig, EvalResult
from gemma4_audio.metrics import compute_corpus_metrics, compute_sample_metrics
from gemma4_audio.output import format_stdout


def _result(wall_clock_s):
    sm = [compute_sample_metrics("s1", "a b", "a b", 0.5, 2.0)]
    corpus = compute_corpus_metrics(sm, ["a b"], ["a b"], wall_clock_s=wall_clock_s)
    return EvalResult(
        config=EvalConfig(model="m"), corpus_metrics=corpus, sample_results=sm
    )


def test_format_stdout_shows_throughput_when_present():
    out = format_stdout(_result(wall_clock_s=1.0))
    assert "Throughput" in out


def test_format_stdout_omits_throughput_when_none():
    out = format_stdout(_result(wall_clock_s=None))
    assert "Throughput" not in out

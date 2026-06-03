import numpy as np

from gemma4_audio.config import EvalConfig, TranscriptionResult
from gemma4_audio.datasets.base import Sample
from gemma4_audio.eval import run_eval


class _ListDataset:
    name = "librispeech"

    def __init__(self, samples):
        self._samples = samples

    def __iter__(self):
        return iter(self._samples)

    def __len__(self):
        return len(self._samples)


class _RecordingBatchBackend:
    """Records batch sizes and echoes each item's audio length as text."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def load_model(self, *a, **k):
        pass

    def transcribe(self, audio, sample_rate, prompt, max_output_tokens=512):
        return TranscriptionResult(str(len(audio)), 0.1, 1)

    def transcribe_batch(self, items):
        self.batch_sizes.append(len(items))
        return [TranscriptionResult(str(len(it.audio)), 0.1, 1) for it in items]

    def cleanup(self):
        pass


def _samples(n):
    return [
        Sample(
            id=f"s{i}",
            audio=np.zeros((i + 1) * 16000, dtype=np.float32),
            sample_rate=16000,
            reference=str((i + 1) * 16000),
        )
        for i in range(n)
    ]


def test_eval_groups_into_batches_of_batch_size():
    backend = _RecordingBatchBackend()
    cfg = EvalConfig(model="m", batch_size=2, quiet=True, output_dir="")
    result = run_eval(cfg, backend=backend, dataset=_ListDataset(_samples(5)))
    assert backend.batch_sizes == [2, 2, 1]
    assert result.corpus_metrics.num_samples == 5


def test_eval_preserves_sample_order_across_batches():
    backend = _RecordingBatchBackend()
    cfg = EvalConfig(model="m", batch_size=2, quiet=True, output_dir="")
    result = run_eval(cfg, backend=backend, dataset=_ListDataset(_samples(5)))
    # echo backend returns len(audio) as hypothesis; references match it -> WER 0
    assert [s.hypothesis for s in result.sample_results] == [
        str((i + 1) * 16000) for i in range(5)
    ]
    assert result.corpus_metrics.wer == 0.0


def test_eval_flushes_batch_before_chunked_sample():
    # s0 short, s1 long (-> chunked, per-sample path), s2 short.
    # With a large batch_size, s0 would normally wait; the chunked s1 must
    # flush s0 first so output order stays s0, s1, s2.
    sr = 16000
    samples = [
        Sample(id="s0", audio=np.zeros(sr, dtype=np.float32), sample_rate=sr, reference="a"),
        Sample(id="s1", audio=np.zeros(sr * 70, dtype=np.float32), sample_rate=sr, reference="b"),
        Sample(id="s2", audio=np.zeros(sr, dtype=np.float32), sample_rate=sr, reference="c"),
    ]
    backend = _RecordingBatchBackend()
    cfg = EvalConfig(
        model="m", batch_size=16, chunk_duration_s=30.0, quiet=True, output_dir=""
    )
    result = run_eval(cfg, backend=backend, dataset=_ListDataset(samples))
    assert [s.id for s in result.sample_results] == ["s0", "s1", "s2"]
    # s0 flushed alone (before chunked s1), then s2 flushed alone at the end.
    assert backend.batch_sizes == [1, 1]


def test_eval_batch_size_one_still_works():
    backend = _RecordingBatchBackend()
    cfg = EvalConfig(model="m", batch_size=1, quiet=True, output_dir="")
    result = run_eval(cfg, backend=backend, dataset=_ListDataset(_samples(3)))
    assert backend.batch_sizes == [1, 1, 1]
    assert result.corpus_metrics.num_samples == 3

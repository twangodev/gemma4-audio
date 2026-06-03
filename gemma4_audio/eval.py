import itertools
import time

from tqdm import tqdm

from gemma4_audio.backends import select_backend
from gemma4_audio.backends.base import InferenceBackend
from gemma4_audio.chunking import chunked_transcribe
from gemma4_audio.config import BatchItem, EvalConfig, EvalResult
from gemma4_audio.datasets import get_dataset
from gemma4_audio.datasets.base import Dataset
from gemma4_audio.metrics import compute_corpus_metrics, compute_sample_metrics
from gemma4_audio.output import (
    format_stdout,
    resolve_output_paths,
    write_csv,
    write_json,
)


def _resolve_max_tokens(config: EvalConfig, duration_s: float) -> int:
    if config.max_output_tokens is not None:
        return config.max_output_tokens
    # 4 tokens/sec ≈ 240 wpm (above typical speech);
    # floor of 512 preserves prior behavior on short clips.
    return max(512, int(duration_s * 4))


def run_eval(
    config: EvalConfig,
    *,
    backend: InferenceBackend | None = None,
    dataset: Dataset | None = None,
) -> EvalResult:
    """Run ASR evaluation. Accepts optional injected backend/dataset for testing."""
    # Setup dataset
    if dataset is None:
        dataset = get_dataset(config.dataset)
        dataset.load(config.split, seed=config.seed, streaming=config.streaming)

    # Setup backend
    if backend is None:
        backend = select_backend(config.backend)
        backend.load_model(config.model, quantization=config.quantization)

    # Iterate and transcribe
    sample_results = []
    all_references = []
    all_hypotheses = []

    samples_iter = iter(dataset)
    if config.limit is not None:
        samples_iter = itertools.islice(samples_iter, config.limit)

    total = config.limit

    progress = tqdm(
        samples_iter,
        total=total,
        desc="Evaluating",
        disable=config.quiet,
        position=0,
        leave=True,
    )

    # Pending non-chunked samples awaiting a batched transcribe call.
    batch: list[tuple[object, float, BatchItem]] = []

    def _record(sample, audio_duration: float, result) -> None:
        sample_results.append(
            compute_sample_metrics(
                id=sample.id,
                reference=sample.reference,
                hypothesis=result.text,
                latency_s=result.elapsed_seconds,
                audio_duration_s=audio_duration,
            )
        )
        all_references.append(sample.reference)
        all_hypotheses.append(result.text)
        if not config.quiet:
            running_wer = sum(s.wer for s in sample_results) / len(sample_results)
            progress.set_postfix(wer=f"{running_wer:.2%}")

    def _flush() -> None:
        if not batch:
            return
        results = backend.transcribe_batch([item for (_, _, item) in batch])
        for (sample, audio_duration, _), result in zip(batch, results, strict=True):
            _record(sample, audio_duration, result)
        batch.clear()

    wall_start = time.perf_counter()
    for sample in progress:
        audio_duration = len(sample.audio) / sample.sample_rate
        chunk_s = config.chunk_duration_s
        if chunk_s is not None and audio_duration > 2 * chunk_s:
            # Long-form path is per-sample; flush pending batch to keep order.
            _flush()
            result = chunked_transcribe(
                backend,
                sample.audio,
                sample.sample_rate,
                config.prompt,
                chunk_duration_s=chunk_s,
                max_output_tokens_fn=lambda d: _resolve_max_tokens(config, d),
            )
            _record(sample, audio_duration, result)
        else:
            batch.append(
                (
                    sample,
                    audio_duration,
                    BatchItem(
                        audio=sample.audio,
                        sample_rate=sample.sample_rate,
                        prompt=config.prompt,
                        max_output_tokens=_resolve_max_tokens(config, audio_duration),
                    ),
                )
            )
            if len(batch) >= config.batch_size:
                _flush()
    _flush()
    wall_clock_s = time.perf_counter() - wall_start

    corpus_metrics = compute_corpus_metrics(
        sample_results, all_references, all_hypotheses, wall_clock_s=wall_clock_s
    )
    eval_result = EvalResult(
        config=config,
        corpus_metrics=corpus_metrics,
        sample_results=sample_results,
    )

    # Output
    if not config.quiet:
        print(format_stdout(eval_result))
    paths = resolve_output_paths(config)
    if paths.root is not None:
        paths.root.mkdir(parents=True, exist_ok=True)
    if paths.json is not None:
        write_json(eval_result, paths.json)
    if paths.csv is not None:
        write_csv(eval_result, paths.csv)

    # Cleanup
    backend.cleanup()

    return eval_result

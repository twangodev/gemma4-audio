import itertools

from tqdm import tqdm

from gemma4_asr.backends import select_backend
from gemma4_asr.backends.base import InferenceBackend
from gemma4_asr.config import EvalConfig, EvalResult
from gemma4_asr.datasets import get_dataset
from gemma4_asr.datasets.base import Dataset
from gemma4_asr.metrics import compute_corpus_metrics, compute_sample_metrics
from gemma4_asr.output import format_stdout, write_csv, write_json


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
        dataset.load(config.split, seed=config.seed)

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

    total = config.limit if config.limit is not None else len(dataset)

    progress = tqdm(
        samples_iter,
        total=total,
        desc="Evaluating",
        disable=config.quiet,
        position=0,
        leave=True,
    )

    for sample in progress:
        result = backend.transcribe(sample.audio, sample.sample_rate, config.prompt)
        audio_duration = len(sample.audio) / sample.sample_rate

        sample_metric = compute_sample_metrics(
            id=sample.id,
            reference=sample.reference,
            hypothesis=result.text,
            latency_s=result.elapsed_seconds,
            audio_duration_s=audio_duration,
        )
        sample_results.append(sample_metric)
        all_references.append(sample.reference)
        all_hypotheses.append(result.text)

        if not config.quiet:
            running_wer = sum(s.wer for s in sample_results) / len(sample_results)
            progress.set_postfix(wer=f"{running_wer:.2%}")

    corpus_metrics = compute_corpus_metrics(sample_results, all_references, all_hypotheses)
    eval_result = EvalResult(
        config=config,
        corpus_metrics=corpus_metrics,
        sample_results=sample_results,
    )

    # Output
    if not config.quiet:
        print(format_stdout(eval_result))
    if config.output_json:
        write_json(eval_result, config.output_json)
    if config.output_csv:
        write_csv(eval_result, config.output_csv)

    # Cleanup
    backend.cleanup()

    return eval_result

import jiwer
import numpy as np
from whisper_normalizer.english import EnglishTextNormalizer

from gemma4_asr.config import AudioDurationStats, CorpusMetrics, LatencyStats, SampleResult

_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    """Normalize text using Whisper's English normalizer."""
    return _normalizer(text)


def compute_sample_metrics(
    id: str,
    reference: str,
    hypothesis: str,
    latency_s: float,
    audio_duration_s: float,
) -> SampleResult:
    """Compute per-sample ASR metrics."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    # Handle empty reference edge case
    if not ref_norm.strip():
        return SampleResult(
            id=id, reference=reference, hypothesis=hypothesis,
            wer=0.0 if not hyp_norm.strip() else 1.0,
            cer=0.0 if not hyp_norm.strip() else 1.0,
            mer=0.0 if not hyp_norm.strip() else 1.0,
            wil=0.0 if not hyp_norm.strip() else 1.0,
            substitutions=0, insertions=len(hyp_norm.split()) if hyp_norm.strip() else 0,
            deletions=0, latency_s=latency_s,
            rtfx=audio_duration_s / latency_s if latency_s > 0 else 0.0,
            audio_duration_s=audio_duration_s,
        )

    word_output = jiwer.process_words(ref_norm, hyp_norm)
    char_output = jiwer.process_characters(ref_norm, hyp_norm)

    rtfx = audio_duration_s / latency_s if latency_s > 0 else 0.0

    return SampleResult(
        id=id,
        reference=reference,
        hypothesis=hypothesis,
        wer=word_output.wer,
        cer=char_output.cer,
        mer=word_output.mer,
        wil=word_output.wil,
        substitutions=word_output.substitutions,
        insertions=word_output.insertions,
        deletions=word_output.deletions,
        latency_s=latency_s,
        rtfx=rtfx,
        audio_duration_s=audio_duration_s,
    )


def compute_corpus_metrics(
    sample_results: list[SampleResult],
    all_references: list[str],
    all_hypotheses: list[str],
) -> CorpusMetrics:
    """Compute corpus-level aggregated metrics."""
    refs_norm = [normalize_text(r) for r in all_references]
    hyps_norm = [normalize_text(h) for h in all_hypotheses]

    word_output = jiwer.process_words(refs_norm, hyps_norm)
    char_output = jiwer.process_characters(refs_norm, hyps_norm)

    # Total reference words = substitutions + deletions + hits (correct matches)
    n = max(word_output.substitutions + word_output.deletions + word_output.hits, 1)

    latencies = np.array([s.latency_s for s in sample_results])
    rtfxs = np.array([s.rtfx for s in sample_results])
    durations = np.array([s.audio_duration_s for s in sample_results])

    return CorpusMetrics(
        wer=word_output.wer,
        cer=char_output.cer,
        mer=word_output.mer,
        wil=word_output.wil,
        substitution_rate=word_output.substitutions / n,
        insertion_rate=word_output.insertions / n,
        deletion_rate=word_output.deletions / n,
        latency=LatencyStats(
            mean=float(np.mean(latencies)),
            p50=float(np.percentile(latencies, 50)),
            p95=float(np.percentile(latencies, 95)),
        ),
        rtfx=LatencyStats(
            mean=float(np.mean(rtfxs)),
            p50=float(np.percentile(rtfxs, 50)),
            p95=float(np.percentile(rtfxs, 95)),
        ),
        audio_duration=AudioDurationStats(
            mean=float(np.mean(durations)),
            min=float(np.min(durations)),
            max=float(np.max(durations)),
            p50=float(np.percentile(durations, 50)),
            total=float(np.sum(durations)),
        ),
        num_samples=len(sample_results),
    )

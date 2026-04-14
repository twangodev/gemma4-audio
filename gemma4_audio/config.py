from dataclasses import dataclass, field

DEFAULT_PROMPT = (
    "Transcribe the following speech segment in its original language.\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 "
    "and not one point seven, and write 3 instead of three."
)


@dataclass(frozen=True)
class EvalConfig:
    model: str
    dataset: str = "librispeech"
    split: str = "test.clean"
    backend: str = "auto"
    quantization: str | None = None
    limit: int | None = None
    seed: int = 42
    output_json: str | None = None
    output_csv: str | None = None
    quiet: bool = False
    streaming: bool = False
    prompt: str = field(default=DEFAULT_PROMPT)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    elapsed_seconds: float
    tokens_generated: int


@dataclass(frozen=True)
class SampleResult:
    id: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    mer: float
    wil: float
    substitutions: int
    insertions: int
    deletions: int
    latency_s: float
    rtfx: float
    audio_duration_s: float


@dataclass(frozen=True)
class LatencyStats:
    mean: float
    p50: float
    p95: float


@dataclass(frozen=True)
class AudioDurationStats:
    mean: float
    min: float
    max: float
    p50: float
    total: float


@dataclass(frozen=True)
class CorpusMetrics:
    wer: float
    cer: float
    mer: float
    wil: float
    substitution_rate: float
    insertion_rate: float
    deletion_rate: float
    latency: LatencyStats
    rtfx: LatencyStats
    audio_duration: AudioDurationStats
    num_samples: int
    bleu: float | None = None


@dataclass(frozen=True)
class EvalResult:
    config: EvalConfig
    corpus_metrics: CorpusMetrics
    sample_results: list[SampleResult]

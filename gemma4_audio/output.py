import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gemma4_audio.config import EvalConfig, EvalResult


@dataclass(frozen=True)
class OutputPaths:
    """Resolved on-disk locations for a single eval run's artifacts."""

    root: Path | None
    json: Path | None
    csv: Path | None


def _run_slug(config: EvalConfig) -> str:
    """Stable per-run directory name: {model}__{dataset}_{split} with / -> _."""
    return f"{config.model.replace('/', '_')}__{config.dataset}_{config.split}"


def resolve_output_paths(config: EvalConfig) -> OutputPaths:
    """Compute artifact paths for a config.

    Precedence: explicit --output-json / --output-csv override the auto-derived
    paths. When output_dir is set (default), artifacts land in
    {output_dir}/{slug}/results.{json,csv}. Setting output_dir to an empty
    string / None disables auto-derivation.
    """
    root = Path(config.output_dir) / _run_slug(config) if config.output_dir else None

    def _resolve(explicit: str | None, default_name: str) -> Path | None:
        if explicit:
            return Path(explicit)
        return root / default_name if root is not None else None

    return OutputPaths(
        root=root,
        json=_resolve(config.output_json, "results.json"),
        csv=_resolve(config.output_csv, "results.csv"),
    )


def format_stdout(result: EvalResult) -> str:
    """Format eval results as a human-readable string."""
    c = result.config
    m = result.corpus_metrics
    lines = [
        f"=== Gemma 4 ASR Eval: {c.model} ===",
        f"Dataset:      {c.dataset} / {c.split}",
        f"Backend:      {c.backend}",
        f"Quantization: {c.quantization or 'none'}",
        f"Samples:      {m.num_samples}",
        "",
        "--- Corpus Metrics ---",
        f"WER:    {m.wer:.2%}",
        f"CER:    {m.cer:.2%}",
        f"MER:    {m.mer:.2%}",
        f"WIL:    {m.wil:.2%}",
        f"Sub:    {m.substitution_rate:.2%}  "
        f"Ins: {m.insertion_rate:.2%}  "
        f"Del: {m.deletion_rate:.2%}",
        "",
        "--- Audio Duration ---",
        f"Mean:   {m.audio_duration.mean:.1f}s   "
        f"Min: {m.audio_duration.min:.1f}s   "
        f"Max: {m.audio_duration.max:.1f}s",
        f"P50:    {m.audio_duration.p50:.1f}s   "
        f"Total: {m.audio_duration.total:.0f}s ({m.audio_duration.total / 60:.1f}m)",
        "",
        "--- Latency ---",
        f"Mean:   {m.latency.mean:.2f}s   RTFx: {m.rtfx.mean:.1f}x",
        f"P50:    {m.latency.p50:.2f}s   P95:  {m.latency.p95:.2f}s",
    ]
    if m.bleu is not None:
        lines.extend(
            [
                "",
                "--- BLEU ---",
                f"BLEU:   {m.bleu:.1f}",
            ]
        )
    return "\n".join(lines)


def write_json(result: EvalResult, path: str | Path) -> None:
    """Write full eval results to a JSON file."""
    data = asdict(result)
    # Remove numpy arrays that can't be serialized
    for sample in data.get("sample_results", []):
        sample.pop("audio", None)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def write_csv(result: EvalResult, path: str | Path) -> None:
    """Write per-sample results to a CSV file."""
    if not result.sample_results:
        return
    fieldnames = [
        "id",
        "reference",
        "hypothesis",
        "wer",
        "cer",
        "mer",
        "wil",
        "substitutions",
        "insertions",
        "deletions",
        "latency_s",
        "rtfx",
        "audio_duration_s",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in result.sample_results:
            writer.writerow(
                {
                    "id": sample.id,
                    "reference": sample.reference,
                    "hypothesis": sample.hypothesis,
                    "wer": sample.wer,
                    "cer": sample.cer,
                    "mer": sample.mer,
                    "wil": sample.wil,
                    "substitutions": sample.substitutions,
                    "insertions": sample.insertions,
                    "deletions": sample.deletions,
                    "latency_s": sample.latency_s,
                    "rtfx": sample.rtfx,
                    "audio_duration_s": sample.audio_duration_s,
                }
            )

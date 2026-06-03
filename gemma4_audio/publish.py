"""Prepare eval results for web redistribution.

Produces, for each redistributable run under ``results_dir``:
  - ``results.json`` with ``sample_results`` emptied (config + corpus metrics
    only) so no upstream reference/hypothesis text is republished;
  - ``results.csv`` with the ``reference``/``hypothesis`` columns dropped
    (per-utterance metrics retained).

Only datasets cleared for redistribution are copied. SPGISpeech (Kensho
research-only) and TED-LIUM (CC BY-NC-ND) are excluded — their aggregate
metrics may still feed charts, but their per-sample files are not shared.
"""

import csv
import json
from pathlib import Path

REDISTRIBUTABLE_DATASETS = frozenset(
    {"ami", "earnings22", "gigaspeech", "librispeech", "voxpopuli"}
)

_DROP_COLUMNS = ("reference", "hypothesis")


def _strip_csv(src: Path, dst: Path) -> None:
    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        keep = [c for c in (reader.fieldnames or []) if c not in _DROP_COLUMNS]
        rows = [{c: row[c] for c in keep} for row in reader]
    with open(dst, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keep)
        writer.writeheader()
        writer.writerows(rows)


def publish(results_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Strip + filter every run under ``results_dir`` into ``out_dir``.

    Returns the list of published run directories.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    published: list[Path] = []

    for json_path in sorted(results_dir.glob("*/results.json")):
        data = json.loads(json_path.read_text())
        dataset = (data.get("config") or {}).get("dataset")
        if dataset not in REDISTRIBUTABLE_DATASETS:
            continue

        run_dir = json_path.parent
        dest = out_dir / run_dir.name
        dest.mkdir(parents=True, exist_ok=True)

        stripped = {**data, "sample_results": []}
        (dest / "results.json").write_text(json.dumps(stripped, indent=2))

        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            _strip_csv(csv_path, dest / "results.csv")

        published.append(dest)

    return published

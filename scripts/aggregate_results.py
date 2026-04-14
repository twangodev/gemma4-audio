#!/usr/bin/env python3
"""Aggregate g4 eval JSON results into a markdown summary for GitHub Actions.

Reads every ``*/results.json`` under the given directory, pivots on
(dataset, model), and prints markdown tables suitable for appending to
``$GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

EMPTY_CELL = "—"


def load_results(root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(root.glob("*/results.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config = data.get("config") or {}
        metrics = data.get("corpus_metrics") or {}
        if not config.get("model") or not metrics:
            continue
        results.append(
            {
                "model": config["model"],
                "dataset": f"{config.get('dataset', '?')}:{config.get('split', '?')}",
                "metrics": metrics,
            }
        )
    return results


def _render_table(
    title: str,
    results: list[dict[str, Any]],
    cell_fn: Callable[[dict[str, Any]], str],
) -> str:
    datasets = sorted({r["dataset"] for r in results})
    models = sorted({r["model"] for r in results})
    cells = {(r["dataset"], r["model"]): cell_fn(r["metrics"]) for r in results}
    short = {m: m.rsplit("/", 1)[-1] for m in models}

    lines = [f"### {title}", ""]
    lines.append("| Dataset | " + " | ".join(short[m] for m in models) + " |")
    lines.append("|---|" + "|".join(["---"] * len(models)) + "|")
    for d in datasets:
        row = [d] + [cells.get((d, m), EMPTY_CELL) for m in models]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def build_summary(root: Path) -> str:
    results = load_results(root)
    if not results:
        return "## ASR Benchmark Results\n\n_No results found._\n"

    sections = [
        "## ASR Benchmark Results",
        "",
        _render_table(
            "WER (lower is better)",
            results,
            lambda m: f"{m['wer']:.2%}",
        ),
        _render_table(
            "RTFx mean (higher is better)",
            results,
            lambda m: f"{m['rtfx']['mean']:.2f}x",
        ),
        _render_table(
            "Samples evaluated",
            results,
            lambda m: str(m["num_samples"]),
        ),
    ]
    return "\n".join(sections)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: aggregate_results.py <artifacts-dir>", file=sys.stderr)
        return 1
    print(build_summary(Path(sys.argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
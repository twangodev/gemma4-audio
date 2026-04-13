import argparse

from gemma4_audio.config import DEFAULT_PROMPT, EvalConfig


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    """Parse CLI arguments and return an EvalConfig."""
    parser = argparse.ArgumentParser(
        prog="g4",
        description="Gemma 4 audio toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Run ASR evaluation")
    eval_parser.add_argument("--model", required=True, help="HuggingFace model ID")
    eval_parser.add_argument("--dataset", default="librispeech", help="Dataset name")
    eval_parser.add_argument("--split", default="test-clean", help="Dataset split")
    eval_parser.add_argument(
        "--benchmark", default=None,
        help="Dataset:split shorthand (e.g. librispeech:test-clean). Overrides --dataset and --split.",
    )
    eval_parser.add_argument(
        "--backend", default="auto",
        choices=["auto", "vllm", "mlx", "transformers"],
        help="Inference backend",
    )
    eval_parser.add_argument(
        "--quantization", default=None,
        choices=["4bit", "8bit"],
        help="Quantization mode",
    )
    eval_parser.add_argument("--limit", type=int, default=None, help="Max samples to evaluate")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    eval_parser.add_argument("--output-json", default=None, help="Path to write JSON results")
    eval_parser.add_argument("--output-csv", default=None, help="Path to write CSV results")
    eval_parser.add_argument("--quiet", action="store_true", help="Suppress stdout output")
    eval_parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Transcription prompt")

    args = parser.parse_args(argv)

    dataset = args.dataset
    split = args.split
    if args.benchmark:
        parts = args.benchmark.split(":", 1)
        if len(parts) != 2:
            parser.error("--benchmark must be in dataset:split format (e.g. librispeech:test-clean)")
        dataset, split = parts

    return EvalConfig(
        model=args.model,
        dataset=dataset,
        split=split,
        backend=args.backend,
        quantization=args.quantization,
        limit=args.limit,
        seed=args.seed,
        output_json=args.output_json,
        output_csv=args.output_csv,
        quiet=args.quiet,
        prompt=args.prompt,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    from gemma4_audio.eval import run_eval
    run_eval(config)

# Batched ASR Benchmark + Gemma 4 12B + Publish Pipeline — Design

Date: 2026-06-03
Status: Blockers cleared & recipe validated (see "Validated findings"); pending user review of implementation plan

## Validated findings (from debugging the 12B)

The 12B (`gemma4_unified`) initially produced garbage WER (0.58). Root-caused to
two independent issues, both fixed and verified on librispeech test.clean (16):

1. **Reasoning thought-channel leak.** The 12B always emits an (empty) thought
   channel `<|channel>thought\n<channel|>` even with thinking disabled (per the
   model card; E2B/E4B are exempt). The `thought` label leaked into transcripts.
   Fix: `skip_special_tokens=False` + `vllm.reasoning.gemma4_utils.parse_thinking_output`.
2. **Message order.** Audio-first ordering destabilized the unified model
   (greedy → `<|channel>thought` repetition loops; sampling → chatty refusals).
   The model card's canonical audio example uses **text-then-audio**. Switching
   to text-first with plain greedy gives clean, deterministic output: **WER
   0.0336, beating E4B's 0.0417.** No sampling or repetition penalty needed.

**Validated recipe:** text-first message order + greedy (temperature 0) +
`skip_special_tokens=False` + `parse_thinking_output`.

3. **Batched-audio crash (upstream vllm day-1 bug, no issue filed).**
   `gemma4_unified._process_audio_input` assumed a single stacked tensor and
   called `.squeeze(1)`; a ragged batch arrives as a list of `[frames, dim]`
   tensors → `AttributeError`. Fix normalizes both shapes. **Verified: batched
   WER == single-sample WER (0.0336), 15/16 transcripts identical.** Will ship as
   an in-repo monkeypatch (applied when the vllm backend loads) so the fix is
   version-controlled and survives vllm reinstalls; remove when fixed upstream.

## Motivation

Gemma 4 12B (`google/gemma-4-12B-it`, released 2026-06-02) is the first medium
Gemma with native audio, via the new encoder-free `gemma4_unified` architecture.
We want it benchmarked on the Open ASR Leaderboard suite alongside the existing
E2B/E4B results.

Three coupled needs surfaced:

1. **12B support** — `gemma4_unified` is only in vllm `main` (commit `a248b45d`,
   PR #44429, merged 2026-06-03 19:01Z); no released/nightly wheel has it yet.
2. **Real batching** — the harness has *no* batching today (one `generate()` per
   sample). The published article calls current RTFx "a floor." We want true
   batched throughput so RTFx is meaningful and comparable across models.
3. **Apples-to-apples** — re-run E2B + E4B on the *same* nightly-vllm + batched
   stack as the 12B, so all three are comparable.

The real deliverable is the data landing in the twangodev site
(`static/data/gemma4-audio/eval_results/`), which a SvelteKit article renders.

## Environment (already established)

- vllm `0.1.dev1+ga248b45d0.precompiled`, built from commit `a248b45d` via
  `VLLM_USE_PRECOMPILED=1` (PR #44429 is pure-Python: model files + registry +
  config, no CUDA kernels — 29 commits ahead of the dev116 base). Registers
  `Gemma4UnifiedForConditionalGeneration` ✅.
- torch 2.11.0+cu130 (Blackwell RTX PRO 6000, 96 GB), transformers 5.10.1,
  torchcodec 0.11.0.
- Branch `chore/gemma4-12b-bench`. `pyproject.toml` `[vllm]` extra bumped; will
  pin vllm to the `a248b45d` commit for reproducibility.

## 1. Batched inference (harness change)

### Config / CLI
- Add `batch_size: int = 16` to `EvalConfig`; `--batch-size` CLI flag.

### Backend
- Add `transcribe_batch(items: list[BatchItem]) -> list[TranscriptionResult]` to
  `VLLMBackend`: build N chat prompts, submit a single
  `llm.generate([...], sampling_params)` so vllm continuous-batches them.
- Per-request latency comes from `RequestOutput.metrics` (finished − arrival).
- Keep single-sample `transcribe()` for the long-form chunking path and for the
  transformers/mlx backends (which stay serial). `InferenceBackend` base gets a
  default `transcribe_batch` that falls back to looping `transcribe`.

### Eval loop
- Accumulate up to `batch_size` non-chunked samples, flush via `transcribe_batch`.
- Long-form (chunked) samples still go one at a time.
- Wrap the whole dataset run with a wall-clock timer for throughput.

### Metrics (decision: throughput + per-request latency)
- Add headline **`throughput_rtfx: float`** to `CorpusMetrics` =
  `total_audio_seconds / total_wallclock_seconds`.
- Keep `rtfx` (mean/p50/p95) and `latency` (mean/p50/p95), now computed from
  per-request end-to-end times (these include queue wait under batching).
- **Additive schema change** — existing consumers that read `wer`, `rtfx.mean`
  keep working; the site's "speed" axis will be re-pointed at `throughput_rtfx`.

## 2. Benchmark run

For each model in {E2B, E4B, 12B} × each of the 8 datasets:
`g4 eval --model <m> --benchmark <ds> --backend vllm --batch-size 16 --seed 42`
(no quant, full test sets), overwriting `eval_results/<model>__<dataset>/`.
Single GPU → runs are sequential. ~hours (12B dominates).

## 3. `g4 publish` command

New subcommand: `g4 publish --results-dir eval_results --out <twangodev/static/data/...>`.
- Copies each `results.json` with `sample_results` emptied to `[]`.
- Copies each `results.csv` with `reference`/`hypothesis` columns dropped.
- Redistributes only the 6 permitted datasets (ami, earnings22, gigaspeech,
  librispeech test.clean/other, voxpopuli). Excludes spgispeech + tedlium per
  upstream licensing (their aggregates still feed charts via the JSON, which is
  config+corpus_metrics only — TBD confirm during impl whether excluded datasets'
  JSON aggregates are needed by charts and may be published without per-sample text).
- Tested (golden-file style).

## 4. Frontend (twangodev) — full update

- Publish stripped data for all 3 models.
- `EvalDataTable.svelte`: add a 12B column; widen the `model` union type.
- Charts (`WerHeatmap`, `WerRtfxScatter`, `ErrorCompositionChart`,
  `DurationHallucinationChart`, `StatCard`s): include 12B series; re-point speed
  axis to `throughput_rtfx`.
- `gemma4-asr-benchmark.svx`: rewrite for 3 models + landed batching (drop the
  "batching broken / RTFx is a floor / unbatched" framing), regenerate StatCard
  numbers and per-model analysis (sub-1s cliff, refusals, worst-sample) from the
  new batched results.

## Testing (TDD)
- Unit: `transcribe_batch` ordering/length; eval-loop batching incl. a remainder
  batch and a chunked-sample interleave; `throughput_rtfx` math; `g4 publish`
  strip/filter (golden files). Reuse the `integration` marker for GPU/data tests.

## Risks
- Day-1 vllm `gemma4_unified` audio path may have bugs → **gated by the 16-sample
  vllm smoke before any full run**.
- vllm pinned to an unreleased commit; documented + pinned by SHA. Revisit when an
  official wheel ships.
- Batched per-request latency conflates queue wait — documented; throughput_rtfx
  is the headline.

## Out of scope
- transformers/mlx batching (stay serial).
- CI matrix changes for the 12B (CPU CI can't run the new arch easily).

# Model Compression Demo Architecture

This repository is a runnable companion project for an LLM serving compression
guide. It helps compare and operationalize model compression choices for text
LLMs, especially `llm-compressor` GPTQ W4A16, RTN W8A16, and dynamic FP8
workflows served through vLLM.

The project is deliberately split into a dependency-light planning layer and
optional heavyweight execution paths. The default CLI, planner, recipe printer,
tests, and HTML smoke check run with the Python standard library plus dev tools.
GPU quantization, model loading, vLLM serving, and `lm_eval` checks import their
large ML dependencies only when those commands are actually executed.

![Architecture overview](architecture-overview.svg)

## What It Can Do

- List supported compression algorithms and quantization schemes.
- Estimate serving memory for an LLM from parameter count, architecture shape,
  context length, concurrency, KV-cache precision, and quantization scheme.
- Recommend practical GPU targets from built-in GPU instance metadata.
- Select a default algorithm from goal, hardware generation, and deployment
  runtime.
- Print copy-pasteable compression recipes for RTN W8A16, GPTQ W4A16, and
  dynamic FP8.
- Run or dry-run `llm-compressor` quantization jobs.
- Generate vLLM serving commands for compressed checkpoints.
- Compare base and compressed checkpoint directory sizes.
- Plan reproducible vLLM benchmark commands for quantized variants.
- Run local CUDA generation benchmarks across model, variant, and kernel axes.
- Write benchmark JSON plus self-contained HTML reports with inline SVG plots.
- Run or dry-run quality gates: generation comparison, perplexity delta,
  `lm_eval` task metrics, and a long-context anchor probe.
- Smoke-test the standalone `index.html` guide.

## Main Entry Points

- `demo.py` is the top-level executable wrapper.
- `compression-demo` and `model-compression-demo` are package scripts declared in
  `pyproject.toml`.
- `Makefile` wraps common development, planning, compression, quality, serving,
  and benchmark workflows.
- `index.html` is the standalone reader-facing selector and memory calculator.

Example commands:

```bash
uv run python demo.py list-algorithms
uv run python demo.py estimate --model-preset qwen3-8b --scheme w4a16
uv run python demo.py plan --model-preset qwen3-8b --goal fit-memory --hardware ampere
uv run python demo.py quantize --algorithm gptq-w4a16 --dry-run
uv run python demo.py quality-eval --base-model Qwen/Qwen3-8B --compressed-model outputs/Qwen3-8B-W4A16 --dry-run
uv run python demo.py gpu-benchmark --dry-run
```

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `compression_demo/cli.py` | Builds the argparse CLI and routes subcommands to planner, recipe, benchmark, quality, environment, and smoke-check handlers. |
| `compression_demo/catalog.py` | Defines supported quantization schemes, compression algorithms, GPU instance metadata, and hardware compute capability mappings. |
| `compression_demo/model_specs.py` | Provides built-in Qwen3 architecture presets and parses local Hugging Face `config.json` files into layer, hidden-size, and KV-head-ratio inputs. |
| `compression_demo/planner.py` | Estimates model weight memory, KV-cache memory, runtime overhead, safety buffer, compression job memory, and GPU recommendations. |
| `compression_demo/recipes.py` | Produces recipe snippets, default output directories, vLLM serve commands, calibration loading, dry-run summaries, and optional `llm-compressor` execution. |
| `compression_demo/benchmarks.py` | Generates dependency-light vLLM benchmark command plans and matching quality-eval commands. |
| `compression_demo/gpu_benchmarks.py` | Runs CUDA generation benchmarks, supports Transformers and vLLM paths, captures environment metadata, writes JSON, and renders HTML reports with inline SVG charts. |
| `compression_demo/evals.py` | Builds quality-evaluation plans, runs generation/perplexity/long-context/`lm_eval` comparisons, writes incremental JSON, and summarizes pass/fail/needs-review verdicts. |
| `index.html` | Standalone guide and browser calculator for algorithm, package, hardware, and memory selection. |
| `tests/` | Pytest coverage for CLI behavior, planner math, model-spec parsing, recipe snippets, quality summaries, GPU benchmark planning/reporting, and docs presence. |

## How Planning Works

The planner combines catalog metadata, model architecture, and serving shape into
a first-pass memory target:

```text
total GPU target =
  quantized weights
  + KV cache
  + runtime overhead
  + safety buffer
```

Weights use each scheme's effective average bits per parameter. That estimate
accounts for the fact that not every tensor is quantized; sensitive tensors such
as `lm_head` can remain higher precision.

KV cache is estimated from two tensors per token, layer count, hidden size,
context tokens, concurrent sequences, KV-cache precision, and KV-head ratio.
This is why long context and high concurrency can dominate memory even when
weights are aggressively quantized.

Runtime overhead is modeled conservatively as the larger of 2 GiB or 15 percent
of model weights. A safety buffer is then added as the larger of 1 GiB or 10
percent of the subtotal.

## How Algorithm Selection Works

`select_algorithm()` chooses pragmatic defaults:

- CPU and Apple targets use the low-risk RTN W8A16 path as a local-runtime
  planning proxy.
- Ada/Hopper/Blackwell throughput or latency goals prefer dynamic FP8.
- Memory-fit goals prefer GPTQ W4A16.
- Fine-tuning or QLoRA-like goals prefer RTN W8A16 in this demo.
- Quality-first goals prefer FP8 where the hardware supports it.

Explicit `--algorithm` input always overrides the default selection.

## How Quantization Works

The `quantize` command has a dry-run mode that prints the exact planned command,
selected algorithm, model, output directory, calibration source, sample count,
and sequence length.

When dry-run is disabled, `recipes.py` lazily imports the required ML stack and
runs one of three executable paths:

- `gptq-w4a16`: builds an `llmcompressor.modifiers.quantization.GPTQModifier`
  with `scheme="W4A16"`, targets Linear layers, ignores `lm_head`, and uses
  either a local calibration file or WikiText records.
- `rtn-w8a16`: builds a `QuantizationModifier` with `scheme="W8A16"` and writes a
  compressed checkpoint.
- `fp8-dynamic`: loads a Transformers causal LM and tokenizer, applies
  `FP8_DYNAMIC`, and saves with `save_compressed=True`.

Default output paths follow `outputs/<model>-<suffix>`, for example
`outputs/Qwen3-8B-W4A16`.

## How Quality And Benchmarking Work

Quality checks and benchmark reports are separate but connected parts of the
promotion workflow.

![Quality and benchmark flow](quality-benchmark-flow.svg)

Quality evaluation builds a plan first, then runs only the requested checks:

- Generation comparison loads base and compressed models and records
  deterministic responses for the same prompts.
- Perplexity comparison computes base and compressed perplexity on a dataset
  slice, usually WikiText.
- Long-context probing creates a synthetic prompt with an anchor fact and checks
  whether the compressed model can recover it.
- `lm_eval` runs task metrics such as HellaSwag for base and compressed models.

The summary verdict is:

- `pass` when configured gates hold and no warnings are raised.
- `fail` when perplexity, task regression, long-context retrieval, or
  subprocess return-code gates fail.
- `needs_review` when a check produced warning-level ambiguity.

Benchmarking has two levels:

- `benchmark-plan` emits commands for external vLLM serving benchmarks on target
  hardware.
- `gpu-benchmark` runs local CUDA generation measurements directly and writes
  JSON plus an HTML report.

The GPU benchmark runner supports Transformers kernels (`eager`, `sdpa`,
`sdpa-flash`, `sdpa-math`, `flash-attn-2`) for BF16/FP16 and bitsandbytes
variants, plus a vLLM subprocess path for BF16/FP16 and FP8 variants. Unsupported
variant/kernel combinations are recorded as skipped rather than silently ignored.

## Runtime Dependency Strategy

The project avoids declaring heavy CUDA packages as normal package extras. The
core package has no runtime dependencies, while `pyproject.toml` documents
runtime stacks under `tool.model-compression-demo.runtime-stacks`.

The Makefile keeps incompatible stacks isolated:

- `make venv` syncs the dev environment for tests and formatting.
- `make install-compression` installs the `llm-compressor` stack.
- `make install-serving` creates `.venv-vllm` and installs vLLM separately.
- `make install-gptqmodel` creates `.venv-gptqmodel` for GPTQModel experiments.

This separation keeps normal development fast while still preserving executable
GPU workflows for machines that have the right driver, CUDA, and model caches.

## Generated And Checked-In Artifacts

- `outputs/` is for compressed checkpoints and local run outputs.
- `reports/` holds benchmark and quality report artifacts.
- `sparse_logs/` contains local compression logs.
- `reports/rtx4090-fp8-comparison.json` and
  `reports/rtx4090-fp8-comparison.html` are checked-in benchmark fixtures
  described in [`reports.md`](reports.md).

## Validation

Use the standard development pipeline:

```bash
uv run ruff format --check .
uv run ruff check .
uv run python -m pytest
uv run python demo.py smoke-html
```

Or run the Makefile wrapper:

```bash
make check
```

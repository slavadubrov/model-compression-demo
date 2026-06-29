# LLM Serving Compression Demo and Selection Guide

This directory is a standalone support project for a blog article about model
compression and quantization approaches. It focuses on text LLM serving:
`llm-compressor` GPTQ W4A16 and FP8 paths, current industry alternatives, local
GGUF deployment, vLLM serving commands, and quality gates for deciding whether a
compressed checkpoint is safe to promote.

The runnable planner and tests use only the Python standard library. The actual
quantization paths are optional because this machine does not currently have
`torch`, `transformers`, `llmcompressor`, `vllm`, `bitsandbytes`, or `gptqmodel`
installed.

## What is included

- `demo.py`: CLI for algorithm listing, recipe generation, memory estimation,
  instance recommendations, size comparison, dry-run quantization, environment
  checks, benchmark command planning, local CUDA benchmark reporting, quality
  evaluation, and HTML smoke checks.
- `compression_demo/`: importable planner, algorithm catalog, quality evals,
  and recipe helpers.
- `examples/`: focused code examples for `llm-compressor`, bitsandbytes,
  GPTQModel, vLLM serving, representative calibration, and perplexity
  comparison.
- `index.html`: standalone guide for choosing algorithms, packages, and
  GPU-memory targets.
- `tests/`: pytest coverage for planner math, CLI behavior, quality evaluation,
  project metadata, and the HTML guide structure.

## Quick start

Architecture notes and code diagrams live in [`docs/architecture.md`](docs/architecture.md).
Checked-in report fixture provenance is documented in [`docs/reports.md`](docs/reports.md).

```bash
# List available algorithms and schemes
uv run python demo.py list-algorithms
uv run python demo.py list-schemes

# Print a compression recipe
uv run python demo.py recipe --algorithm gptq-w4a16
uv run python demo.py recipe --algorithm fp8-dynamic

# Estimate GPU memory for a configuration
uv run python demo.py estimate \
  --model-preset qwen3-8b --scheme w4a16 --context 4096 --concurrency 4

# Run the memory/instance planner
uv run python demo.py plan \
  --model-preset qwen3-8b --goal fit-memory --hardware ampere --context 4096 --concurrency 4

# Dry-run quantization (no GPU needed)
uv run python demo.py quantize --dry-run
uv run python demo.py quantize --algorithm fp8-dynamic --model Qwen/Qwen3-8B --dry-run

# Print a vLLM serve command
uv run python demo.py serve-command --algorithm fp8-dynamic --fp8-kv-cache --enable-prefix-caching

# Generate reproducible vLLM benchmark commands
uv run python demo.py benchmark-plan \
  --model Qwen/Qwen3-8B \
  --algorithms gptq-w4a16,rtn-w8a16,fp8-dynamic

# Run tests
uv run pytest
```

From the repository root:

```bash
uv run --project compression/model-compression-demo python demo.py plan \
  --params-b 7 \
  --goal fit-memory \
  --hardware ampere \
  --context 4096 \
  --concurrency 4
```

Open `index.html` in a browser for the guide and calculator.

## GPU benchmarking

The `gpu-benchmark` command runs local CUDA generation tests and writes both
machine-readable JSON and a self-contained HTML report with conclusions, tables,
and SVG plots. It compares BF16 baseline against FP8 dynamic quantization via
vLLM, measuring throughput, memory, and compression ratio.

The default models are `Qwen/Qwen3-8B` (primary) and `Qwen/Qwen3-0.6B` (small
comparison). Default variants are `bf16`, `fp8-dynamic` (W8A8 block-wise FP8),
and `fp8-dynamic-kv` (W8A8 + FP8 KV cache).

### Prerequisites

```bash
# Install vLLM in an isolated venv (required)
make install-serving
```

### Running benchmarks

**Quick smoke test** on a GPU with ≥ 8 GiB VRAM (BF16 only, PyTorch SDPA):

```bash
make gpu-bench
```

This benchmarks `Qwen3-8B` and `Qwen3-0.6B` with BF16 only. Expect ~15 min for
the first run (model download) and ~3 min for subsequent runs.

**Full FP8 benchmark** (vLLM, requires ≥ 24 GiB GPU, e.g., RTX 4090):

```bash
make gpu-bench-vllm
```

This benchmarks `Qwen/Qwen3-8B` with `bf16`, `fp8-dynamic`, and
`fp8-dynamic-kv` through vLLM's serving engine. Writes results to
`reports/rtx4090-qwen3-8b-fp8.json` and
`reports/rtx4090-qwen3-8b-fp8.html`.

**Compare two model sizes** (8B vs 0.6B):

```bash
uv run python demo.py gpu-benchmark \
  --models Qwen/Qwen3-8B,Qwen/Qwen3-0.6B \
  --variants bf16,fp8-dynamic,fp8-dynamic-kv \
  --kernels vllm \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --repeat-runs 3 \
  --vllm-max-model-len 2048 \
  --vllm-gpu-memory-utilization 0.92 \
  --output-json reports/comparison.json \
  --report-html reports/comparison.html
```

**Custom benchmark** (any model, any configuration):

```bash
uv run python demo.py gpu-benchmark \
  --models meta-llama/Llama-3.1-8B \
  --variants bf16,fp8-dynamic \
  --kernels vllm \
  --max-new-tokens 128 \
  --warmup-runs 2 \
  --repeat-runs 5 \
  --vllm-max-model-len 4096 \
  --vllm-gpu-memory-utilization 0.85 \
  --output-json reports/llama3-fp8.json
```

**Preview the plan** (no GPU needed):

```bash
uv run python demo.py gpu-benchmark --dry-run
```

### Interpreting results

The HTML report includes:

- **Throughput table**: token/s per variant, with BF16 baseline.
- **Memory table**: model memory and total GPU memory delta per variant.
- **Compression ratio**: BF16 model memory ÷ variant model memory when available.
- **SVG plots**: speed, model-memory, total-GPU-delta, and compression comparisons.

The JSON output contains the full run data for further analysis.

> **Note on memory measurement**: For vLLM rows, `model_memory_gib` is parsed from
> vLLM's model-loading log (`Model loading took ... GiB`). `gpu_memory_delta_gib`
> is the `nvidia-smi` used-memory delta, which includes model weights, CUDA
> graphs, runtime workspaces, and vLLM's KV cache block pool. With high
> `--vllm-gpu-memory-utilization`, the KV cache intentionally fills most of the
> remaining GPU space, so total GPU delta can look almost identical across BF16
> and FP8 even when the model weights are smaller. Compression ratios use model
> memory first and fall back to allocator/total GPU memory only when model memory
> is unavailable.
>
> **Note on small models**: FP8 is not automatically faster. For small models or
> very short generations, online FP8 quantization and FP8 KV-cache conversion
> overhead can be larger than the memory-bandwidth savings. Increase
> `--max-new-tokens`, `--repeat-runs`, and prompt batch size before drawing a
> steady-state throughput conclusion.

## Model compression (quantization)

### Installation

Install the compression packages in the project venv. Keep vLLM in its own
isolated environment (`make install-serving`) — current vLLM and llm-compressor
releases require incompatible dependency stacks.

```bash
# Install llm-compressor and dependencies
make install-compression

# Install vLLM in a separate venv (for serving after compression)
make install-serving
```

### Quick compression (dry-run first)

Preview the command without running it:

```bash
uv run python demo.py quantize \
  --algorithm gptq-w4a16 \
  --model Qwen/Qwen3-8B \
  --dry-run
```

### Full compression run

```bash
uv run python demo.py quantize \
  --algorithm gptq-w4a16 \
  --model Qwen/Qwen3-8B \
  --output-dir outputs/Qwen3-8B-W4A16 \
  --calibration-file examples/representative_calibration.jsonl \
  --text-column text \
  --num-calibration-samples 256 \
  --max-seq-length 4096
```

For a smoke test, omit `--calibration-file` to use WikiText. For a real
checkpoint, pass a JSONL or text file that matches production traffic:

```bash
uv run python demo.py quantize \
  --algorithm fp8-dynamic \
  --model Qwen/Qwen3-8B \
  --output-dir outputs/Qwen3-8B-FP8-Dynamic \
  --calibration-file traces/rag_queries.jsonl \
  --text-column text
```

### Supported compression algorithms

| Algorithm | Package | Scheme | Output format |
|-----------|---------|--------|---------------|
| `gptq-w4a16` | llm-compressor | GPTQ W4A16 | compressed-tensors |
| `rtn-w8a16` | llm-compressor | RTN W8A16 | compressed-tensors |
| `fp8-dynamic` | llm-compressor | FP8 Dynamic W8A8 | compressed-tensors |

List all available:

```bash
uv run python demo.py list-algorithms
```

### Serving after compression

Generate a vLLM serve command for the compressed checkpoint:

```bash
uv run python demo.py serve-command \
  --algorithm gptq-w4a16 \
  --model-path outputs/Qwen3-8B-W4A16 \
  --tensor-parallel-size 1 \
  --port 8000 \
  --enable-prefix-caching
```

The output is a ready-to-run shell command:

```bash
vllm serve outputs/Qwen3-8B-W4A16 --max-model-len 4096
```

### Quality evaluation

After compression, validate the checkpoint before promoting to production. Install
quality-eval dependencies with `make install-quality`; `make quality-eval` runs
that sync before executing the gate.

```bash
uv run python demo.py quality-eval \
  --base-model Qwen/Qwen3-8B \
  --compressed-model outputs/Qwen3-8B-W4A16 \
  --mode all \
  --lm-eval-task hellaswag \
  --lm-eval-limit 50 \
  --max-perplexity-delta-pct 5 \
  --output-json reports/qwen3-8b-w4a16-quality.json
```

Dry-run first on machines without the full GPU stack:

```bash
uv run python demo.py quality-eval \
  --base-model Qwen/Qwen3-8B \
  --compressed-model outputs/Qwen3-8B-W4A16 \
  --lm-eval-task hellaswag \
  --dry-run
```

The JSON report contains a compact `summary.verdict` of `pass`, `fail`, or
`needs_review`.

Checks implemented:
- Generation comparison: side-by-side deterministic responses.
- Perplexity comparison: WikiText perplexity delta.
- Task metrics: `lm_eval` task runs (e.g., `hellaswag`).
- Long-context anchor probe: synthetic retrieval check.

## Benchmark planning workflow

The benchmark planner generates commands for a real vLLM benchmark run. Run the
generated commands on the target serving hardware.

```bash
uv run python demo.py benchmark-plan \
  --model Qwen/Qwen3-8B \
  --algorithms gptq-w4a16,rtn-w8a16,fp8-dynamic \
  --dataset-name sharegpt \
  --num-prompts 200 \
  --input-len 1024 \
  --output-len 256 \
  --output-json reports/quantization-benchmark-plan.json
```

Each row includes:
- `serve_command`: vLLM serving command for the quantized variant.
- `bench_command`: matching `vllm bench serve` command.
- `quality_eval_command`: quality-gate command to run before promoting.

For a quick one-liner, run `make serve-bench-plan`.

## Makefile recipes

```bash
make help             # List available recipes
make venv             # uv sync --group dev --inexact
make format           # Apply Ruff formatting
make lint             # Run Ruff lint checks
make test             # Run pytest suite
make check            # format-check + lint + test + smoke-html
make dev              # format + lint + test + smoke-check
make clean            # Remove .venv and caches
make smoke-html       # Validate HTML guide structure
make dry-run-all      # Run article-support dry-run pipeline

# GPU benchmarks
make gpu-bench        # Quick smoke test (BF16, PyTorch SDPA)
make gpu-bench-vllm   # Full FP8 benchmark (vLLM, Qwen3-8B, RTX 4090)

# Compression and quality
make install-quality       # Install quality-eval dependencies
make quality-eval          # Run the quality gate
make quality-eval-plan     # Dry-run the quality gate plan
make install-compression   # Install llm-compressor + dependencies
make install-serving       # Install vLLM in .venv-vllm
make install-gptqmodel     # Install GPTQModel isolated

# Custom runs
make run ARGS="<args>"                     # Run demo.py with arbitrary args
make plan PARAMS_B=13 HARDWARE=hopper      # Memory planner with vars
make recipe ALGORITHM=fp8-dynamic          # Print a recipe

# Variables accepted by most recipes
make plan PARAMS_B=13 HARDWARE=hopper CONTEXT=8192 CONCURRENCY=8
make gpu-bench \
  GPU_MODELS=Qwen/Qwen3-8B,Qwen/Qwen3-0.6B \
  GPU_VARIANTS=bf16,fp8-dynamic \
  GPU_KERNELS=vllm
```

## Model architecture inputs

The memory planner needs layer count, hidden size, and KV-head ratio. Start
with built-in presets:

```bash
uv run python demo.py plan --model-preset qwen3-8b --hardware ada --goal throughput
uv run python demo.py plan --model-preset qwen3-0.6b --hardware cpu
```

For a model not listed, pass a local Hugging Face `config.json`:

```bash
uv run python demo.py plan --params-b 13 --hf-config ./config.json --hardware hopper
```

## Practical algorithm defaults

| Use case | Algorithm | Package |
|----------|-----------|---------|
| Fit model into GPU memory | `gptq-w4a16` | llm-compressor |
| Throughput (Ada/Hopper) | `fp8-dynamic` | llm-compressor + vLLM |
| Activation quantization | `rtn-w8a16` or SmoothQuant | llm-compressor |
| Fast experiments, QLoRA | NF4 | bitsandbytes |
| CPU / Apple Silicon / edge | GGUF | llama.cpp / Ollama |
| GPTQ/AWQ across runtimes | GPTQ | GPTQModel |

## Memory model

```text
total GPU ~= quantized weights + KV cache + runtime overhead + safety buffer
```

For serving, KV cache often dominates when context length or concurrency rises.
For offline compression: `llm-compressor` can onload decoder layers one at a
time, but GPTQ-like methods also allocate auxiliary hessian memory for the
active layer, and CPU/disk must hold the source model.

Always validate with a real benchmark on the target serving engine, model
family, prompt lengths, batch/concurrency shape, and quality metric.

## Formatting

```bash
uv sync --group dev
uv run ruff format .
uv run ruff check .
```

Heavyweight ML packages are intentionally not declared as project extras. `uv`
resolves optional dependencies while locking the project, so putting CUDA-only
packages such as `vllm` in `[project.optional-dependencies]` can break a normal
`uv sync --group dev` on local development machines.

## Article support references

This demo supports the compression and quantization article with:
- executable LLM serving workflows for RTN W8A16, GPTQ W4A16, and dynamic FP8.
- local-runtime guidance for GGUF CPU and Apple Silicon deployment.
- explicit recipe stubs for AutoRound, NVFP4/MXFP4, and SVDQuant/Nunchaku paths.
- current upstream package docs and repositories linked from the HTML guide.

The HTML guide is the reader-facing package, algorithm, and hardware selector.

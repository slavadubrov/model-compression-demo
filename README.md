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
  checks, quality evaluation, and HTML smoke checks.
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

From this directory:

```bash
uv run python demo.py list-algorithms
uv run python demo.py recipe --algorithm gptq-w4a16
uv run python demo.py estimate --model-preset llama3-8b --scheme w4a16 --context 4096 --concurrency 4
uv run python demo.py plan --model-preset llama3-8b --goal fit-memory --hardware ampere --context 4096 --concurrency 4
uv run python demo.py plan --model-preset llama3-8b --goal fit-memory --hardware apple
uv run python demo.py quantize --dry-run
uv run python demo.py quantize --calibration-file examples/representative_calibration.jsonl --dry-run
uv run python demo.py serve-command --algorithm fp8-dynamic --fp8-kv-cache --enable-prefix-caching
uv run python demo.py quality-eval --base-model Qwen/Qwen3-0.6B --compressed-model outputs/Qwen3-0.6B-W4A16 --dry-run
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

## Makefile recipes

The project includes a Makefile for the common local workflows:

```bash
make venv
make format
make lint
make test
make smoke-html
make check
make clean
make pipeline_dev
make pipeline_article
```

Available recipes:

- `make help`: list available recipes.
- `make venv`: run `uv sync --group dev`.
- `make format`: apply Ruff formatting.
- `make format-check`: verify Ruff formatting without editing files.
- `make lint`: run Ruff lint checks.
- `make test`: run the pytest suite.
- `make smoke-html`: validate the HTML guide structure.
- `make check`: run `format-check`, `lint`, `test`, and `smoke-html`.
- `make clean`: remove the local `.venv` and generated caches.
- `make run ARGS="..."`: run arbitrary `demo.py` CLI arguments.
- `make run_<command> ARGS="..."`: run a named `demo.py` command; underscores
  become hyphens, so `make run_quality_eval ARGS="--help"` runs
  `demo.py quality-eval --help`.
- `make plan`: run the memory and instance planner.
- `make estimate`: estimate memory for `PARAMS_B` and `SCHEME`.
- `make recipe`: print the selected compression recipe.
- `make quantize-dry-run`: show the `llm-compressor` quantization plan.
- `make quality-eval-dry-run`: show the quality evaluation plan.
- `make pipeline_dev`: format, lint, test, and smoke-check the project.
- `make pipeline_article`: run the article-support dry-run pipeline commands.
- `make pipeline_quality`: run quality-eval dry-run plus HTML smoke check.
- `make install-compression`: install optional compression/eval packages.
- `make install-alternatives`: install optional alternatives such as GPTQModel.
- `make install-serving`: install `vllm`; use only on a supported CUDA/Linux
  serving stack.

Most recipes accept variables. Examples:

```bash
make plan PARAMS_B=13 HARDWARE=hopper CONTEXT=8192 CONCURRENCY=8
make recipe ALGORITHM=fp8-dynamic
make quality-eval-dry-run \
  BASE_MODEL=Qwen/Qwen3-0.6B \
  COMPRESSED_MODEL=outputs/Qwen3-0.6B-W4A16
make run_plan ARGS="--params-b 7 --goal throughput --hardware hopper"
```

## Full llm-compressor quantization path

Use this only in a CUDA-capable environment with enough disk, CPU memory, and GPU
memory:

```bash
uv sync --group dev
uv pip install \
  accelerate \
  compressed-tensors \
  datasets \
  llmcompressor \
  lm_eval \
  torch \
  "transformers>=4.52.1"

# Optional alternatives covered in the guide.
uv pip install bitsandbytes gptqmodel peft

# Optional serving runtime. Install this only on a supported CUDA/Linux stack.
uv pip install vllm

uv run python demo.py quantize \
  --algorithm gptq-w4a16 \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/Qwen3-0.6B-W4A16 \
  --calibration-file examples/representative_calibration.jsonl \
  --text-column text \
  --num-calibration-samples 256 \
  --max-seq-length 4096
```

For a smoke test, leaving out `--calibration-file` uses WikiText. For a real
checkpoint, pass a JSONL or text file that looks like production traffic:

```bash
uv run python demo.py quantize \
  --algorithm gptq-w4a16 \
  --model Qwen/Qwen3-0.6B \
  --calibration-file traces/rag_queries.jsonl \
  --text-column text
```

Good calibration sources include SQL assistant prompts, RAG traces, chat logs,
support conversations, internal coding prompts, or any corpus that matches the
shape and vocabulary of the deployed workload. Keep sensitive data out of the
file or redact it before using it for calibration.

Reference GPTQ recipe:

```python
import json

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

rows = [
    json.loads(line)
    for line in open("examples/representative_calibration.jsonl", encoding="utf-8")
    if line.strip()
]

recipe = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=["lm_head"],
)

oneshot(
    model="Qwen/Qwen3-0.6B",
    dataset=Dataset.from_list(rows),
    recipe=recipe,
    output_dir="outputs/Qwen3-0.6B-W4A16",
    max_seq_length=4096,
    num_calibration_samples=len(rows),
)
```

## Quality evaluation workflow

The guide recommends validating compression with generation samples,
perplexity, task metrics, and long-context behavior. The `quality-eval` command
implements that workflow:

```bash
uv run python demo.py quality-eval \
  --base-model Qwen/Qwen3-0.6B \
  --compressed-model outputs/Qwen3-0.6B-W4A16 \
  --mode all \
  --lm-eval-task hellaswag \
  --lm-eval-limit 50 \
  --max-perplexity-delta-pct 5 \
  --max-task-regression 0.02 \
  --output-json reports/qwen3-0.6b-w4a16-quality.json
```

Use `--dry-run` first on machines without the full GPU stack. The real eval path
imports `torch`, `transformers`, `datasets`, and `lm_eval` only when execution is
requested:

```bash
uv run python demo.py quality-eval \
  --base-model Qwen/Qwen3-0.6B \
  --compressed-model outputs/Qwen3-0.6B-W4A16 \
  --lm-eval-task hellaswag \
  --dry-run
```

The implemented checks are:

- generation comparison: side-by-side deterministic responses for representative
  prompts.
- perplexity comparison: WikiText-style perplexity delta against the full
  precision model.
- task metrics: `lm_eval` task runs, such as `hellaswag`, for base and
  compressed models.
- long-context anchor probe: synthetic long-context retrieval check to catch
  obvious cache/context regressions.

By default `--mode all` includes the small `hellaswag` task with limit `50`,
loads base and compressed models sequentially to reduce VRAM pressure, writes
partial JSON after each completed phase when `--output-json` is set, and exits
non-zero if a deployment gate fails. The JSON report contains a compact
`summary.verdict` of `pass`, `fail`, or `needs_review`.

## Model architecture inputs

The memory planner needs layer count, hidden size, and KV-head ratio. Beginners
can start with built-in presets:

```bash
uv run python demo.py plan --model-preset qwen3-0.6b --hardware ada --goal throughput
uv run python demo.py plan --model-preset llama3-8b --hardware cpu
```

For a model that is not listed, pass a local Hugging Face `config.json`:

```bash
uv run python demo.py plan --params-b 13 --hf-config ./config.json --hardware hopper
```

If neither `--model-preset` nor `--hf-config` is provided, the CLI prints a
warning that it is using generic 7B-style architecture assumptions.

## Formatting

`ruff` is configured as the formatter and included in the `dev` dependency
group:

```bash
uv sync --group dev
uv run ruff format .
uv run ruff check .
```

The heavyweight ML and serving packages are intentionally not declared as
project extras. `uv` resolves optional dependencies while locking the project, so
putting CUDA-only packages such as `vllm` in `[project.optional-dependencies]`
can break a normal `uv sync --group dev` on local development machines.

## Practical algorithm defaults

Use `llm-compressor` when the output should be a production checkpoint for vLLM
or another runtime that understands compressed-tensors metadata. Start with
GPTQ W4A16 for fitting a model into memory, FP8 dynamic on Ada/Hopper for
throughput, and SmoothQuant plus W8A8 when activation quantization matters.

Use bitsandbytes for fast experiments and low-memory fine-tuning. Its NF4 path is
especially common for QLoRA-style workflows, but it is not the cleanest way to
publish a vLLM production artifact.

Use GPTQModel when you want a current alternative for GPTQ/AWQ checkpoints across
Transformers, Optimum, PEFT, vLLM, and SGLang. Avoid starting new projects on
AutoAWQ or AutoGPTQ unless you are pinned to an older stack.

Use GGUF and llama.cpp/Ollama/MLX-LM for CPU, Apple Silicon, desktop, and edge
deployment.

## Memory model

The planner estimates:

```text
total GPU target ~= quantized weights + KV cache + runtime overhead + safety buffer
```

For serving, KV cache often dominates when context length or concurrency rises.
For offline compression, the relevant number is different: `llm-compressor` can
onload text decoder layers one at a time, but GPTQ-like methods also allocate
auxiliary hessian memory for the active layer, and CPU or disk must hold the
source model.

The output is a first-pass sizing guide. Always validate with a real benchmark on
the target serving engine, model family, prompt lengths, batch/concurrency shape,
and quality metric.

## Article support references

This demo supports the compression and quantization article with:

- executable LLM serving workflows for RTN W8A16, GPTQ W4A16, and dynamic FP8.
- local-runtime guidance for GGUF CPU and Apple Silicon deployment.
- explicit recipe stubs for AutoRound, NVFP4/MXFP4, and SVDQuant/Nunchaku
  paths that are article-relevant but not executable in this beginner repo.
- current upstream package docs and repositories linked from the HTML guide.

The HTML guide is the reader-facing package, algorithm, and hardware selector.

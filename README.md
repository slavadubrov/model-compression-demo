# Model Compression Demo and Selection Guide

This directory expands the L4 notebook into a reusable model-compression demo.
It keeps the original `llm-compressor` GPTQ W4A16 path, adds current industry
alternatives, and includes a planner for choosing an algorithm, package, and GPU
memory target.

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
  GPTQModel, vLLM serving, and perplexity comparison.
- `index.html`: standalone guide for choosing algorithms, packages, and
  GPU-memory targets.
- `tests/`: stdlib `unittest` coverage for planner math, CLI behavior, and the
  HTML guide structure.

## Quick start

From this directory:

```bash
uv run python demo.py list-algorithms
uv run python demo.py recipe --algorithm gptq-w4a16
uv run python demo.py estimate --params-b 7 --scheme w4a16 --context 4096 --concurrency 4
uv run python demo.py plan --params-b 7 --goal fit-memory --hardware ampere --context 4096 --concurrency 4
uv run python demo.py quantize --dry-run
uv run python demo.py quality-eval --base-model Qwen/Qwen3-0.6B --compressed-model outputs/Qwen3-0.6B-W4A16 --dry-run
uv run python -m unittest discover -s tests
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

## Full llm-compressor quantization path

Use this only in a CUDA-capable environment with enough disk, CPU memory, and GPU
memory:

```bash
uv sync --extra gpu --extra alternatives --group dev

uv run python demo.py quantize \
  --algorithm gptq-w4a16 \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/Qwen3-0.6B-W4A16 \
  --dataset wikitext \
  --dataset-config-name wikitext-2-raw-v1 \
  --num-calibration-samples 256 \
  --max-seq-length 4096
```

The GPTQ recipe mirrors `notebooks/L4/L4.ipynb`:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

recipe = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=["lm_head"],
)

oneshot(
    model="Qwen/Qwen3-0.6B",
    dataset="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    recipe=recipe,
    output_dir="outputs/Qwen3-0.6B-W4A16",
    max_seq_length=4096,
    num_calibration_samples=256,
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

## Formatting

`ruff` is configured as the formatter and included in the `dev` dependency
group:

```bash
uv sync --group dev
uv run ruff format .
uv run ruff check .
```

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

## Source inspiration

This demo is based on:

- `notebooks/L4/L4.ipynb`: GPTQ W4A16 recipe, size comparison, generation test,
  and perplexity comparison.
- `presentation/grouped_pdfs/04_compression_quantization.pdf`: local slides on
  quantization, memory reduction, W8A16/W8A8, INT8/FP8, INT4/FP4, latency,
  throughput, and benchmark tradeoffs.
- `build_pdfs.py`: slide grouping comments for the image-based PDF.

The HTML guide links to current upstream docs and repositories for the package
landscape.

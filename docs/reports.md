# Reports And Benchmark Artifacts

The repository can produce two report families:

- quality-evaluation JSON from `demo.py quality-eval`
- GPU benchmark JSON plus self-contained HTML from `demo.py gpu-benchmark`

The checked-in report fixture is an RTX 4090 vLLM FP8 comparison:

- `reports/rtx4090-fp8-comparison.json`
- `reports/rtx4090-fp8-comparison.html`

The HTML report is generated from the JSON payload and includes inline SVG bar
charts for throughput, model memory, total GPU memory delta, and compression
ratio. No external assets are needed to view it.

## Checked-In Fixture Provenance

The fixture was generated on `2026-06-29T14:49:40.084906+00:00` inside WSL2 on a
machine with:

- GPU: NVIDIA GeForce RTX 4090
- compute capability: 8.9
- GPU memory: 23.99 GiB reported by PyTorch
- Python: 3.11.11
- PyTorch: 2.11.0+cu130
- vLLM: 0.23.0
- benchmark kernel: `vllm`
- vLLM max model length: 2048
- vLLM GPU memory utilization: 0.92
- warmup runs: 1
- repeat runs: 1
- generated tokens per run: 64

Models and variants:

| Model | Variant | Tok/s | Model GiB | GPU Delta GiB | Compression vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3-8B | bf16 | 84.58 | 15.27 | 22.39 | 1.00x |
| Qwen/Qwen3-8B | fp8-dynamic | 122.24 | 8.80 | 22.97 | 1.74x |
| Qwen/Qwen3-8B | fp8-dynamic-kv | 49.36 | 8.80 | 22.89 | 1.74x |
| Qwen/Qwen3-0.6B | bf16 | 297.78 | 1.12 | 23.14 | 1.00x |
| Qwen/Qwen3-0.6B | fp8-dynamic | 348.78 | 0.72 | 22.91 | 1.56x |
| Qwen/Qwen3-0.6B | fp8-dynamic-kv | 210.29 | 0.72 | 22.97 | 1.56x |

This fixture is useful for validating the report writer and demonstrating the
shape of a real result. It should not be treated as a universal performance
claim. Throughput and memory behavior depend on model revision, prompt mix,
generation length, vLLM version, driver, CUDA runtime, GPU state, and concurrency
settings.

## JSON Structure

GPU benchmark JSON has four top-level sections:

| Key | Description |
| --- | --- |
| `environment` | Timestamp, Python/platform metadata, package versions, `nvidia-smi` output, and PyTorch CUDA device details when available. |
| `config` | Models, variants, kernels, prompts, token counts, output paths, vLLM settings, and skipped run planning. |
| `runs` | One row per model/variant/kernel combination with status, timings, token counts, memory measurements, compression ratio, preview text, and errors. |
| `summary` | Compact rollup: total/ok/skipped/failed counts plus fastest, lowest-memory, and best-compression runs. |

Quality-evaluation JSON is organized around the plan and enabled checks:

| Key | Description |
| --- | --- |
| `plan` | Base model, compressed model, mode, checks, prompts, dataset, limits, gates, and output path. |
| `generation` | Side-by-side deterministic responses for configured prompts. |
| `perplexity` | Base perplexity, compressed perplexity, absolute delta, and relative delta percent. |
| `long_context` | Anchor-probe target length, base/compressed anchor booleans, and model responses. |
| `lm_eval` | Subprocess command, return code, stdout, and stderr for base and compressed task runs. |
| `summary` | `pass`, `fail`, or `needs_review` with failures and warnings. |

## Memory Interpretation

The report separates two memory concepts:

- `model_memory_gib`: model weight footprint. For vLLM rows this is parsed from
  engine logs such as `Model loading took X GiB`.
- `gpu_memory_delta_gib`: total `nvidia-smi` used-memory delta during the run.
  This can include weights, CUDA graphs, runtime workspaces, KV-cache block pool,
  and other serving allocations.

Compression ratio uses model memory first, then falls back to allocator or total
GPU memory when model memory is unavailable. With high
`--vllm-gpu-memory-utilization`, vLLM intentionally reserves most remaining GPU
memory for cache blocks, so total GPU delta can look similar across BF16 and FP8
even when the model weights are much smaller.

## Regeneration Commands

Install the development environment:

```bash
uv sync --group dev
```

Install vLLM into the isolated serving environment:

```bash
make install-serving
```

Regenerate the checked-in RTX 4090-style comparison:

```bash
env PATH="$(pwd)/.venv-vllm/bin:$PATH" .venv-vllm/bin/python demo.py gpu-benchmark \
  --models Qwen/Qwen3-8B,Qwen/Qwen3-0.6B \
  --variants bf16,fp8-dynamic,fp8-dynamic-kv \
  --kernels vllm \
  --max-new-tokens 32 \
  --warmup-runs 1 \
  --repeat-runs 1 \
  --vllm-max-model-len 2048 \
  --vllm-gpu-memory-utilization 0.92 \
  --output-json reports/rtx4090-fp8-comparison.json \
  --report-html reports/rtx4090-fp8-comparison.html
```

The Makefile wrapper for the same workflow is:

```bash
make gpu-bench-vllm
```

Generate a lighter dry-run plan without downloading models or touching CUDA:

```bash
uv run python demo.py gpu-benchmark --dry-run
```

Generate benchmark commands for target hardware instead of running the local
CUDA harness:

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

Generate quality-evaluation JSON after producing a compressed checkpoint:

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

Use dry-run mode when the full ML stack is not installed:

```bash
uv run python demo.py quality-eval \
  --base-model Qwen/Qwen3-8B \
  --compressed-model outputs/Qwen3-8B-W4A16 \
  --lm-eval-task hellaswag \
  --dry-run
```

## Report Verification

After regenerating report artifacts, run:

```bash
uv run python -m pytest tests/test_gpu_benchmarks.py tests/test_project_metadata.py
uv run python demo.py smoke-html
```

The GPU benchmark tests validate report rendering with a small synthetic payload;
they do not require CUDA. The real `gpu-benchmark` command does require CUDA,
model downloads or cached models, and the selected runtime stack.

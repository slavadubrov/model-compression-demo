# Report Fixtures

The `reports/` directory contains checked-in example outputs from local demo runs. Treat these files as fixtures that show the JSON and HTML shapes produced by the CLI, not as universal benchmark claims.

## Regeneration Commands

After creating local compressed checkpoints under `outputs/`, refresh quality reports with commands shaped like these:

```bash
uv run python demo.py quality-eval \
  --base-model Qwen/Qwen3-0.6B \
  --compressed-model outputs/Qwen3-0.6B-W4A16 \
  --mode generation \
  --output-json reports/qwen3-0.6b-w4a16-generation.json

uv run python demo.py quality-eval \
  --base-model Qwen/Qwen3-0.6B \
  --compressed-model outputs/Qwen3-0.6B-W4A16 \
  --mode perplexity \
  --output-json reports/qwen3-0.6b-w4a16-perplexity.json
```

Refresh the GPU benchmark report on the target CUDA workstation with:

```bash
make gpu-benchmark
```

The benchmark report records package versions, CUDA visibility, GPU metadata, run status, throughput, memory, and compression-ratio summaries. Do not compare numbers across machines unless the driver, CUDA stack, model, prompt mix, and benchmark arguments match.
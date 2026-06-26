#!/usr/bin/env bash
set -euo pipefail

# Generate benchmark commands for the variants discussed in the article.
# This script does not run the GPU benchmark by itself. Review the generated
# serve and benchmark commands, then run them on the target vLLM host.

uv run python demo.py benchmark-plan \
  --model Qwen/Qwen2.5-32B-Instruct \
  --algorithms gptq-w4a16,awq-w4a16,bnb-nf4,gguf-q4 \
  --dataset-name sharegpt \
  --num-prompts 200 \
  --input-len 1024 \
  --output-len 256 \
  --output-json reports/quantization-benchmark-plan.json

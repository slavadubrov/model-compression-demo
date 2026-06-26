#!/usr/bin/env bash
set -euo pipefail

# Serve a full precision reference model.
vllm serve Qwen/Qwen3-0.6B \
  --dtype bfloat16 \
  --max-model-len 4096

# Serve the GPTQ W4A16 compressed checkpoint.
vllm serve ./outputs/Qwen3-0.6B-W4A16 \
  --max-model-len 4096

# Serve a dynamic FP8 checkpoint on Ada/Hopper-class hardware.
vllm serve ./outputs/Qwen3-0.6B-FP8-Dynamic \
  --quantization fp8 \
  --max-model-len 32768 \
  --enable-prefix-caching

# Serve FP8 weights with FP8 KV cache for long-context/high-concurrency workloads.
# Confirm exact flag names against your installed vLLM version.
vllm serve ./outputs/Qwen3-0.6B-FP8-Dynamic \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --enable-prefix-caching

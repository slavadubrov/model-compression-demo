# Model Compression Demo Feedback Report

Date: 2026-06-26

Scope: review of this repository as the supporting demo for
https://slavadubrov.github.io/review/posts/2026-06-19-model-quantization-guide/
and as an out-of-the-box instrument set for model compression planning,
quantization, serving, and evaluation.

## Executive Summary

The demo has a solid foundation: the dependency-light planner works, the
README is practical, the static HTML guide is useful, and the local quality
pipeline passes. The repo is already strongest as an educational selector plus
dry-run planner for `llm-compressor` GPTQ W4A16.

The main gaps are in the parts that make it an instrument rather than a guide.
The FP8 path is not aligned with the article recipe, the planner gives misleading
recommendations for CPU or Apple/GGUF workflows, and the evaluation command
collects signals without enforcing deployment guardrails. There are also
documentation drift issues and coverage gaps around representative calibration,
vLLM serving flags, image-model quantization, and newer low-precision formats.

## Verification Performed

- `UV_CACHE_DIR=.uv-cache uv run pytest` passed: 19 tests.
- `UV_CACHE_DIR=.uv-cache make check` passed: Ruff format check, Ruff lint,
  pytest, and HTML smoke check.
- `UV_CACHE_DIR=.uv-cache make pipeline_article` passed: planner, quantize
  dry-run, quality-eval dry-run, and HTML smoke check.
- The first sandboxed `uv run pytest` failed only because dependency download
  was blocked by DNS/network restrictions. The same command passed after network
  access was allowed.
- The full GPU compression stack was not executed in this review because this
  environment does not have the optional `torch`, `transformers`,
  `llmcompressor`, `vllm`, `bitsandbytes`, or `gptqmodel` stack installed.

## Findings

### P1. FP8 quantization can write to the wrong output path and may not save a compressed checkpoint

Evidence:

- `compression_demo/cli.py:145-150` uses one default output directory,
  `outputs/Qwen3-0.6B-W4A16`, for every `quantize` algorithm.
- `compression_demo/recipes.py:101-118` and `compression_demo/recipes.py:241-251`
  save the FP8 path with `save_pretrained(output_dir)` and do not pass
  `save_compressed=True`.
- Running
  `uv run python demo.py quantize --algorithm fp8-dynamic --dry-run` reports:

  ```text
  Algorithm: Dynamic FP8 W8A8
  Output:    outputs/Qwen3-0.6B-W4A16
  ```

Why it matters:

The article presents FP8 W8A8 as the production server default on Ada/Hopper
class hardware. A reader who follows the demo can accidentally place FP8 output
under the W4A16 path and may not get the compressed artifact format described by
the article.

Recommendation:

- Derive the default output directory from the selected algorithm, for example
  `outputs/Qwen3-0.6B-FP8-Dynamic` for `fp8-dynamic` and
  `outputs/Qwen3-0.6B-W8A16` for `rtn-w8a16`.
- Match the article FP8 recipe by using `device_map="auto"` where appropriate
  and saving with `save_compressed=True` if that is the intended
  `llm-compressor` artifact.
- Add CLI tests that dry-run every executable quantization algorithm and assert
  algorithm-specific output directories.
- Add a recipe test that checks the FP8 snippet contains the compressed-save
  call.

### P1. CPU and Apple/GGUF planning returns NVIDIA GPU recommendations

Evidence:

- `compression_demo/planner.py:160-173` filters out the `CPU / Apple Silicon`
  catalog row because it has `memory_gib = 0`.
- `compression_demo/planner.py:228-229` selects `gguf-q4` for CPU or Apple, but
  the later recommendation path still uses the GPU list.
- Running
  `uv run python demo.py plan --params-b 7 --goal fit-memory --hardware cpu`
  selects GGUF, then recommends NVIDIA T4, A10G, L4, and RTX 4090.

Why it matters:

The article explicitly separates CUDA server deployment from edge/local
deployment. For the local path, the demo currently chooses the right algorithm
but then tells the user to use server GPUs.

Recommendation:

- Split serving targets into GPU and local runtime families.
- For `hardware=cpu` and `hardware=apple`, report estimated RAM or unified
  memory instead of GPU instances.
- Suppress "Compression GPU" for GGUF/local conversion paths unless the selected
  tool actually uses a GPU.
- Add tests for `plan --hardware cpu` and `plan --hardware apple`.

### P1. Evaluation commands do not enforce deployment guardrails

Evidence:

- `compression_demo/evals.py:62-79` omits task metrics from `--mode all` unless
  `--lm-eval-task` is provided.
- The README quick start uses `quality-eval --dry-run` without
  `--lm-eval-task` (`README.md:31-38`), so the first advertised workflow skips
  task metrics.
- `compression_demo/evals.py:345-363` runs `lm_eval` with `check=False` and
  stores return codes, but `run_quality_eval` does not fail when either base or
  compressed evaluation returns non-zero.
- `compression_demo/evals.py:319-324` returns long-context booleans but does not
  turn them into a pass/fail result.

Why it matters:

The article's core production message is that a quantized model must be
validated with perplexity, downstream task metrics, reasoning checks, and
long-context behavior before promotion. The demo can currently exit
successfully while task evals are skipped or failed.

Recommendation:

- Make `--mode all` either require an `--lm-eval-task` or default to a lightweight
  task such as `hellaswag` with an explicit limit for smoke runs.
- Add threshold flags such as `--max-perplexity-delta-pct`,
  `--max-task-regression`, and `--require-long-context-anchor`.
- Fail with a non-zero exit code when `lm_eval` fails or when thresholds are
  breached.
- Write a compact summary verdict into the JSON report: `pass`, `fail`, or
  `needs_review`.

### P1. Calibration is too generic for the article's guidance

Evidence:

- The executable quantization CLI accepts only Hugging Face dataset identifiers:
  `--dataset` and `--dataset-config-name` (`compression_demo/cli.py:151-152`).
- The main example hardcodes WikiText calibration
  (`examples/llmcompressor_gptq_w4a16.py:22-29`).
- The README describes representative prompts for evaluation, but the
  compression path does not provide a first-class local calibration data input.

Why it matters:

The article warns that calibration data should reflect production traffic. A
demo that defaults to WikiText is fine for a smoke test, but an out-of-box
instrument should make representative calibration easy.

Recommendation:

- Add `--calibration-file` for JSONL or text files and `--text-column` for
  structured records.
- Add a small sample calibration JSONL under `examples/` with domain-specific
  prompts.
- Update dry-run output to say whether calibration is generic demo data or
  user-supplied representative data.
- Add documentation showing how to calibrate on SQL, RAG traces, chat logs, or
  other production-shaped corpora.

### P2. vLLM serving example does not include the article's FP8/KV-cache serving flags

Evidence:

- `examples/vllm_serve_commands.sh:13-15` serves the FP8 checkpoint with only
  `--max-model-len 4096`.
- The article's vLLM serving recipe enables FP8 quantization, FP8 KV cache, a
  long context window, and prefix caching.

Why it matters:

KV cache is one of the article's main themes. The demo's serving example does
not show the flags that make the KV-cache recommendation concrete.

Recommendation:

- Add an FP8 serving command that includes the relevant vLLM flags, for example
  `--quantization fp8`, `--kv-cache-dtype fp8`, `--max-model-len 32768`, and
  `--enable-prefix-caching`, with a version note if flag names vary.
- Add a CLI `serve-command` generator so the planner can produce the matching
  vLLM command for W4A16, FP8, and FP8 KV cache scenarios.

### P2. Python `>=3.14` is a risky floor for an out-of-box ML demo

Evidence:

- `pyproject.toml:10` requires Python `>=3.14`.
- `pyproject.toml:41-58` documents runtime stacks that include heavy ML and
  serving packages such as `torch`, `transformers`, `llmcompressor`,
  `bitsandbytes`, `gptqmodel`, and `vllm`.

Why it matters:

For an "out of the box" compression demo, the Python floor should match the
ecosystem that users are likely to run for CUDA and ML packages. Python 3.14 may
be fine for the dependency-light planner, but it is an unnecessarily narrow
entry point unless all optional stacks are verified on it.

Recommendation:

- Lower the project floor to the oldest version actually needed by the code,
  likely Python 3.11 or 3.12, unless there is a deliberate Python 3.14
  requirement.
- Keep the Ruff target aligned with that floor.
- Add a CI matrix for the supported Python versions, at least for the
  dependency-light tests.

### P2. README references files that are not present in this repo

Evidence:

- `README.md:254-262` references
  `presentation/grouped_pdfs/04_compression_quantization.pdf` and
  `build_pdfs.py`.
- Neither path exists in the reviewed repository.

Why it matters:

The article-support references section should be reliable. Missing files make
the repo look incomplete and can send readers looking for supporting material
that is not shipped.

Recommendation:

- Either add the referenced files or remove the bullets.
- If those files live in a parent article repository, link to the real location
  and state that they are external to this demo package.

### P2. The repo overstates coverage of some article topics

Evidence:

- `index.html:269-272` says `llm-compressor` covers AutoRound, KV-cache
  quantization, and newer FP4 paths, but the CLI has no executable command or
  recipe for AutoRound or FP4/NVFP4/MXFP4.
- `compression_demo/catalog.py` contains an `nvfp4-mxfp4` scheme, but there is
  no matching algorithm recipe or planner selection path.
- Searching the repo found no coverage for SVDQuant, Nunchaku, diffusion, or
  image-model quantization, even though the article has a dedicated image-model
  section.

Why it matters:

The demo is strongest for LLM server quantization. That is acceptable, but the
reader-facing text should not imply runnable coverage for article sections that
are not instrumented.

Recommendation:

- Rename the current scope explicitly to "LLM serving compression demo" or add a
  roadmap section for unsupported article topics.
- Add at least recipe stubs for AutoRound, NVFP4/MXFP4, and SVDQuant/Nunchaku,
  clearly marked as non-executable if they require specialized hardware or
  separate runtimes.
- Add tests that ensure every catalog algorithm has a recipe or an explicit
  unsupported status.

### P2. Planner memory defaults are generic and can mislead for real model families

Evidence:

- `compression_demo/cli.py:135-140` defaults to `layers=32`,
  `hidden-size=4096`, and `kv-head-ratio=1.0`.
- The planner does not accept `--model` to derive architecture from a Hugging
  Face config and does not provide model-family presets.

Why it matters:

The article emphasizes that KV cache memory depends on model architecture,
context, concurrency, and grouped-query attention. A 7B default is useful for
smoke checks but is not enough for reliable sizing of 13B, 70B, MoE, or
long-context models.

Recommendation:

- Add `--model` or `--hf-config` support to derive `num_hidden_layers`,
  `hidden_size`, and KV-head ratio.
- Add common presets for Qwen, Llama, Mistral, and MoE families.
- Print a warning when defaults are used: "Using generic 7B-style architecture
  assumptions."

### P3. Quality evaluation loads base and compressed models in the same process

Evidence:

- `compression_demo/evals.py:269-270` loads the base model and compressed model
  before calculating perplexity.
- `compare_generations` follows the same pattern by loading both models before
  iterating prompts.

Why it matters:

For small smoke models this is fine. For article-class deployments, loading both
models at once can exceed VRAM and make the evaluation tool unusable precisely
when users need it most.

Recommendation:

- Add a subprocess or sequential mode that evaluates one model at a time and
  frees memory before loading the next.
- Persist intermediate metrics to JSON so a failed second run does not lose the
  baseline result.

### P3. Dry-run output is a plan, not a reproducible command

Evidence:

- `compression_demo/recipes.py:171-198` prints configuration fields for
  quantization dry-runs but does not print the exact command to rerun.

Why it matters:

The repo's stated goal includes out-of-box instruments. Dry-run output becomes
more useful if users can copy the exact command and see the exact recipe that
will execute.

Recommendation:

- Include the exact `uv run python demo.py quantize ...` command in dry-run
  output.
- Optionally add `--emit-python-script` or `--emit-recipe` to materialize a
  runnable recipe for review before execution.

## Suggested Fix Order

1. Fix the FP8 path and algorithm-specific output directories.
2. Fix CPU/Apple/GGUF planning so local deployment recommendations are not GPU
   recommendations.
3. Turn `quality-eval` into a real guardrail command with task defaults,
   thresholds, and non-zero failure behavior.
4. Add first-class representative calibration input.
5. Update vLLM serving examples and add a command generator.
6. Remove missing README references and narrow or expand the stated article
   coverage.
7. Add model presets or config-derived KV sizing.


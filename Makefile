# ── Shared overrides ─────────────────────────────────
ALGORITHM ?= gptq-w4a16
MODEL     ?= Qwen/Qwen3-8B
ARGS      ?=

# ── Derived values (from MODEL and ALGORITHM) ────────
_MODEL_SLUG = $(subst /,-,$(notdir $(MODEL)))
_ALGO_SUFFIX = $(strip $(if $(filter gptq-w4a16,$(ALGORITHM)),W4A16,$(if $(filter rtn-w8a16,$(ALGORITHM)),W8A16,$(if $(filter fp8-dynamic,$(ALGORITHM)),FP8-Dynamic,COMPRESSED))))
COMPRESSED_DIR ?= outputs/$(_MODEL_SLUG)-$(_ALGO_SUFFIX)

.PHONY: help
help:
	@printf '%s\n' \
		'Model compression demo recipes:' \
		'  make venv                  Sync the uv dev environment.' \
		'  make format                Apply Ruff formatting.' \
		'  make format-check          Check Ruff formatting.' \
		'  make lint                  Run Ruff lint checks.' \
		'  make test                  Run pytest.' \
		'  make smoke-html            Validate the HTML guide structure.' \
		'  make check                 Run format-check, lint, test, smoke-html.' \
		'  make clean                 Remove the local venv and generated caches.' \
		'  make install-compression   Install llm-compressor compatible packages.' \
		'  make install-serving       Install vLLM into .venv-vllm.' \
		'  make install-gptqmodel     Install GPTQModel into .venv-gptqmodel.' \
		'' \
		'  make plan                  Run the memory/instance planner.' \
		'  make estimate              Estimate memory for a model and scheme.' \
		'  make recipe                Print the selected compression recipe.' \
		'  make list-algorithms       List compression algorithms.' \
		'  make list-schemes          List quantization schemes.' \
		'  make env                   Show optional ML dependency availability.' \
		'' \
		'  make quantize              Run llm-compressor quantization.' \
		'  make quantize-plan         Dry-run the quantization command.' \
		'  make serve-command         Print a vLLm serve command.' \
		'' \
		'  make quality-eval          Run quality evaluation.' \
		'  make quality-eval-plan     Dry-run quality evaluation.' \
		'' \
		'  make serve-bench-plan      Generate vLLM benchmark command plan.' \
		'  make gpu-bench             Run quick BF16 GPU smoke benchmark (PyTorch SDPA).' \
		'  make gpu-bench-vllm        Run Qwen3-8B BF16 vs FP8 benchmarks on GPU.' \
		'' \
		'  make dev                   Format, lint, test, and smoke-check.' \
		'  make dry-run-all           Run article-support dry-run pipeline.' \
		'  make run ARGS="..."        Run demo.py with arbitrary CLI args.'

# ── Development ──────────────────────────────────────

.PHONY: venv
venv:
	uv sync --group dev

.PHONY: clean
clean:
	rm -rf .venv .ruff_cache .pytest_cache .mypy_cache .uv-cache compression_demo/__pycache__ tests/__pycache__

.PHONY: format
format: venv
	uv run ruff format .

.PHONY: format-check
format-check: venv
	uv run ruff format --check .

.PHONY: lint
lint: venv
	uv run ruff check .

.PHONY: test
test: venv
	uv run python -m pytest

.PHONY: smoke-html
smoke-html: venv
	uv run python demo.py smoke-html

.PHONY: check
check: format-check lint test smoke-html

# ── Install runtimes ─────────────────────────────────

.PHONY: install-compression
install-compression:
	uv sync --group dev --group compression
	-uv pip uninstall torchvision torchaudio

.PHONY: install-serving
install-serving:
	uv venv --python 3.11.11 --clear .venv-vllm
	uv pip install --python .venv-vllm/bin/python "vllm==0.23.0"

.PHONY: install-gptqmodel
install-gptqmodel:
	uv venv --python 3.11.11 --clear .venv-gptqmodel
	uv pip install --python .venv-gptqmodel/bin/python "gptqmodel==7.1.0" torchvision

# ── Planning tools ───────────────────────────────────

.PHONY: plan
plan: venv
	uv run python demo.py plan --params-b 7 --goal fit-memory --hardware ampere --context 4096 --concurrency 4

.PHONY: estimate
estimate: venv
	uv run python demo.py estimate --params-b 7 --scheme w4a16 --context 4096 --concurrency 4

.PHONY: recipe
recipe: venv
	uv run python demo.py recipe --algorithm $(ALGORITHM)

.PHONY: list-algorithms
list-algorithms: venv
	uv run python demo.py list-algorithms

.PHONY: list-schemes
list-schemes: venv
	uv run python demo.py list-schemes

.PHONY: env
env: venv
	uv run python demo.py env

# ── Quantization ─────────────────────────────────────

.PHONY: quantize
quantize: venv
	uv run python demo.py quantize --algorithm $(ALGORITHM) --model $(MODEL) $(ARGS)

.PHONY: quantize-plan
quantize-plan: venv
	uv run python demo.py quantize --algorithm $(ALGORITHM) --model $(MODEL) --dry-run

# ── Quality evaluation ───────────────────────────────

.PHONY: quality-eval
quality-eval: venv
	uv run python demo.py quality-eval \
		--base-model $(MODEL) \
		--compressed-model $(COMPRESSED_DIR) \
		--mode all --lm-eval-task hellaswag \
		--output-json reports/$(_MODEL_SLUG)-$(_ALGO_SUFFIX)-quality.json

.PHONY: quality-eval-plan
quality-eval-plan: venv
	uv run python demo.py quality-eval \
		--base-model $(MODEL) \
		--compressed-model $(COMPRESSED_DIR) \
		--lm-eval-task hellaswag --dry-run

# ── Serving ──────────────────────────────────────────

.PHONY: serve-command
serve-command: venv
	uv run python demo.py serve-command --algorithm $(ALGORITHM)

# ── Benchmarking ─────────────────────────────────────

.PHONY: serve-bench-plan
serve-bench-plan: venv
	uv run python demo.py benchmark-plan \
		--model $(MODEL) \
		--algorithms gptq-w4a16,rtn-w8a16,fp8-dynamic \
		--output-json reports/benchmark-plan.json

.PHONY: gpu-bench
gpu-bench:
	uv run python demo.py gpu-benchmark \
		--models Qwen/Qwen3-8B,Qwen/Qwen3-0.6B \
		--variants bf16 --kernels sdpa,eager \
		--max-new-tokens 32 --warmup-runs 1 --repeat-runs 1 \
		--output-json reports/gpu-benchmark-results.json \
		--report-html reports/gpu-benchmark-report.html

.PHONY: gpu-bench-vllm
gpu-bench-vllm:
	env PATH="$$(pwd)/.venv-vllm/bin:$$PATH" .venv-vllm/bin/python demo.py gpu-benchmark \
		--models Qwen/Qwen3-8B,Qwen/Qwen3-0.6B \
		--variants bf16,fp8-dynamic,fp8-dynamic-kv --kernels vllm \
		--max-new-tokens 32 --warmup-runs 1 --repeat-runs 1 \
		--vllm-max-model-len 2048 --vllm-gpu-memory-utilization 0.92 \
		--output-json reports/rtx4090-fp8-comparison.json \
		--report-html reports/rtx4090-fp8-comparison.html

# ── Pipelines ────────────────────────────────────────

.PHONY: dev
dev: format lint test smoke-html

.PHONY: dry-run-all
dry-run-all: plan quantize-plan quality-eval-plan serve-bench-plan smoke-html

# ── Convenience run target ───────────────────────────

.PHONY: run run_%
run: venv
	uv run python demo.py $(ARGS)

run_%: venv
	uv run python demo.py $(subst _,-,$*) $(ARGS)

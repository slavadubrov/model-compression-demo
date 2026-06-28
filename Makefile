UV ?= uv
PYTHON ?= python
DEMO ?= demo.py

PARAMS_B ?= 7
SCHEME ?= w4a16
CONTEXT ?= 4096
CONCURRENCY ?= 4
HARDWARE ?= ampere
GOAL ?= fit-memory
ALGORITHM ?= gptq-w4a16
BASE_MODEL ?= Qwen/Qwen3-0.6B
COMPRESSED_MODEL ?= outputs/Qwen3-0.6B-W4A16
LM_EVAL_TASK ?= hellaswag
QUALITY_JSON ?= reports/qwen3-0.6b-w4a16-quality.json
BENCHMARK_MODEL ?= Qwen/Qwen2.5-32B-Instruct
BENCHMARK_ALGORITHMS ?= gptq-w4a16,awq-w4a16,bnb-nf4,gguf-q4
BENCHMARK_JSON ?= reports/quantization-benchmark-plan.json
GPU_BENCHMARK_MODELS ?= Qwen/Qwen3-0.6B
GPU_BENCHMARK_VARIANTS ?= bf16,bnb-int8,bnb-nf4
GPU_BENCHMARK_KERNELS ?= sdpa,eager
GPU_BENCHMARK_JSON ?= reports/gpu-benchmark-results.json
GPU_BENCHMARK_HTML ?= reports/gpu-benchmark-report.html
SERVING_ENV ?= .venv-vllm
GPTQMODEL_ENV ?= .venv-gptqmodel
SERVING_VLLM ?= vllm==0.23.0
GPTQMODEL_PACKAGE ?= gptqmodel==7.1.0 torchvision
ARGS ?=

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
		'  make run ARGS="..."        Run demo.py with arbitrary CLI args.' \
		'  make run_<command> ARGS="..." Run a demo.py command; underscores become hyphens.' \
		'  make plan                  Run the memory/instance planner.' \
		'  make estimate              Estimate memory for PARAMS_B and SCHEME.' \
		'  make recipe                Print the selected compression recipe.' \
		'  make quantize-dry-run      Show the llm-compressor quantization plan.' \
		'  make quality-eval-dry-run  Show the quality evaluation plan.' \
		'  make benchmark-plan        Generate vLLM benchmark commands.' \
		'  make gpu-benchmark         Run local CUDA benchmarks and HTML report.' \
		'  make pipeline_dev          Format, lint, test, and smoke-check the project.' \
		'  make pipeline_article      Run article-support dry-run pipeline commands.' \
		'  make install-compression   Install llm-compressor compatible packages.' \
		'  make install-alternatives  Install same-venv alternatives like bitsandbytes.' \
		'  make install-serving       Install vLLM into SERVING_ENV.' \
		'  make install-gptqmodel     Install GPTQModel into GPTQMODEL_ENV.'

.PHONY: venv
venv:
	$(UV) sync --group dev

.PHONY: clean
clean:
	rm -rf .venv .ruff_cache .pytest_cache .mypy_cache .uv-cache compression_demo/__pycache__ tests/__pycache__

.PHONY: install-compression
install-compression:
	$(UV) sync --group dev --group compression
	-$(UV) pip uninstall torchvision torchaudio

.PHONY: install-alternatives
install-alternatives: install-compression

.PHONY: install-serving
install-serving:
	$(UV) venv --python 3.11.11 --clear $(SERVING_ENV)
	$(UV) pip install --python $(SERVING_ENV)/bin/python $(SERVING_VLLM)

.PHONY: install-gptqmodel
install-gptqmodel:
	$(UV) venv --python 3.11.11 --clear $(GPTQMODEL_ENV)
	$(UV) pip install --python $(GPTQMODEL_ENV)/bin/python $(GPTQMODEL_PACKAGE)

.PHONY: format
format: venv
	$(UV) run ruff format .

.PHONY: format-check
format-check: venv
	$(UV) run ruff format --check .

.PHONY: lint
lint: venv
	$(UV) run ruff check .

.PHONY: test
test: venv
	$(UV) run $(PYTHON) -m pytest

.PHONY: smoke-html
smoke-html: venv
	$(UV) run $(PYTHON) $(DEMO) smoke-html

.PHONY: check
check: format-check lint test smoke-html

.PHONY: run
run: venv
	$(UV) run $(PYTHON) $(DEMO) $(ARGS)

.PHONY: run_%
run_%: venv
	$(UV) run $(PYTHON) $(DEMO) $(subst _,-,$*) $(ARGS)

.PHONY: plan
plan: venv
	$(UV) run $(PYTHON) $(DEMO) plan --params-b $(PARAMS_B) --goal $(GOAL) --hardware $(HARDWARE) --context $(CONTEXT) --concurrency $(CONCURRENCY)

.PHONY: estimate
estimate: venv
	$(UV) run $(PYTHON) $(DEMO) estimate --params-b $(PARAMS_B) --scheme $(SCHEME) --context $(CONTEXT) --concurrency $(CONCURRENCY)

.PHONY: recipe
recipe: venv
	$(UV) run $(PYTHON) $(DEMO) recipe --algorithm $(ALGORITHM)

.PHONY: quantize-dry-run
quantize-dry-run: venv
	$(UV) run $(PYTHON) $(DEMO) quantize --algorithm $(ALGORITHM) --dry-run

.PHONY: quantize
quantize: venv
	$(UV) run $(PYTHON) $(DEMO) quantize --algorithm $(ALGORITHM)

.PHONY: quality-eval-dry-run
quality-eval-dry-run: venv
	$(UV) run $(PYTHON) $(DEMO) quality-eval --base-model $(BASE_MODEL) --compressed-model $(COMPRESSED_MODEL) --lm-eval-task $(LM_EVAL_TASK) --dry-run

.PHONY: quality-eval
quality-eval: venv
	$(UV) run $(PYTHON) $(DEMO) quality-eval --base-model $(BASE_MODEL) --compressed-model $(COMPRESSED_MODEL) --mode all --lm-eval-task $(LM_EVAL_TASK) --output-json $(QUALITY_JSON)

.PHONY: benchmark-plan
benchmark-plan: venv
	$(UV) run $(PYTHON) $(DEMO) benchmark-plan --model $(BENCHMARK_MODEL) --algorithms $(BENCHMARK_ALGORITHMS) --output-json $(BENCHMARK_JSON)

.PHONY: gpu-benchmark
gpu-benchmark: venv
	$(UV) run $(PYTHON) $(DEMO) gpu-benchmark --models $(GPU_BENCHMARK_MODELS) --variants $(GPU_BENCHMARK_VARIANTS) --kernels $(GPU_BENCHMARK_KERNELS) --output-json $(GPU_BENCHMARK_JSON) --report-html $(GPU_BENCHMARK_HTML)

.PHONY: pipeline_dev
pipeline_dev: format lint test smoke-html

.PHONY: pipeline_article
pipeline_article: plan quantize-dry-run quality-eval-dry-run benchmark-plan smoke-html

.PHONY: pipeline_quality
pipeline_quality: quality-eval-dry-run smoke-html

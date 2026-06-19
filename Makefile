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
		'  make pipeline_dev          Format, lint, test, and smoke-check the project.' \
		'  make pipeline_article      Run article-support dry-run pipeline commands.' \
		'  make install-compression   Install optional compression/eval packages.' \
		'  make install-alternatives  Install optional alternative packages.' \
		'  make install-serving       Install vLLM; use only on a supported CUDA/Linux stack.'

.PHONY: venv
venv:
	$(UV) sync --group dev

.PHONY: clean
clean:
	rm -rf .venv .ruff_cache .pytest_cache .mypy_cache .uv-cache compression_demo/__pycache__ tests/__pycache__

.PHONY: install-compression
install-compression: venv
	$(UV) pip install accelerate compressed-tensors datasets llmcompressor lm_eval torch 'transformers>=4.52.1'

.PHONY: install-alternatives
install-alternatives: venv
	$(UV) pip install bitsandbytes gptqmodel peft

.PHONY: install-serving
install-serving: venv
	$(UV) pip install vllm

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
	$(UV) run pytest

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

.PHONY: pipeline_dev
pipeline_dev: format lint test smoke-html

.PHONY: pipeline_article
pipeline_article: plan quantize-dry-run quality-eval-dry-run smoke-html

.PHONY: pipeline_quality
pipeline_quality: quality-eval-dry-run smoke-html

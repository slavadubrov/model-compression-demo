"""Command line interface for the model compression demo."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from html.parser import HTMLParser

from .catalog import ALGORITHMS, ARCH_TO_COMPUTE_CAPABILITY, GPU_INSTANCES, SCHEMES
from .evals import (
    DEFAULT_LM_EVAL_LIMIT,
    DEFAULT_LM_EVAL_TASK,
    DEFAULT_MAX_PERPLEXITY_DELTA_PCT,
    DEFAULT_MAX_TASK_REGRESSION,
    DEFAULT_PROMPTS,
    QualityGateError,
    build_quality_eval_plan,
    format_quality_eval_plan,
    run_quality_eval,
)
from .model_specs import MODEL_PRESETS, architecture_from_hf_config, generic_architecture
from .planner import build_plan, estimate_serving_memory, select_algorithm
from .recipes import (
    EXECUTABLE_QUANTIZATION_ALGORITHMS,
    build_vllm_serve_command,
    default_output_dir,
    dry_run_quantization_command,
    recipe_snippet,
    run_llmcompressor_quantization,
)


def _gib(value: float) -> str:
    return f"{value:.2f} GiB"


def _print_algorithm_table() -> None:
    print("Algorithm                           Scheme        Package")
    print("-" * 78)
    for algorithm in ALGORITHMS.values():
        print(f"{algorithm.name[:35]:35} {algorithm.scheme_key[:12]:12} {algorithm.package}")


def _print_scheme_table() -> None:
    print("Scheme             Eff bits  Min CC  Package")
    print("-" * 74)
    for scheme in SCHEMES.values():
        print(
            f"{scheme.key[:18]:18} "
            f"{scheme.effective_weight_bits:7.2f}  "
            f"{scheme.min_compute_capability:6.1f}  "
            f"{scheme.package}"
        )


def _folder_size(path: pathlib.Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _format_bytes(nbytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if nbytes < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes = nbytes / 1024
    return f"{nbytes:.1f} TiB"


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_architecture(args: argparse.Namespace):
    if getattr(args, "hf_config", None):
        return architecture_from_hf_config(args.hf_config)
    preset = getattr(args, "model_preset", None)
    if preset:
        return MODEL_PRESETS[preset]
    return generic_architecture(
        layers=args.layers,
        hidden_size=args.hidden_size,
        kv_head_ratio=args.kv_head_ratio,
    )


def _resolve_params_b(args: argparse.Namespace, architecture) -> float:
    if args.params_b is not None:
        return args.params_b
    if architecture.params_b is not None:
        return architecture.params_b
    raise ValueError("Pass --params-b or choose a --model-preset with a parameter count.")


def _print_architecture_notes(architecture) -> None:
    print(f"Architecture:     {architecture.name} ({architecture.source})")
    print(
        "Layers/hidden/KV: "
        f"{architecture.layers} / {architecture.hidden_size} / {architecture.kv_head_ratio:.3f}"
    )
    for note in architecture.notes:
        print(f"Note:             {note}")


class _HTMLSmokeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.h1 = 0
        self.tables = 0
        self.links = 0
        self.scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h1":
            self.h1 += 1
        elif tag == "table":
            self.tables += 1
        elif tag == "a":
            self.links += 1
        elif tag == "script":
            self.scripts += 1


def _smoke_html(path: pathlib.Path) -> None:
    parser = _HTMLSmokeParser()
    parser.feed(path.read_text(encoding="utf-8"))
    missing = []
    if parser.h1 != 1:
        missing.append("exactly one h1")
    if parser.tables < 3:
        missing.append("at least three tables")
    if parser.links < 8:
        missing.append("source links")
    if parser.scripts < 1:
        missing.append("calculator script")
    if missing:
        raise AssertionError(f"HTML smoke check failed: missing {', '.join(missing)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and demo LLM model compression choices.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "list-algorithms", help="List compression algorithms covered by the demo."
    )
    subparsers.add_parser(
        "list-schemes", help="List quantization schemes and hardware requirements."
    )

    estimate = subparsers.add_parser(
        "estimate", help="Estimate serving memory for a model and scheme."
    )
    estimate.add_argument("--params-b", type=float, help="Model parameter count in billions.")
    estimate.add_argument("--model-preset", choices=sorted(MODEL_PRESETS))
    estimate.add_argument("--hf-config", help="Path to a local Hugging Face config.json.")
    estimate.add_argument("--scheme", choices=sorted(SCHEMES), default="w4a16")
    estimate.add_argument("--layers", type=int, default=32)
    estimate.add_argument("--hidden-size", type=int, default=4096)
    estimate.add_argument("--context", type=int, default=4096)
    estimate.add_argument("--concurrency", type=int, default=1)
    estimate.add_argument("--kv-cache-bits", type=float, default=16)
    estimate.add_argument("--kv-head-ratio", type=float, default=1.0)
    estimate.add_argument("--json", action="store_true")

    plan = subparsers.add_parser("plan", help="Choose an algorithm and GPU memory target.")
    plan.add_argument("--params-b", type=float)
    plan.add_argument("--model-preset", choices=sorted(MODEL_PRESETS))
    plan.add_argument("--hf-config", help="Path to a local Hugging Face config.json.")
    plan.add_argument(
        "--goal", default="fit-memory", help="Examples: fit-memory, quality, throughput, qlora."
    )
    plan.add_argument("--hardware", choices=sorted(ARCH_TO_COMPUTE_CAPABILITY), default="ampere")
    plan.add_argument("--deployment", default="vllm")
    plan.add_argument("--algorithm", choices=sorted(ALGORITHMS))
    plan.add_argument("--layers", type=int, default=32)
    plan.add_argument("--hidden-size", type=int, default=4096)
    plan.add_argument("--context", type=int, default=4096)
    plan.add_argument("--concurrency", type=int, default=1)
    plan.add_argument("--kv-cache-bits", type=float, default=16)
    plan.add_argument("--kv-head-ratio", type=float, default=1.0)

    recipe = subparsers.add_parser("recipe", help="Print a reference recipe.")
    recipe.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="gptq-w4a16")

    quantize = subparsers.add_parser("quantize", help="Run or dry-run llm-compressor quantization.")
    quantize.add_argument(
        "--algorithm", choices=EXECUTABLE_QUANTIZATION_ALGORITHMS, default="gptq-w4a16"
    )
    quantize.add_argument("--model", default="Qwen/Qwen3-0.6B")
    quantize.add_argument("--output-dir")
    quantize.add_argument("--dataset", default="wikitext")
    quantize.add_argument("--dataset-config-name", default="wikitext-2-raw-v1")
    quantize.add_argument(
        "--calibration-file",
        help="Representative JSONL or text calibration file. Overrides --dataset.",
    )
    quantize.add_argument("--text-column", default="text")
    quantize.add_argument("--num-calibration-samples", type=int, default=256)
    quantize.add_argument("--max-seq-length", type=int, default=4096)
    quantize.add_argument("--dry-run", action="store_true")

    serve = subparsers.add_parser("serve-command", help="Print a vLLM serve command.")
    serve.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="gptq-w4a16")
    serve.add_argument("--model-path")
    serve.add_argument("--max-model-len", type=int)
    serve.add_argument("--tensor-parallel-size", type=int, default=1)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--fp8-kv-cache", action="store_true")
    serve.add_argument("--enable-prefix-caching", action="store_true")

    compare = subparsers.add_parser(
        "compare-size", help="Compare base and compressed model folder sizes."
    )
    compare.add_argument("--base-dir", required=True)
    compare.add_argument("--compressed-dir", required=True)

    quality = subparsers.add_parser(
        "quality-eval",
        help="Run or dry-run quality checks for a base and compressed model.",
    )
    quality.add_argument("--base-model", required=True)
    quality.add_argument("--compressed-model", required=True)
    quality.add_argument(
        "--mode",
        choices=["all", "generation", "perplexity", "long-context", "lm-eval"],
        default="all",
    )
    quality.add_argument("--prompt", action="append", default=[])
    quality.add_argument("--dataset", default="wikitext")
    quality.add_argument("--dataset-config-name", default="wikitext-2-raw-v1")
    quality.add_argument("--dataset-split", default="test")
    quality.add_argument(
        "--lm-eval-task",
        help=f"Task name for lm_eval; defaults to {DEFAULT_LM_EVAL_TASK!r} in all/lm-eval mode.",
    )
    quality.add_argument("--lm-eval-limit", type=int, default=DEFAULT_LM_EVAL_LIMIT)
    quality.add_argument("--long-context-tokens", type=int, default=4096)
    quality.add_argument(
        "--max-perplexity-delta-pct",
        type=float,
        default=DEFAULT_MAX_PERPLEXITY_DELTA_PCT,
    )
    quality.add_argument("--max-task-regression", type=float, default=DEFAULT_MAX_TASK_REGRESSION)
    quality.add_argument(
        "--allow-long-context-anchor-miss",
        dest="require_long_context_anchor",
        action="store_false",
    )
    quality.add_argument(
        "--eval-loading",
        choices=["sequential", "together"],
        default="sequential",
        help="Load base and compressed models sequentially or in the same process.",
    )
    quality.add_argument("--max-new-tokens", type=int, default=80)
    quality.add_argument("--max-tokens", type=int, default=5000)
    quality.add_argument("--stride", type=int, default=512)
    quality.add_argument("--output-json")
    quality.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("env", help="Show optional ML dependency availability.")

    smoke = subparsers.add_parser(
        "smoke-html", help="Validate that the HTML guide has the expected structure."
    )
    smoke.add_argument("--path", default=str(pathlib.Path(__file__).parents[1] / "index.html"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-algorithms":
        _print_algorithm_table()
        return 0

    if args.command == "list-schemes":
        _print_scheme_table()
        return 0

    if args.command == "estimate":
        architecture = _resolve_architecture(args)
        try:
            params_b = _resolve_params_b(args, architecture)
        except ValueError as exc:
            parser.error(str(exc))
        estimate = estimate_serving_memory(
            params_b=params_b,
            scheme_key=args.scheme,
            layers=architecture.layers,
            hidden_size=architecture.hidden_size,
            context_tokens=args.context,
            concurrency=args.concurrency,
            kv_cache_bits=args.kv_cache_bits,
            kv_head_ratio=architecture.kv_head_ratio,
        )
        if args.json:
            print(json.dumps(estimate.__dict__, indent=2, sort_keys=True))
        else:
            _print_architecture_notes(architecture)
            print(f"Scheme:           {args.scheme}")
            print(f"Weights:          {_gib(estimate.weight_gib)}")
            print(f"KV cache:         {_gib(estimate.kv_cache_gib)}")
            print(f"Runtime overhead: {_gib(estimate.runtime_overhead_gib)}")
            print(f"Safety buffer:    {_gib(estimate.safety_buffer_gib)}")
            print(f"Total target:     {_gib(estimate.total_gib)}")
        return 0

    if args.command == "plan":
        architecture = _resolve_architecture(args)
        try:
            params_b = _resolve_params_b(args, architecture)
        except ValueError as exc:
            parser.error(str(exc))
        algorithm_key = args.algorithm or select_algorithm(
            goal=args.goal,
            hardware=args.hardware,
            deployment=args.deployment,
        )
        plan = build_plan(
            params_b=params_b,
            algorithm_key=algorithm_key,
            layers=architecture.layers,
            hidden_size=architecture.hidden_size,
            context_tokens=args.context,
            concurrency=args.concurrency,
            kv_cache_bits=args.kv_cache_bits,
            kv_head_ratio=architecture.kv_head_ratio,
        )
        algorithm = ALGORITHMS[algorithm_key]
        scheme = SCHEMES[plan.scheme_key]
        _print_architecture_notes(architecture)
        print(f"Algorithm:         {algorithm.name}")
        print(f"Scheme:            {scheme.label}")
        print(f"Package:           {algorithm.package}")
        print(f"{plan.serving_target_label}: {_gib(plan.serving_memory.total_gib)}")
        print(f"Compression CPU:   {_gib(plan.compression_memory.cpu_gib)}")
        if plan.compression_memory.gpu_gib > 0:
            print(f"Compression GPU:   {_gib(plan.compression_memory.gpu_gib)}")
        else:
            print("Compression GPU:   not required for this local conversion path")
        if plan.local_recommendations:
            print("Recommended local runtimes:")
            for rec in plan.local_recommendations:
                print(f"  - {rec.name}: {_gib(rec.memory_target_gib)}; {rec.reason}")
        else:
            print("Recommended GPUs:")
            for rec in plan.recommendations:
                marker = "fits" if rec.fits else "needs sharding"
                print(f"  - {rec.instance.name}: {marker}; {rec.reason}")
        print("Notes:")
        for note in plan.notes:
            print(f"  - {note}")
        return 0

    if args.command == "recipe":
        print(recipe_snippet(args.algorithm), end="")
        return 0

    if args.command == "quantize":
        output_dir = args.output_dir or default_output_dir(
            model=args.model,
            algorithm_key=args.algorithm,
        )
        if args.dry_run:
            print(
                dry_run_quantization_command(
                    algorithm_key=args.algorithm,
                    model=args.model,
                    output_dir=output_dir,
                    dataset=args.dataset,
                    dataset_config_name=args.dataset_config_name,
                    calibration_file=args.calibration_file,
                    text_column=args.text_column,
                    num_calibration_samples=args.num_calibration_samples,
                    max_seq_length=args.max_seq_length,
                )
            )
            return 0
        run_llmcompressor_quantization(
            algorithm_key=args.algorithm,
            model=args.model,
            output_dir=output_dir,
            dataset=args.dataset,
            dataset_config_name=args.dataset_config_name,
            calibration_file=args.calibration_file,
            text_column=args.text_column,
            num_calibration_samples=args.num_calibration_samples,
            max_seq_length=args.max_seq_length,
        )
        return 0

    if args.command == "serve-command":
        model_path = args.model_path or default_output_dir(
            model="Qwen/Qwen3-0.6B",
            algorithm_key=args.algorithm,
        )
        max_model_len = args.max_model_len
        if max_model_len is None:
            max_model_len = 32768 if args.algorithm in {"fp8-dynamic", "kv-cache-fp8"} else 4096
        print(
            build_vllm_serve_command(
                algorithm_key=args.algorithm,
                model_path=model_path,
                max_model_len=max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                port=args.port,
                fp8_kv_cache=args.fp8_kv_cache,
                enable_prefix_caching=args.enable_prefix_caching,
            )
        )
        print("# Check your installed vLLM version because FP8 flag names can vary.")
        return 0

    if args.command == "compare-size":
        base = _folder_size(pathlib.Path(args.base_dir))
        compressed = _folder_size(pathlib.Path(args.compressed_dir))
        reduction = 0.0 if base == 0 else (1 - compressed / base) * 100
        print(f"Base:       {_format_bytes(base)}")
        print(f"Compressed: {_format_bytes(compressed)}")
        print(f"Reduction:  {reduction:.1f}%")
        return 0

    if args.command == "quality-eval":
        prompts = tuple(args.prompt) if args.prompt else DEFAULT_PROMPTS
        plan = build_quality_eval_plan(
            base_model=args.base_model,
            compressed_model=args.compressed_model,
            mode=args.mode,
            prompts=prompts,
            dataset=args.dataset,
            dataset_config_name=args.dataset_config_name,
            dataset_split=args.dataset_split,
            lm_eval_task=args.lm_eval_task,
            lm_eval_limit=args.lm_eval_limit,
            long_context_tokens=args.long_context_tokens,
            max_perplexity_delta_pct=args.max_perplexity_delta_pct,
            max_task_regression=args.max_task_regression,
            require_long_context_anchor=args.require_long_context_anchor,
            output_json=args.output_json,
        )
        if args.dry_run:
            print(format_quality_eval_plan(plan))
            return 0
        try:
            results = run_quality_eval(
                plan=plan,
                max_new_tokens=args.max_new_tokens,
                max_tokens=args.max_tokens,
                stride=args.stride,
                lm_eval_limit=args.lm_eval_limit,
                sequential_models=args.eval_loading == "sequential",
            )
        except QualityGateError as exc:
            print(json.dumps(exc.results, indent=2, sort_keys=True))
            return 2
        else:
            print(json.dumps(results, indent=2, sort_keys=True))
            return 0

    if args.command == "env":
        modules = [
            "torch",
            "transformers",
            "datasets",
            "llmcompressor",
            "compressed_tensors",
            "vllm",
            "bitsandbytes",
            "gptqmodel",
        ]
        for module in modules:
            print(f"{module:20} {'installed' if _installed(module) else 'missing'}")
        print("\nKnown GPU classes:")
        for gpu in GPU_INSTANCES:
            print(f"{gpu.name:22} {gpu.memory_gib:6.1f} GiB  cc {gpu.compute_capability:>4}")
        return 0

    if args.command == "smoke-html":
        _smoke_html(pathlib.Path(args.path))
        print(f"HTML guide OK: {args.path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

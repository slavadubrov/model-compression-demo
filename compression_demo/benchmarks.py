"""Benchmark command planning for quantized serving workflows."""

from __future__ import annotations

import json
import pathlib
import shlex
from dataclasses import asdict, dataclass

from .catalog import ALGORITHMS
from .evals import DEFAULT_LM_EVAL_LIMIT, DEFAULT_LM_EVAL_TASK
from .recipes import build_vllm_serve_command, default_output_dir

BENCHMARK_WARNING = (
    "This command plan generates reproducible benchmark commands only. GPU benchmark "
    "numbers are environment-specific and must be measured on the target hardware, "
    "driver, CUDA, vLLM, model, prompt mix, and concurrency settings."
)

_VLLM_QUANTIZATION_FLAGS = {
    "gptq-w4a16": ("--quantization", "gptq"),
}


@dataclass(frozen=True)
class BenchmarkPlanRow:
    algorithm_key: str
    algorithm_name: str
    model_path: str
    serve_command: str
    bench_command: str
    quality_eval_command: str
    hardware_notes: str
    runtime_notes: str


@dataclass(frozen=True)
class BenchmarkPlan:
    warning: str
    model: str
    dataset_name: str
    num_prompts: int
    input_len: int
    output_len: int
    rows: tuple[BenchmarkPlanRow, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rows"] = [asdict(row) for row in self.rows]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def parse_algorithm_list(value: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated algorithm list."""

    algorithms = tuple(item.strip() for item in value.split(",") if item.strip())
    if not algorithms:
        raise ValueError("Pass at least one algorithm key.")
    unknown = [algorithm for algorithm in algorithms if algorithm not in ALGORITHMS]
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown benchmark algorithm(s): {joined}")
    return algorithms


def benchmark_model_path(*, model: str, algorithm_key: str) -> str:
    """Return the conventional benchmark model path for an algorithm."""

    return default_output_dir(model=model, algorithm_key=algorithm_key)


def _quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _serve_command(
    *,
    model: str,
    algorithm_key: str,
    model_path: str,
    max_model_len: int,
    port: int,
) -> str:
    if algorithm_key == "fp8-dynamic":
        return build_vllm_serve_command(
            algorithm_key=algorithm_key,
            model_path=model_path,
            max_model_len=max_model_len,
            port=port,
        )

    parts = [
        "vllm",
        "serve",
        model_path,
        "--max-model-len",
        str(max_model_len),
    ]
    parts.extend(_VLLM_QUANTIZATION_FLAGS.get(algorithm_key, ()))
    if port != 8000:
        parts.extend(["--port", str(port)])
    return _quote_command(parts)


def _bench_command(
    *,
    model_path: str,
    dataset_name: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    port: int,
) -> str:
    parts = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model_path,
        "--dataset-name",
        dataset_name,
        "--num-prompts",
        str(num_prompts),
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
    ]
    if port != 8000:
        parts.extend(["--port", str(port)])
    return _quote_command(parts)


def _quality_eval_command(*, model: str, model_path: str) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "demo.py",
        "quality-eval",
        "--base-model",
        model,
        "--compressed-model",
        model_path,
        "--mode",
        "all",
        "--lm-eval-task",
        DEFAULT_LM_EVAL_TASK,
        "--lm-eval-limit",
        str(DEFAULT_LM_EVAL_LIMIT),
        "--dry-run",
    ]
    return _quote_command(parts)


def _hardware_notes(algorithm_key: str) -> str:
    if algorithm_key == "fp8-dynamic":
        return "Use Ada, Hopper, Blackwell, or another stack with verified FP8 kernels."
    return "Benchmark on the exact GPU family and vLLM version planned for deployment."


def _runtime_notes(algorithm_key: str) -> str:
    if algorithm_key == "gptq-w4a16":
        return "GPTQ is a checkpoint method; compare the active vLLM kernel path before promotion."
    if algorithm_key == "fp8-dynamic":
        return "Confirm FP8 flag names against the installed vLLM version."
    return "Validate runtime support before treating this as a production path."


def build_benchmark_plan(
    *,
    model: str,
    algorithm_keys: tuple[str, ...],
    dataset_name: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    max_model_len: int,
    port: int = 8000,
) -> BenchmarkPlan:
    """Build a dependency-light command plan for vLLM serving benchmarks."""

    rows = []
    for algorithm_key in algorithm_keys:
        algorithm = ALGORITHMS[algorithm_key]
        model_path = benchmark_model_path(model=model, algorithm_key=algorithm_key)
        rows.append(
            BenchmarkPlanRow(
                algorithm_key=algorithm_key,
                algorithm_name=algorithm.name,
                model_path=model_path,
                serve_command=_serve_command(
                    model=model,
                    algorithm_key=algorithm_key,
                    model_path=model_path,
                    max_model_len=max_model_len,
                    port=port,
                ),
                bench_command=_bench_command(
                    model_path=model_path,
                    dataset_name=dataset_name,
                    num_prompts=num_prompts,
                    input_len=input_len,
                    output_len=output_len,
                    port=port,
                ),
                quality_eval_command=_quality_eval_command(model=model, model_path=model_path),
                hardware_notes=_hardware_notes(algorithm_key),
                runtime_notes=_runtime_notes(algorithm_key),
            )
        )

    return BenchmarkPlan(
        warning=BENCHMARK_WARNING,
        model=model,
        dataset_name=dataset_name,
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
        rows=tuple(rows),
    )


def write_benchmark_plan_json(plan: BenchmarkPlan, output_json: str) -> None:
    """Write a benchmark plan JSON file."""

    path = pathlib.Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.to_json() + "\n", encoding="utf-8")


def format_benchmark_plan(plan: BenchmarkPlan) -> str:
    """Format a benchmark plan for terminal output."""

    lines = [
        "Quantization benchmark command plan",
        "-----------------------------------",
        f"Model:        {plan.model}",
        f"Dataset:      {plan.dataset_name}",
        f"Prompts:      {plan.num_prompts}",
        f"Input/output: {plan.input_len} / {plan.output_len}",
        f"Warning:      {plan.warning}",
        "",
    ]
    for row in plan.rows:
        lines.extend(
            [
                f"[{row.algorithm_key}] {row.algorithm_name}",
                f"Model path: {row.model_path}",
                f"Serve:     {row.serve_command}",
                f"Benchmark: {row.bench_command}",
                f"Quality:   {row.quality_eval_command}",
                f"Hardware:  {row.hardware_notes}",
                f"Runtime:   {row.runtime_notes}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"

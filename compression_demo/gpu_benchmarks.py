"""GPU benchmark runner and HTML report generation.

The public helpers in this module avoid heavyweight imports. The actual runner
imports torch/transformers/bitsandbytes lazily so the default CLI and tests stay
usable on machines without a CUDA stack.
"""

from __future__ import annotations

import gc
import html
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import time
import traceback
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BYTES_PER_GIB = 1024**3

DEFAULT_GPU_BENCHMARK_MODELS = ("Qwen/Qwen3-0.6B",)
DEFAULT_GPU_BENCHMARK_VARIANTS = ("bf16", "bnb-int8", "bnb-nf4")
DEFAULT_GPU_BENCHMARK_KERNELS = ("sdpa", "eager")
DEFAULT_GPU_BENCHMARK_PROMPTS = (
    "Explain why model quantization changes memory bandwidth pressure.",
    "Give two practical checks before promoting a compressed LLM checkpoint.",
)

VARIANT_LABELS = {
    "bf16": "BF16 baseline",
    "fp16": "FP16 baseline",
    "bnb-int8": "bitsandbytes LLM.int8",
    "bnb-nf4": "bitsandbytes NF4 4-bit",
}

KERNEL_LABELS = {
    "eager": "Transformers eager attention",
    "sdpa": "PyTorch SDPA default",
    "sdpa-flash": "PyTorch SDPA flash forced",
    "sdpa-math": "PyTorch SDPA math forced",
    "flash-attn-2": "FlashAttention 2",
}


@dataclass(frozen=True)
class GPUBenchmarkRun:
    model: str
    variant: str
    variant_label: str
    kernel: str
    kernel_label: str
    status: str
    error: str | None = None
    load_seconds: float | None = None
    generation_seconds_mean: float | None = None
    generation_seconds_min: float | None = None
    generated_tokens_per_second: float | None = None
    input_tokens: int | None = None
    generated_tokens: int | None = None
    model_memory_gib: float | None = None
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None
    compression_ratio_vs_bf16: float | None = None
    response_preview: str | None = None


def parse_csv(value: str | Iterable[str]) -> tuple[str, ...]:
    """Parse a comma-separated CLI value into a normalized tuple."""

    if isinstance(value, str):
        items = value.split(",")
    else:
        items = value
    parsed = tuple(item.strip() for item in items if item and item.strip())
    if not parsed:
        raise ValueError("Pass at least one value.")
    return parsed


def validate_gpu_benchmark_axes(
    *,
    variants: tuple[str, ...],
    kernels: tuple[str, ...],
) -> None:
    unknown_variants = [variant for variant in variants if variant not in VARIANT_LABELS]
    unknown_kernels = [kernel for kernel in kernels if kernel not in KERNEL_LABELS]
    if unknown_variants:
        raise ValueError(f"Unknown GPU benchmark variant(s): {', '.join(unknown_variants)}")
    if unknown_kernels:
        raise ValueError(f"Unknown GPU benchmark kernel(s): {', '.join(unknown_kernels)}")


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_gpu_benchmark_numbers(
    *,
    max_new_tokens: int,
    warmup_runs: int,
    repeat_runs: int,
) -> None:
    """Validate benchmark loop counts before a GPU run starts."""

    _validate_positive_int("max_new_tokens", max_new_tokens)
    _validate_positive_int("warmup_runs", warmup_runs)
    _validate_positive_int("repeat_runs", repeat_runs)


def build_gpu_benchmark_plan(
    *,
    models: tuple[str, ...],
    variants: tuple[str, ...],
    kernels: tuple[str, ...],
    prompts: tuple[str, ...],
    max_new_tokens: int,
    warmup_runs: int,
    repeat_runs: int,
    output_json: str,
    report_html: str,
) -> dict[str, Any]:
    """Return a dependency-light description of the intended GPU benchmark."""

    validate_gpu_benchmark_axes(variants=variants, kernels=kernels)
    validate_gpu_benchmark_numbers(
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
    )
    return {
        "models": list(models),
        "variants": list(variants),
        "kernels": list(kernels),
        "prompts": list(prompts),
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "output_json": output_json,
        "report_html": report_html,
        "total_runs": len(models) * len(variants) * len(kernels),
    }


def format_gpu_benchmark_plan(plan: dict[str, Any]) -> str:
    """Format a dry-run plan for terminal output."""

    lines = [
        "GPU benchmark plan",
        "------------------",
        f"Models:       {', '.join(plan['models'])}",
        f"Variants:     {', '.join(plan['variants'])}",
        f"Kernels:      {', '.join(plan['kernels'])}",
        f"Prompts:      {len(plan['prompts'])}",
        f"New tokens:   {plan['max_new_tokens']}",
        f"Warmup/repeat:{plan['warmup_runs']} / {plan['repeat_runs']}",
        f"Total runs:   {plan['total_runs']}",
        f"JSON:         {plan['output_json']}",
        f"HTML report:  {plan['report_html']}",
        "",
        "This command downloads models if they are not already cached.",
    ]
    return "\n".join(lines)


def _module_version(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return None
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "installed"


def collect_gpu_environment() -> dict[str, Any]:
    """Collect runtime details for a benchmark result file."""

    env: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            module: _module_version(module)
            for module in (
                "torch",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "llmcompressor",
                "vllm",
                "gptqmodel",
                "lm_eval",
            )
        },
    }

    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        env["nvidia_smi"] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        env["nvidia_smi"] = {"error": str(exc)}

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            env["torch"] = {
                "version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                env["torch"]["device_name"] = torch.cuda.get_device_name(device)
                env["torch"]["compute_capability"] = f"{props.major}.{props.minor}"
                env["torch"]["total_memory_gib"] = props.total_memory / BYTES_PER_GIB
        except Exception as exc:  # pragma: no cover - defensive environment capture
            env["torch"] = {"error": str(exc)}

    return env


def _attention_implementation(kernel: str) -> str:
    if kernel == "eager":
        return "eager"
    if kernel == "flash-attn-2":
        return "flash_attention_2"
    return "sdpa"


@contextmanager
def _sdpa_kernel_context(kernel: str):
    if kernel not in {"sdpa", "sdpa-flash", "sdpa-math"}:
        yield
        return

    import torch

    backends = torch.backends.cuda
    getters = {
        "flash": getattr(backends, "flash_sdp_enabled", None),
        "mem_efficient": getattr(backends, "mem_efficient_sdp_enabled", None),
        "math": getattr(backends, "math_sdp_enabled", None),
    }
    setters = {
        "flash": getattr(backends, "enable_flash_sdp", None),
        "mem_efficient": getattr(backends, "enable_mem_efficient_sdp", None),
        "math": getattr(backends, "enable_math_sdp", None),
    }
    previous = {
        name: getter()
        for name, getter in getters.items()
        if callable(getter) and callable(setters.get(name))
    }

    if kernel == "sdpa-flash":
        desired = {"flash": True, "mem_efficient": False, "math": False}
    elif kernel == "sdpa-math":
        desired = {"flash": False, "mem_efficient": False, "math": True}
    else:
        desired = {"flash": True, "mem_efficient": True, "math": True}

    try:
        for name, value in desired.items():
            setter = setters.get(name)
            if callable(setter):
                setter(value)
        yield
    finally:
        for name, value in previous.items():
            setter = setters.get(name)
            if callable(setter):
                setter(value)


def _require_cuda_stack() -> None:
    missing = [
        module for module in ("torch", "transformers") if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing GPU benchmark dependencies: "
            f"{', '.join(missing)}. Install torch and transformers first."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; CUDA is not visible.")


def _clear_cuda() -> None:
    gc.collect()
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _model_input_device(model: Any):
    for parameter in model.parameters():
        return parameter.device
    raise RuntimeError("Model does not expose any parameters.")


def _load_model_and_tokenizer(
    *,
    model_name: str,
    variant: str,
    kernel: str,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "attn_implementation": _attention_implementation(kernel),
        "trust_remote_code": trust_remote_code,
    }

    if variant in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if variant == "bf16" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
        model.to("cuda")
        model.eval()
        return model, tokenizer

    if variant.startswith("bnb-"):
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("bitsandbytes is not installed.")
        from transformers import BitsAndBytesConfig

        if variant == "bnb-int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif variant == "bnb-nf4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:  # pragma: no cover - protected by validation
            raise ValueError(f"Unsupported bitsandbytes variant: {variant}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            quantization_config=quantization_config,
            **kwargs,
        )
        model.eval()
        return model, tokenizer

    raise ValueError(f"Unsupported benchmark variant: {variant}")


def _generate_once(
    *,
    model: Any,
    tokenizer: Any,
    prompts: tuple[str, ...],
    max_new_tokens: int,
) -> tuple[int, int, str]:
    import torch

    device = _model_input_device(model)
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True).to(device)
    input_tokens = int(encoded["attention_mask"].sum().item())
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_tokens = int((output.shape[-1] - encoded["input_ids"].shape[-1]) * len(prompts))
    preview_tokens = output[0][encoded["input_ids"].shape[-1] :]
    response_preview = tokenizer.decode(preview_tokens, skip_special_tokens=True)
    return input_tokens, generated_tokens, response_preview[:400]


def run_single_gpu_benchmark(
    *,
    model_name: str,
    variant: str,
    kernel: str,
    prompts: tuple[str, ...],
    max_new_tokens: int,
    warmup_runs: int,
    repeat_runs: int,
    trust_remote_code: bool = False,
) -> GPUBenchmarkRun:
    """Run one model/variant/kernel benchmark and return a structured result."""

    validate_gpu_benchmark_axes(variants=(variant,), kernels=(kernel,))
    validate_gpu_benchmark_numbers(
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
    )

    label = GPUBenchmarkRun(
        model=model_name,
        variant=variant,
        variant_label=VARIANT_LABELS[variant],
        kernel=kernel,
        kernel_label=KERNEL_LABELS[kernel],
        status="failed",
    )
    try:
        _require_cuda_stack()
        import torch

        _clear_cuda()
        with _sdpa_kernel_context(kernel):
            load_start = time.perf_counter()
            model, tokenizer = _load_model_and_tokenizer(
                model_name=model_name,
                variant=variant,
                kernel=kernel,
                trust_remote_code=trust_remote_code,
            )
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - load_start

            model_memory_gib = None
            if hasattr(model, "get_memory_footprint"):
                model_memory_gib = float(model.get_memory_footprint()) / BYTES_PER_GIB

            for _ in range(warmup_runs):
                _generate_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                )
                torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            durations: list[float] = []
            input_tokens = 0
            generated_tokens = 0
            response_preview = ""
            for _ in range(repeat_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                input_tokens, generated_tokens, response_preview = _generate_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                )
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)

            peak_allocated_gib = torch.cuda.max_memory_allocated() / BYTES_PER_GIB
            peak_reserved_gib = torch.cuda.max_memory_reserved() / BYTES_PER_GIB

        del model, tokenizer
        _clear_cuda()

        mean_seconds = sum(durations) / len(durations)
        return GPUBenchmarkRun(
            model=model_name,
            variant=variant,
            variant_label=VARIANT_LABELS[variant],
            kernel=kernel,
            kernel_label=KERNEL_LABELS[kernel],
            status="ok",
            load_seconds=load_seconds,
            generation_seconds_mean=mean_seconds,
            generation_seconds_min=min(durations),
            generated_tokens_per_second=generated_tokens / mean_seconds,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            model_memory_gib=model_memory_gib,
            peak_allocated_gib=peak_allocated_gib,
            peak_reserved_gib=peak_reserved_gib,
            response_preview=response_preview,
        )
    except Exception as exc:  # pragma: no cover - exercised by real GPU runs
        _clear_cuda()
        return GPUBenchmarkRun(
            **{
                **asdict(label),
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}",
            }
        )


def _apply_compression_ratios(runs: list[GPUBenchmarkRun]) -> list[GPUBenchmarkRun]:
    baselines: dict[tuple[str, str], float] = {}
    for run in runs:
        if (
            run.status == "ok"
            and run.variant == "bf16"
            and run.model_memory_gib
            and run.model_memory_gib > 0
        ):
            baselines[(run.model, run.kernel)] = run.model_memory_gib

    updated = []
    for run in runs:
        ratio = None
        baseline = baselines.get((run.model, run.kernel))
        if baseline and run.model_memory_gib and run.model_memory_gib > 0:
            ratio = baseline / run.model_memory_gib
        updated.append(
            GPUBenchmarkRun(
                **{
                    **asdict(run),
                    "compression_ratio_vs_bf16": ratio,
                }
            )
        )
    return updated


def _result_payload(
    *,
    environment: dict[str, Any],
    config: dict[str, Any],
    runs: list[GPUBenchmarkRun],
) -> dict[str, Any]:
    return {
        "environment": environment,
        "config": config,
        "runs": [asdict(run) for run in runs],
        "summary": summarize_gpu_benchmark_results(runs),
    }


def write_gpu_benchmark_json(payload: dict[str, Any], output_json: str) -> None:
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_gpu_benchmarks(
    *,
    models: tuple[str, ...],
    variants: tuple[str, ...],
    kernels: tuple[str, ...],
    prompts: tuple[str, ...],
    max_new_tokens: int,
    warmup_runs: int,
    repeat_runs: int,
    output_json: str,
    report_html: str,
    trust_remote_code: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run all GPU benchmarks, writing JSON and HTML progress as each run finishes."""

    config = build_gpu_benchmark_plan(
        models=models,
        variants=variants,
        kernels=kernels,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
        output_json=output_json,
        report_html=report_html,
    )
    environment = collect_gpu_environment()
    runs: list[GPUBenchmarkRun] = []

    for model in models:
        for kernel in kernels:
            for variant in variants:
                run = run_single_gpu_benchmark(
                    model_name=model,
                    variant=variant,
                    kernel=kernel,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    warmup_runs=warmup_runs,
                    repeat_runs=repeat_runs,
                    trust_remote_code=trust_remote_code,
                )
                runs.append(run)
                runs = _apply_compression_ratios(runs)
                payload = _result_payload(environment=environment, config=config, runs=runs)
                write_gpu_benchmark_json(payload, output_json)
                write_gpu_benchmark_report(payload, report_html)
                if fail_fast and run.status != "ok":
                    return payload

    payload = _result_payload(environment=environment, config=config, runs=runs)
    write_gpu_benchmark_json(payload, output_json)
    write_gpu_benchmark_report(payload, report_html)
    return payload


def summarize_gpu_benchmark_results(runs: list[GPUBenchmarkRun]) -> dict[str, Any]:
    ok_runs = [run for run in runs if run.status == "ok"]
    failed_runs = [run for run in runs if run.status != "ok"]
    fastest = max(
        ok_runs,
        key=lambda run: run.generated_tokens_per_second or 0,
        default=None,
    )
    lowest_memory = min(
        (run for run in ok_runs if run.peak_allocated_gib is not None),
        key=lambda run: run.peak_allocated_gib or float("inf"),
        default=None,
    )
    best_compression = max(
        (run for run in ok_runs if run.compression_ratio_vs_bf16 is not None),
        key=lambda run: run.compression_ratio_vs_bf16 or 0,
        default=None,
    )
    return {
        "total_runs": len(runs),
        "ok_runs": len(ok_runs),
        "failed_runs": len(failed_runs),
        "fastest": asdict(fastest) if fastest else None,
        "lowest_memory": asdict(lowest_memory) if lowest_memory else None,
        "best_compression": asdict(best_compression) if best_compression else None,
    }


def format_gpu_benchmark_summary(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "GPU benchmark summary",
        "---------------------",
        f"Runs:       {summary['ok_runs']} ok / {summary['failed_runs']} failed",
    ]
    fastest = summary.get("fastest")
    if fastest:
        lines.append(
            "Fastest:    "
            f"{fastest['model']} {fastest['variant']} {fastest['kernel']} "
            f"at {fastest['generated_tokens_per_second']:.2f} tok/s"
        )
    lowest = summary.get("lowest_memory")
    if lowest:
        lines.append(
            "Lowest mem: "
            f"{lowest['model']} {lowest['variant']} {lowest['kernel']} "
            f"at {lowest['peak_allocated_gib']:.2f} GiB peak allocated"
        )
    best = summary.get("best_compression")
    if best and best.get("compression_ratio_vs_bf16"):
        lines.append(
            "Best comp:  "
            f"{best['model']} {best['variant']} {best['kernel']} "
            f"at {best['compression_ratio_vs_bf16']:.2f}x vs BF16"
        )
    lines.append(f"JSON:       {payload['config']['output_json']}")
    lines.append(f"Report:     {payload['config']['report_html']}")
    return "\n".join(lines)


def _short_model(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def _run_label(run: dict[str, Any]) -> str:
    return f"{_short_model(run['model'])} / {run['variant']} / {run['kernel']}"


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if isinstance(value, int | float):
        return f"{value:.{digits}f}{suffix}"
    return "-"


def _bar_chart(
    *,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    unit: str,
    lower_is_better: bool = False,
) -> str:
    valid = [row for row in rows if isinstance(row.get(metric), int | float)]
    if not valid:
        return f"<section><h2>{html.escape(title)}</h2><p>No successful measurements.</p></section>"

    width = 980
    row_h = 32
    label_w = 330
    chart_w = width - label_w - 110
    height = 58 + row_h * len(valid)
    max_value = max(float(row[metric]) for row in valid) or 1.0
    rank = sorted(
        valid,
        key=lambda row: float(row[metric]),
        reverse=not lower_is_better,
    )
    best_id = id(rank[0])

    bars = []
    for index, row in enumerate(valid):
        value = float(row[metric])
        y = 42 + index * row_h
        bar_w = max(2, int(chart_w * value / max_value))
        color = "#2f7d6d" if id(row) == best_id else "#5b78d6"
        label = html.escape(_run_label(row))
        value_label = html.escape(_fmt(value, 2, unit))
        bars.append(
            f'<text x="0" y="{y + 17}" class="label">{label}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bar_w}" height="20" rx="3" fill="{color}" />'
            f'<text x="{label_w + bar_w + 8}" y="{y + 16}" class="value">{value_label}</text>'
        )

    axis_note = "Lower is better" if lower_is_better else "Higher is better"
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">'
        f'<text x="0" y="20" class="axis-note">{axis_note}</text>'
        f"{''.join(bars)}</svg></section>"
    )


def write_gpu_benchmark_report(payload: dict[str, Any], report_html: str) -> None:
    """Write a self-contained HTML report with tables and SVG plots."""

    path = Path(report_html)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("runs", [])
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    env = payload.get("environment", {})
    torch_env = env.get("torch", {}) if isinstance(env.get("torch"), dict) else {}
    summary = payload.get("summary", {})

    table_rows = []
    for row in rows:
        error = row.get("error") or ""
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(_short_model(str(row.get('model', ''))))}</td>"
            f"<td>{html.escape(str(row.get('variant')))}</td>"
            f"<td>{html.escape(str(row.get('kernel')))}</td>"
            f"<td>{html.escape(str(row.get('status')))}</td>"
            f"<td>{_fmt(row.get('generated_tokens_per_second'))}</td>"
            f"<td>{_fmt(row.get('peak_allocated_gib'))}</td>"
            f"<td>{_fmt(row.get('model_memory_gib'))}</td>"
            f"<td>{_fmt(row.get('compression_ratio_vs_bf16'), suffix='x')}</td>"
            f"<td><code>{html.escape(error.splitlines()[0] if error else '')}</code></td>"
            "</tr>"
        )

    fastest = summary.get("fastest") or {}
    lowest = summary.get("lowest_memory") or {}
    best = summary.get("best_compression") or {}
    highlights = [
        ("Fastest", fastest, "generated_tokens_per_second", " tok/s"),
        ("Lowest peak memory", lowest, "peak_allocated_gib", " GiB"),
        ("Best compression", best, "compression_ratio_vs_bf16", "x"),
    ]
    highlight_cards = []
    for title, row, metric, unit in highlights:
        if not row:
            continue
        highlight_cards.append(
            "<article>"
            f"<h3>{html.escape(title)}</h3>"
            f"<p>{html.escape(_run_label(row))}</p>"
            f"<strong>{_fmt(row.get(metric), suffix=unit)}</strong>"
            "</article>"
        )

    css = """
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }
    body { margin: 0; background: #f7f8fb; color: #17202a; }
    header { background: #102033; color: white; padding: 36px 48px; }
    main { max-width: 1180px; margin: 0 auto; padding: 28px 28px 48px; }
    h1, h2, h3 { margin: 0; line-height: 1.15; }
    h1 { font-size: 34px; }
    h2 { font-size: 22px; margin-bottom: 16px; }
    h3 { font-size: 15px; color: #506070; }
    p { margin: 8px 0 0; }
    section {
      margin-top: 26px;
      background: white;
      border: 1px solid #dfe5ee;
      border-radius: 8px;
      padding: 22px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
    }
    article {
      background: #eef4f2;
      border: 1px solid #d6e5df;
      border-radius: 8px;
      padding: 16px;
    }
    article strong { display: block; margin-top: 12px; font-size: 24px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td {
      padding: 10px 9px;
      border-bottom: 1px solid #e6ebf2;
      text-align: left;
      vertical-align: top;
    }
    th { color: #4c5c6d; font-weight: 700; background: #f2f5f8; }
    code { white-space: pre-wrap; font-size: 12px; color: #6b1f35; }
    svg { width: 100%; height: auto; display: block; }
    .label { font-size: 13px; fill: #2c3745; }
    .value, .axis-note { font-size: 12px; fill: #5d6978; }
    .meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
    }
    .meta div { background: #f4f6f9; padding: 12px; border-radius: 6px; }
    """

    created = html.escape(str(env.get("created_at", "unknown")))
    device = html.escape(str(torch_env.get("device_name", "unknown")))
    cuda = html.escape(str(torch_env.get("cuda_version", "unknown")))
    torch_version = html.escape(str(torch_env.get("version", "unknown")))
    compute = html.escape(str(torch_env.get("compute_capability", "unknown")))
    highlight_html = "".join(highlight_cards) or "<p>No completed runs yet.</p>"
    throughput_chart = _bar_chart(
        rows=ok_rows,
        metric="generated_tokens_per_second",
        title="Generation Throughput",
        unit=" tok/s",
    )
    memory_chart = _bar_chart(
        rows=ok_rows,
        metric="peak_allocated_gib",
        title="Peak Allocated GPU Memory",
        unit=" GiB",
        lower_is_better=True,
    )
    compression_chart = _bar_chart(
        rows=ok_rows,
        metric="compression_ratio_vs_bf16",
        title="Compression Ratio vs BF16 Model Footprint",
        unit="x",
    )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GPU Compression Benchmark Report</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>GPU Compression Benchmark Report</h1>
    <p>Generated {created}</p>
  </header>
  <main>
    <section>
      <h2>Environment</h2>
      <div class="meta">
        <div><strong>GPU</strong><p>{device}</p></div>
        <div><strong>Compute capability</strong><p>{compute}</p></div>
        <div><strong>PyTorch</strong><p>{torch_version}</p></div>
        <div><strong>CUDA runtime</strong><p>{cuda}</p></div>
      </div>
    </section>
    <section>
      <h2>Highlights</h2>
      <div class="grid">{highlight_html}</div>
    </section>
    {throughput_chart}
    {memory_chart}
    {compression_chart}
    <section>
      <h2>Measurements</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th><th>Variant</th><th>Kernel</th><th>Status</th>
            <th>Tok/s</th><th>Peak GiB</th><th>Model GiB</th><th>Compression</th><th>Error</th>
          </tr>
        </thead>
        <tbody>{"".join(table_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")

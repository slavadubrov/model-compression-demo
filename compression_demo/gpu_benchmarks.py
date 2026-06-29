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
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BYTES_PER_GIB = 1024**3

DEFAULT_GPU_BENCHMARK_MODELS = ("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B")
DEFAULT_GPU_BENCHMARK_VARIANTS = ("bf16", "fp8-dynamic", "fp8-dynamic-kv")
DEFAULT_GPU_BENCHMARK_KERNELS = ("sdpa", "eager")
DEFAULT_VLLM_MAX_MODEL_LEN = 2048
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_GPU_BENCHMARK_PROMPTS = (
    "Explain why model quantization changes memory bandwidth pressure.",
    "Give two practical checks before promoting a compressed LLM checkpoint.",
)

VARIANT_LABELS = {
    "bf16": "BF16 baseline",
    "fp16": "FP16 baseline",
    "bnb-int8": "bitsandbytes LLM.int8",
    "bnb-nf4": "bitsandbytes NF4 4-bit",
    "fp8-dynamic": "vLLM dynamic FP8 W8A8",
    "fp8-dynamic-kv": "vLLM dynamic FP8 W8A8 + FP8 KV cache",
}

KERNEL_LABELS = {
    "eager": "Transformers eager attention",
    "sdpa": "PyTorch SDPA default",
    "sdpa-flash": "PyTorch SDPA flash forced",
    "sdpa-math": "PyTorch SDPA math forced",
    "flash-attn-2": "FlashAttention 2",
    "vllm": "vLLM engine",
}

VLLM_VARIANTS = {"bf16", "fp16", "fp8-dynamic", "fp8-dynamic-kv"}
TRANSFORMERS_ONLY_VARIANTS = {"bnb-int8", "bnb-nf4"}


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


def _compatible_variant_kernel(*, variant: str, kernel: str) -> bool:
    if kernel == "vllm":
        return variant in VLLM_VARIANTS
    return variant not in {"fp8-dynamic", "fp8-dynamic-kv"}


def _skipped_variant_kernel_reason(*, variant: str, kernel: str) -> str:
    if kernel == "vllm" and variant in TRANSFORMERS_ONLY_VARIANTS:
        return f"{variant} is a Transformers/bitsandbytes benchmark variant, not a vLLM path."
    if variant in {"fp8-dynamic", "fp8-dynamic-kv"} and kernel != "vllm":
        return "FP8 benchmark variants require the vLLM kernel/runtime path."
    return "Unsupported variant/kernel combination."


def _benchmark_combinations(
    *,
    models: tuple[str, ...],
    variants: tuple[str, ...],
    kernels: tuple[str, ...],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    runs: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for model in models:
        for kernel in kernels:
            for variant in variants:
                payload = {"model": model, "variant": variant, "kernel": kernel}
                if _compatible_variant_kernel(variant=variant, kernel=kernel):
                    runs.append(payload)
                else:
                    skipped.append(
                        {
                            **payload,
                            "reason": _skipped_variant_kernel_reason(
                                variant=variant,
                                kernel=kernel,
                            ),
                        }
                    )
    return runs, skipped


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
    vllm_max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN,
    vllm_gpu_memory_utilization: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
) -> dict[str, Any]:
    """Return a dependency-light description of the intended GPU benchmark."""

    validate_gpu_benchmark_axes(variants=variants, kernels=kernels)
    validate_gpu_benchmark_numbers(
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
    )
    _validate_positive_int("vllm_max_model_len", vllm_max_model_len)
    if not 0 < vllm_gpu_memory_utilization <= 1:
        raise ValueError("vllm_gpu_memory_utilization must be between 0 and 1")
    planned_runs, skipped_runs = _benchmark_combinations(
        models=models,
        variants=variants,
        kernels=kernels,
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
        "vllm_max_model_len": vllm_max_model_len,
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
        "total_runs": len(planned_runs),
        "skipped_runs": skipped_runs,
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
        f"vLLM max len:{plan['vllm_max_model_len']}",
        f"Total runs:   {plan['total_runs']}",
        f"Skipped:      {len(plan['skipped_runs'])}",
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


def _parse_nvidia_smi_used_memory_gib(stdout: str) -> float | None:
    values: list[float] = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values.append(float(text.split()[0]) / 1024)
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return values[0]


def _nvidia_smi_used_memory_gib(device_index: int | None = None) -> float | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    if device_index is not None:
        command.insert(1, f"--id={device_index}")

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 and device_index is not None:
        return _nvidia_smi_used_memory_gib(None)
    if completed.returncode != 0:
        return None
    return _parse_nvidia_smi_used_memory_gib(completed.stdout)


def _positive_delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    delta = current - baseline
    return delta if delta > 0 else None


def _max_positive(*values: float | None) -> float | None:
    positive = [value for value in values if value is not None and value > 0]
    return max(positive) if positive else None


def _parse_vllm_model_memory_gib(log_path: str) -> float | None:
    """Parse 'Model loading took X GiB' from a vLLM engine log file."""
    import re

    try:
        with open(log_path) as fh:
            return _parse_vllm_model_memory_from_lines(fh)
    except OSError:
        pass
    return None


def _parse_vllm_model_memory_from_lines(lines: Iterable[str]) -> float | None:
    """Parse 'Model loading took X GiB' from lines of text."""
    import re

    for line in lines:
        m = re.search(r"Model loading took\s+([\d.]+)\s+GiB", line)
        if m:
            return float(m.group(1))
    return None


def _running_in_wsl() -> bool:
    return "microsoft" in platform.release().lower()


def _prepare_vllm_wsl_runtime() -> None:
    if not _running_in_wsl():
        return

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if not os.environ.get("CC"):
        compiler = shutil.which("gcc") or shutil.which("cc")
        if compiler:
            os.environ["CC"] = compiler


def _allow_vllm_uva_when_probe_succeeds() -> None:
    if not _running_in_wsl():
        return

    try:
        import torch
        import vllm.utils.platform_utils as platform_utils
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        probe = torch.zeros(1, dtype=torch.int32, device="cpu", pin_memory=True)
        get_accelerator_view_from_cpu_tensor(probe)
        platform_utils.is_uva_available.cache_clear()
        platform_utils.is_uva_available = lambda: True

        try:
            import vllm.v1.worker.gpu.buffer_utils as buffer_utils
        except ImportError:
            return
        buffer_utils.is_uva_available = lambda: True
    except Exception:
        return


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


def _kill_orphan_procs() -> None:
    """Kill leftover multiprocessing children / resource-trackers holding GPU memory.
    
    Uses ``fuser`` to find processes touching ``/dev/nvidia*`` that are not ours,
    then SIGKILLs them.  Falls back to scanning ``/proc`` if fuser is unavailable.
    """
    import signal

    me = os.getpid()
    killed: set[int] = set()

    try:
        completed = subprocess.run(
            ["fuser", "-v", "/dev/nvidia*"],
            capture_output=True, text=True, timeout=5,
        )
        for token in completed.stdout.split():
            try:
                pid = int(token)
                if pid != me and pid != 1 and pid not in killed:
                    os.kill(pid, signal.SIGKILL)
                    killed.add(pid)
            except (ValueError, OSError):
                continue
    except (OSError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: scan /proc for known vLLM resource-tracker / leftover workers.
    if not killed:
        try:
            children = list(Path("/proc").glob("[0-9]*"))
        except OSError:
            children = []
        for child_dir in children:
            try:
                pid = int(child_dir.name)
                if pid == me or pid == 1 or pid in killed:
                    continue
                cmdline = (child_dir / "cmdline").read_text(errors="replace")
                if (
                    "multiprocessing.resource_tracker" in cmdline
                    or "vllm.entrypoints" in cmdline
                    or "vllm/v1/engine" in cmdline
                ):
                    os.kill(pid, signal.SIGKILL)
                    killed.add(pid)
            except (OSError, ValueError):
                continue

    if killed:
        time.sleep(1)  # give the OS a moment to reclaim GPU memory


def _force_cuda_cleanup() -> None:
    """Aggressive GPU cleanup, including orphan multiprocessing children.

    When vLLM's ``LLM()`` constructor fails partway through (e.g. OOM during
    cache-block allocation), background worker processes may keep GPU memory
    alive and poison subsequent benchmark runs.  This helper kills any
    leftover children and forces CUDA memory release.
    """

    gc.collect()
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

    import multiprocessing

    for child in multiprocessing.active_children():
        try:
            child.terminate()
        except Exception:
            pass
    for child in multiprocessing.active_children():
        try:
            child.join(timeout=5)
        except Exception:
            pass


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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_vllm_benchmark(
    *,
    model_name: str,
    variant: str,
    prompts: tuple[str, ...],
    max_new_tokens: int,
    warmup_runs: int,
    repeat_runs: int,
    trust_remote_code: bool,
    vllm_max_model_len: int,
    vllm_gpu_memory_utilization: float,
    vllm_enforce_eager: bool,
) -> GPUBenchmarkRun:
    """Benchmark a variant by launching ``vllm serve`` as a subprocess.

    Running vLLM in a subprocess guarantees the OS reclaims every byte of GPU
    memory when the server exits — no cross-run contamination.
    """

    _prepare_vllm_wsl_runtime()
    _force_cuda_cleanup()

    import torch

    device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
    baseline_used_gib = _nvidia_smi_used_memory_gib(device_index)

    vllm_binary = shutil.which("vllm")
    if vllm_binary is None:
        raise RuntimeError("Could not find the vllm binary on PATH")

    port = _find_free_port()

    cmd: list[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--dtype",
        "float16" if variant == "fp16" else "bfloat16",
        "--max-model-len",
        str(vllm_max_model_len),
        "--gpu-memory-utilization",
        str(vllm_gpu_memory_utilization),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if vllm_enforce_eager:
        cmd.append("--enforce-eager")
    if variant in {"fp8-dynamic", "fp8-dynamic-kv"}:
        cmd.append("--quantization=fp8_per_block")
    if variant == "fp8-dynamic-kv":
        cmd.append("--kv-cache-dtype=fp8")

    load_start = time.perf_counter()
    print(f"  Starting vLLM server for {variant} on port {port} (enforce_eager={vllm_enforce_eager}) ...", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    def _kill_server() -> None:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        # Aggressively kill any remaining child processes (e.g. multiprocessing
        # resource tracker or orphan EngineCore workers) that may hold GPU memory.
        _kill_orphan_procs()

    base_url = f"http://127.0.0.1:{port}"
    health_url = f"{base_url}/health"

    try:
        deadline = time.perf_counter() + 7200
        waited = 0
        while time.perf_counter() < deadline:
            # Check if the subprocess crashed
            poll_code = proc.poll()
            if poll_code is not None:
                _kill_server()
                raise RuntimeError(
                    f"vLLM server exited with code {poll_code} before becoming healthy. "
                    "Check the error output above."
                )
            try:
                urllib.request.urlopen(health_url, timeout=5)
                break
            except Exception:
                time.sleep(2)
                waited += 2
                if waited % 30 == 0:
                    print(f"  Waiting for vLLM server ... ({waited}s elapsed)", flush=True)
        else:
            _kill_server()
            raise RuntimeError(f"vLLM server did not become healthy within 120 min at {base_url}")

        load_seconds = time.perf_counter() - load_start
        load_used_gib = _nvidia_smi_used_memory_gib(device_index)
        load_delta_gib = _positive_delta(load_used_gib, baseline_used_gib)
        peak_used_gib = load_used_gib

        payload = json.dumps(
            {
                "model": model_name,
                "prompt": list(prompts),
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            }
        ).encode("utf-8")

        completion_url = f"{base_url}/v1/completions"

        def _http_generate() -> tuple[int, int, str]:
            req = urllib.request.Request(
                completion_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            usage = body.get("usage", {})
            in_tokens = int(usage.get("prompt_tokens", 0))
            out_tokens = int(usage.get("completion_tokens", 0))
            choices = body.get("choices", [{}])
            text = choices[0].get("text", "") if choices else ""
            return in_tokens, out_tokens, text

        for _ in range(warmup_runs):
            _http_generate()
            warmup_used_gib = _nvidia_smi_used_memory_gib(device_index)
            if warmup_used_gib is not None:
                peak_used_gib = max(peak_used_gib or warmup_used_gib, warmup_used_gib)

        durations: list[float] = []
        input_tokens = 0
        generated_tokens = 0
        response_preview = ""
        for _ in range(repeat_runs):
            start = time.perf_counter()
            i_tok, g_tok, text = _http_generate()
            durations.append(time.perf_counter() - start)
            input_tokens = i_tok
            generated_tokens = g_tok
            response_preview = text[:400]
            repeat_used_gib = _nvidia_smi_used_memory_gib(device_index)
            if repeat_used_gib is not None:
                peak_used_gib = max(peak_used_gib or repeat_used_gib, repeat_used_gib)

    finally:
        _kill_server()

    peak_delta_gib = _positive_delta(peak_used_gib, baseline_used_gib)
    model_memory_gib = _max_positive(load_delta_gib, peak_delta_gib)

    _force_cuda_cleanup()

    mean_seconds = sum(durations) / len(durations)
    return GPUBenchmarkRun(
        model=model_name,
        variant=variant,
        variant_label=VARIANT_LABELS[variant],
        kernel="vllm",
        kernel_label=KERNEL_LABELS["vllm"],
        status="ok",
        load_seconds=load_seconds,
        generation_seconds_mean=mean_seconds,
        generation_seconds_min=min(durations),
        generated_tokens_per_second=generated_tokens / mean_seconds,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        model_memory_gib=model_memory_gib,
        peak_allocated_gib=peak_delta_gib,
        peak_reserved_gib=peak_delta_gib,
        response_preview=response_preview,
    )


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
    vllm_max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN,
    vllm_gpu_memory_utilization: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    vllm_enforce_eager: bool = False,
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
        if not _compatible_variant_kernel(variant=variant, kernel=kernel):
            return GPUBenchmarkRun(
                **{
                    **asdict(label),
                    "status": "skipped",
                    "error": _skipped_variant_kernel_reason(variant=variant, kernel=kernel),
                }
            )
        _require_cuda_stack()
        if kernel == "vllm":
            if importlib.util.find_spec("vllm") is None:
                raise RuntimeError("vLLM is not installed.")
            return _run_vllm_benchmark(
                model_name=model_name,
                variant=variant,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                warmup_runs=warmup_runs,
                repeat_runs=repeat_runs,
                trust_remote_code=trust_remote_code,
                vllm_max_model_len=vllm_max_model_len,
                vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                vllm_enforce_eager=vllm_enforce_eager,
            )

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
        _force_cuda_cleanup() if kernel == "vllm" else _clear_cuda()
        return GPUBenchmarkRun(
            **{
                **asdict(label),
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}",
            }
        )


def _memory_basis_gib(run: GPUBenchmarkRun) -> float | None:
    if run.model_memory_gib and run.model_memory_gib > 0:
        return run.model_memory_gib
    if run.peak_allocated_gib and run.peak_allocated_gib > 0:
        return run.peak_allocated_gib
    return None


def _apply_compression_ratios(runs: list[GPUBenchmarkRun]) -> list[GPUBenchmarkRun]:
    baselines: dict[tuple[str, str], float] = {}
    for run in runs:
        baseline = _memory_basis_gib(run)
        if run.status == "ok" and run.variant == "bf16" and baseline:
            baselines[(run.model, run.kernel)] = baseline

    updated = []
    for run in runs:
        ratio = None
        baseline = baselines.get((run.model, run.kernel))
        current = _memory_basis_gib(run)
        if baseline and current:
            ratio = baseline / current
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
    vllm_max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN,
    vllm_gpu_memory_utilization: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    vllm_enforce_eager: bool = False,
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
        vllm_max_model_len=vllm_max_model_len,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    environment = collect_gpu_environment()
    runs: list[GPUBenchmarkRun] = []
    planned_runs, skipped_runs = _benchmark_combinations(
        models=models,
        variants=variants,
        kernels=kernels,
    )

    for skipped in skipped_runs:
        runs.append(
            GPUBenchmarkRun(
                model=skipped["model"],
                variant=skipped["variant"],
                variant_label=VARIANT_LABELS[skipped["variant"]],
                kernel=skipped["kernel"],
                kernel_label=KERNEL_LABELS[skipped["kernel"]],
                status="skipped",
                error=skipped["reason"],
            )
        )

    for planned in planned_runs:
        run = run_single_gpu_benchmark(
            model_name=planned["model"],
            variant=planned["variant"],
            kernel=planned["kernel"],
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            repeat_runs=repeat_runs,
            trust_remote_code=trust_remote_code,
            vllm_max_model_len=vllm_max_model_len,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_enforce_eager=vllm_enforce_eager,
        )
        runs.append(run)
        runs = _apply_compression_ratios(runs)
        payload = _result_payload(environment=environment, config=config, runs=runs)
        write_gpu_benchmark_json(payload, output_json)
        write_gpu_benchmark_report(payload, report_html)
        if fail_fast and run.status not in {"ok", "skipped"}:
            return payload

    payload = _result_payload(environment=environment, config=config, runs=runs)
    write_gpu_benchmark_json(payload, output_json)
    write_gpu_benchmark_report(payload, report_html)
    return payload


def _summary_run(run: GPUBenchmarkRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    payload = asdict(run)
    payload["memory_basis_gib"] = _memory_basis_gib(run)
    return payload


def summarize_gpu_benchmark_results(runs: list[GPUBenchmarkRun]) -> dict[str, Any]:
    ok_runs = [run for run in runs if run.status == "ok"]
    skipped_runs = [run for run in runs if run.status == "skipped"]
    failed_runs = [run for run in runs if run.status not in {"ok", "skipped"}]
    fastest = max(
        ok_runs,
        key=lambda run: run.generated_tokens_per_second or 0,
        default=None,
    )
    lowest_memory = min(
        (run for run in ok_runs if _memory_basis_gib(run)),
        key=lambda run: _memory_basis_gib(run) or float("inf"),
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
        "skipped_runs": len(skipped_runs),
        "failed_runs": len(failed_runs),
        "fastest": _summary_run(fastest),
        "lowest_memory": _summary_run(lowest_memory),
        "best_compression": _summary_run(best_compression),
    }


def format_gpu_benchmark_summary(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "GPU benchmark summary",
        "---------------------",
        "Runs:       "
        f"{summary['ok_runs']} ok / {summary['skipped_runs']} skipped / "
        f"{summary['failed_runs']} failed",
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
        memory_gib = lowest.get("memory_basis_gib") or lowest.get("peak_allocated_gib")
        if memory_gib:
            lines.append(
                "Lowest mem: "
                f"{lowest['model']} {lowest['variant']} {lowest['kernel']} "
                f"at {memory_gib:.2f} GiB"
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


def _report_memory_gib(row: dict[str, Any]) -> float | None:
    for key in ("peak_allocated_gib", "model_memory_gib", "peak_reserved_gib"):
        value = row.get(key)
        if isinstance(value, int | float) and value > 0:
            return float(value)
    return None


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


def _numeric(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _best_fp8_speedup(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], float] | None:
    bf16_by_model_kernel: dict[tuple[Any, Any], float] = {}
    for row in rows:
        throughput = _numeric(row.get("generated_tokens_per_second"))
        if row.get("status") == "ok" and row.get("variant") == "bf16" and throughput:
            bf16_by_model_kernel[(row.get("model"), row.get("kernel"))] = throughput

    comparisons: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        variant = str(row.get("variant", ""))
        throughput = _numeric(row.get("generated_tokens_per_second"))
        baseline = bf16_by_model_kernel.get((row.get("model"), row.get("kernel")))
        if row.get("status") == "ok" and variant.startswith("fp8") and throughput and baseline:
            comparisons.append((row, throughput / baseline))

    return max(comparisons, key=lambda item: item[1], default=None)


def _report_conclusions(payload: dict[str, Any]) -> str:
    rows = payload.get("runs", [])
    summary = payload.get("summary", {})
    config = payload.get("config", {})
    models = ", ".join(str(model) for model in config.get("models", [])) or "the selected model"
    variants = (
        ", ".join(str(variant) for variant in config.get("variants", [])) or "selected variants"
    )
    kernels = ", ".join(str(kernel) for kernel in config.get("kernels", [])) or "selected kernels"

    items = [
        "This report is self-contained for "
        f"{html.escape(models)} across {html.escape(variants)} on {html.escape(kernels)}."
    ]

    fastest = summary.get("fastest") or {}
    if fastest:
        items.append(
            "Fastest completed run: "
            f"{html.escape(_run_label(fastest))} at "
            f"{html.escape(_fmt(fastest.get('generated_tokens_per_second'), suffix=' tok/s'))}."
        )
    else:
        items.append("No completed run is available yet; inspect failed/skipped rows below.")

    fp8_speedup = _best_fp8_speedup(rows)
    if fp8_speedup:
        row, speedup = fp8_speedup
        items.append(
            "Best FP8 throughput comparison: "
            f"{html.escape(_run_label(row))} measured {speedup:.2f}x the BF16 "
            "throughput for the same model and kernel."
        )
    elif any(str(row.get("variant", "")).startswith("fp8") for row in rows):
        items.append(
            "FP8 variants were included, but no successful BF16/FP8 pair was available "
            "for a direct speedup calculation."
        )

    failed_runs = int(summary.get("failed_runs") or 0)
    skipped_runs = int(summary.get("skipped_runs") or 0)
    if failed_runs or skipped_runs:
        items.append(
            f"Run status: {failed_runs} failed and {skipped_runs} skipped; the Measurements "
            "table keeps the first error line beside each run."
        )

    vllm_memory_missing = any(
        row.get("status") == "ok"
        and row.get("kernel") == "vllm"
        and row.get("peak_allocated_gib") is None
        for row in rows
    )
    vllm_memory_present = any(
        row.get("status") == "ok"
        and row.get("kernel") == "vllm"
        and (
            row.get("peak_allocated_gib") is not None
            or row.get("model_memory_gib") is not None
            or row.get("peak_reserved_gib") is not None
        )
        for row in rows
    )
    if vllm_memory_present:
        items.append(
            "For vLLM rows, memory falls back to an nvidia-smi used-memory delta when "
            "torch CUDA allocator counters do not see the engine process."
        )
    elif vllm_memory_missing:
        items.append(
            "vLLM memory cells are blank because neither torch CUDA counters nor the "
            "nvidia-smi fallback returned a usable allocation delta."
        )

    return (
        '<section><h2>Conclusions</h2><ul class="conclusions">'
        + "".join(f"<li>{item}</li>" for item in items)
        + "</ul></section>"
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
            f"<td>{_fmt(_report_memory_gib(row))}</td>"
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
        ("Lowest memory", lowest, "memory_basis_gib", " GiB"),
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
    .conclusions { margin: 0; padding-left: 20px; }
    .conclusions li { margin: 9px 0; }
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
    conclusions_html = _report_conclusions(payload)
    chart_rows = [{**row, "report_memory_gib": _report_memory_gib(row)} for row in ok_rows]
    throughput_chart = _bar_chart(
        rows=chart_rows,
        metric="generated_tokens_per_second",
        title="Generation Throughput",
        unit=" tok/s",
    )
    memory_chart = _bar_chart(
        rows=chart_rows,
        metric="report_memory_gib",
        title="Peak Allocated GPU Memory",
        unit=" GiB",
        lower_is_better=True,
    )
    compression_chart = _bar_chart(
        rows=ok_rows,
        metric="compression_ratio_vs_bf16",
        title="Compression Ratio vs BF16 Memory Footprint",
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
    {conclusions_html}
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

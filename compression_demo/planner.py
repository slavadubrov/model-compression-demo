"""Memory and algorithm planning helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .catalog import ALGORITHMS, ARCH_TO_COMPUTE_CAPABILITY, GPU_INSTANCES, SCHEMES, GPUInstance

BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class MemoryEstimate:
    params_b: float
    scheme_key: str
    weight_gib: float
    kv_cache_gib: float
    runtime_overhead_gib: float
    safety_buffer_gib: float
    total_gib: float


@dataclass(frozen=True)
class CompressionMemoryEstimate:
    cpu_gib: float
    gpu_gib: float
    notes: str


@dataclass(frozen=True)
class InstanceRecommendation:
    instance: GPUInstance
    fits: bool
    headroom_gib: float
    reason: str


@dataclass(frozen=True)
class LocalRuntimeRecommendation:
    name: str
    memory_target_gib: float
    reason: str


@dataclass(frozen=True)
class CompressionPlan:
    algorithm_key: str
    scheme_key: str
    serving_memory: MemoryEstimate
    compression_memory: CompressionMemoryEstimate
    recommendations: tuple[InstanceRecommendation, ...]
    local_recommendations: tuple[LocalRuntimeRecommendation, ...]
    serving_target_label: str
    notes: tuple[str, ...]


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def estimate_weight_gib(params_b: float, scheme_key: str) -> float:
    """Estimate checkpoint weight memory in GiB for a quantization scheme."""

    _validate_positive("params_b", params_b)
    scheme = SCHEMES[scheme_key]
    bytes_per_param = scheme.effective_weight_bits / 8.0
    return params_b * 1_000_000_000 * bytes_per_param / BYTES_PER_GIB


def estimate_kv_cache_gib(
    *,
    layers: int,
    hidden_size: int,
    context_tokens: int,
    concurrency: int,
    kv_cache_bits: float = 16,
    kv_head_ratio: float = 1.0,
) -> float:
    """Estimate KV cache memory in GiB.

    The formula uses 2 tensors per token, one key and one value. `kv_head_ratio`
    lets grouped-query attention users reduce the effective hidden dimension.
    Keep the default at 1.0 when architecture details are unknown.
    """

    _validate_positive("layers", layers)
    _validate_positive("hidden_size", hidden_size)
    _validate_positive("context_tokens", context_tokens)
    _validate_positive("concurrency", concurrency)
    _validate_positive("kv_cache_bits", kv_cache_bits)
    _validate_positive("kv_head_ratio", kv_head_ratio)
    effective_hidden = hidden_size * kv_head_ratio
    bytes_total = (
        2 * layers * effective_hidden * context_tokens * concurrency * (kv_cache_bits / 8.0)
    )
    return bytes_total / BYTES_PER_GIB


def estimate_serving_memory(
    *,
    params_b: float,
    scheme_key: str,
    layers: int,
    hidden_size: int,
    context_tokens: int,
    concurrency: int,
    kv_cache_bits: float = 16,
    kv_head_ratio: float = 1.0,
) -> MemoryEstimate:
    """Estimate total GPU memory needed for serving."""

    weight_gib = estimate_weight_gib(params_b, scheme_key)
    kv_cache_gib = estimate_kv_cache_gib(
        layers=layers,
        hidden_size=hidden_size,
        context_tokens=context_tokens,
        concurrency=concurrency,
        kv_cache_bits=kv_cache_bits,
        kv_head_ratio=kv_head_ratio,
    )
    runtime_overhead_gib = max(2.0, weight_gib * 0.15)
    subtotal = weight_gib + kv_cache_gib + runtime_overhead_gib
    safety_buffer_gib = max(1.0, subtotal * 0.10)
    total_gib = subtotal + safety_buffer_gib
    return MemoryEstimate(
        params_b=params_b,
        scheme_key=scheme_key,
        weight_gib=weight_gib,
        kv_cache_gib=kv_cache_gib,
        runtime_overhead_gib=runtime_overhead_gib,
        safety_buffer_gib=safety_buffer_gib,
        total_gib=total_gib,
    )


def estimate_compression_memory(
    *,
    params_b: float,
    algorithm_key: str,
    layers: int,
    hidden_size: int | None = None,
) -> CompressionMemoryEstimate:
    """Estimate CPU and GPU memory for an offline compression job."""

    _validate_positive("params_b", params_b)
    _validate_positive("layers", layers)
    algorithm = ALGORITHMS[algorithm_key]
    fp16_model_gib = params_b * 1_000_000_000 * 2 / BYTES_PER_GIB

    if algorithm.scheme_key == "gguf-q4":
        cpu_gib = fp16_model_gib * 1.20
        notes = (
            "GGUF conversion is a local runtime workflow. Plan for CPU RAM, disk, "
            "and optional Apple unified memory; no CUDA compression GPU is required."
        )
        return CompressionMemoryEstimate(cpu_gib=cpu_gib, gpu_gib=0.0, notes=notes)

    cpu_gib = fp16_model_gib * 1.20

    layer_weight_gib = fp16_model_gib / layers
    hessian_multiplier = 2.0 if "gptq" in algorithm.key or "sparse" in algorithm.family else 1.15
    gpu_gib = max(2.0, layer_weight_gib * hessian_multiplier + 1.5)
    if hidden_size and "gptq" in algorithm.key:
        # GPTQ's auxiliary matrices scale with the largest onloaded linear layer.
        matrix_gib = (hidden_size * hidden_size * 2) / BYTES_PER_GIB
        gpu_gib = max(gpu_gib, matrix_gib * 2 + 1.5)

    notes = (
        "Text decoder compression usually onloads one layer at a time. "
        "CPU or disk still needs room for the source model; GPTQ-like methods "
        "add hessian memory on GPU."
    )
    return CompressionMemoryEstimate(cpu_gib=cpu_gib, gpu_gib=gpu_gib, notes=notes)


def recommend_local_runtimes(*, required_gib: float) -> tuple[LocalRuntimeRecommendation, ...]:
    """Return local CPU and Apple runtime memory targets for GGUF-style deployment."""

    _validate_positive("required_gib", required_gib)
    cpu_target = max(8.0, required_gib * 1.20)
    apple_target = max(16.0, required_gib * 1.25)
    return (
        LocalRuntimeRecommendation(
            name="llama.cpp / Ollama on CPU",
            memory_target_gib=cpu_target,
            reason=(
                f"Plan for about {cpu_target:.1f} GiB system RAM for Q4 GGUF weights, "
                "KV cache, runtime overhead, and OS headroom."
            ),
        ),
        LocalRuntimeRecommendation(
            name="MLX-LM / Ollama on Apple Silicon",
            memory_target_gib=apple_target,
            reason=(
                f"Plan for about {apple_target:.1f} GiB unified memory; leave extra headroom "
                "for long context and other desktop workloads."
            ),
        ),
    )


def recommend_instances(
    *,
    required_gib: float,
    min_compute_capability: float,
    max_results: int = 4,
) -> tuple[InstanceRecommendation, ...]:
    """Return the smallest matching GPUs with enough memory and hardware support."""

    _validate_positive("required_gib", required_gib)
    candidates = [
        gpu
        for gpu in GPU_INSTANCES
        if gpu.memory_gib > 0 and gpu.compute_capability >= min_compute_capability
    ]
    candidates.sort(key=lambda gpu: (gpu.memory_gib, gpu.compute_capability, gpu.name))

    recommendations: list[InstanceRecommendation] = []
    for gpu in candidates:
        fits = gpu.memory_gib >= required_gib
        if fits:
            recommendations.append(
                InstanceRecommendation(
                    instance=gpu,
                    fits=True,
                    headroom_gib=gpu.memory_gib - required_gib,
                    reason=f"Fits with {gpu.memory_gib - required_gib:.1f} GiB headroom.",
                )
            )
        if len(recommendations) >= max_results:
            break

    if recommendations:
        return tuple(recommendations)

    largest = max(candidates, key=lambda gpu: gpu.memory_gib) if candidates else None
    if largest is None:
        return ()
    shards = math.ceil(required_gib / largest.memory_gib)
    return (
        InstanceRecommendation(
            instance=largest,
            fits=False,
            headroom_gib=largest.memory_gib - required_gib,
            reason=(
                f"No single listed GPU fits. Approximate tensor-parallel serving needs {shards} "
                f"x {largest.name}; offline llm-compressor compression does not "
                "currently shard this way."
            ),
        ),
    )


def select_algorithm(
    *,
    goal: str,
    hardware: str,
    deployment: str = "vllm",
) -> str:
    """Choose a pragmatic default algorithm for the requested goal and hardware."""

    hw = hardware.lower()
    dep = deployment.lower()
    goal = goal.lower()
    cc = ARCH_TO_COMPUTE_CAPABILITY.get(hw)
    if cc is None:
        expected = ", ".join(sorted(ARCH_TO_COMPUTE_CAPABILITY))
        raise ValueError(f"Unknown hardware '{hardware}'. Expected one of: {expected}")

    if hw in {"cpu", "apple"}:
        return "gguf-q4"
    if "fine" in goal or "qlora" in goal:
        return "bnb-nf4"
    if "transformers" in dep and "production" not in goal:
        return "bnb-nf4"
    if cc >= 10.0 and ("maximum" in goal or "lowest" in goal):
        return "fp8-dynamic"
    if cc >= 8.9 and ("throughput" in goal or "latency" in goal):
        return "fp8-dynamic"
    if "quality" in goal and cc >= 7.5:
        return "smoothquant-w8a8"
    if "memory" in goal or "lowest" in goal or "fit" in goal:
        return "gptq-w4a16"
    return "gptq-w4a16"


def build_plan(
    *,
    params_b: float,
    algorithm_key: str,
    layers: int,
    hidden_size: int,
    context_tokens: int,
    concurrency: int,
    kv_cache_bits: float = 16,
    kv_head_ratio: float = 1.0,
) -> CompressionPlan:
    """Build a complete compression and serving plan."""

    algorithm = ALGORITHMS[algorithm_key]
    scheme = SCHEMES[algorithm.scheme_key]
    serving = estimate_serving_memory(
        params_b=params_b,
        scheme_key=algorithm.scheme_key,
        layers=layers,
        hidden_size=hidden_size,
        context_tokens=context_tokens,
        concurrency=concurrency,
        kv_cache_bits=kv_cache_bits,
        kv_head_ratio=kv_head_ratio,
    )
    compression = estimate_compression_memory(
        params_b=params_b,
        algorithm_key=algorithm_key,
        layers=layers,
        hidden_size=hidden_size,
    )
    if algorithm.scheme_key == "gguf-q4":
        recommendations = ()
        local_recommendations = recommend_local_runtimes(required_gib=serving.total_gib)
        serving_target_label = "RAM / unified memory target"
        runtime_note = (
            "Use llama.cpp, Ollama, or MLX-LM for local deployment instead of CUDA server GPUs."
        )
    else:
        recommendations = recommend_instances(
            required_gib=serving.total_gib,
            min_compute_capability=scheme.min_compute_capability,
        )
        local_recommendations = ()
        serving_target_label = "GPU memory target"
        runtime_note = "Use a real serving load test on the target CUDA runtime before rollout."

    notes = (
        "Validate quality with task metrics and perplexity before production rollout.",
        "Treat KV cache as a first-class memory term for long context or concurrent serving.",
        "Use the compression estimate for the offline quantization job and the serving "
        "estimate for the deployed endpoint.",
        runtime_note,
    )
    return CompressionPlan(
        algorithm_key=algorithm_key,
        scheme_key=algorithm.scheme_key,
        serving_memory=serving,
        compression_memory=compression,
        recommendations=recommendations,
        local_recommendations=local_recommendations,
        serving_target_label=serving_target_label,
        notes=notes,
    )

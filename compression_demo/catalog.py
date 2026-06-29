"""Catalogs for industrial LLM compression choices.

The values here are intentionally conservative. They are used by the demo
planner and by the HTML guide to explain first-pass sizing, not to replace a
load test on the target stack.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationScheme:
    key: str
    label: str
    weight_bits: float
    activation_bits: float
    quantized_weight_fraction: float
    min_compute_capability: float
    calibration: str
    package: str
    serving: str
    summary: str

    @property
    def effective_weight_bits(self) -> float:
        """Average bits per parameter after leaving sensitive tensors in fp16."""

        q = self.quantized_weight_fraction
        return (self.weight_bits * q) + (16.0 * (1.0 - q))


@dataclass(frozen=True)
class CompressionAlgorithm:
    key: str
    name: str
    family: str
    package: str
    scheme_key: str
    calibration: str
    best_for: str
    avoid_when: str
    status: str


@dataclass(frozen=True)
class GPUInstance:
    name: str
    memory_gib: float
    compute_capability: float
    cloud_examples: str
    notes: str


SCHEMES: dict[str, QuantizationScheme] = {
    "bf16": QuantizationScheme(
        key="bf16",
        label="BF16/FP16 baseline",
        weight_bits=16,
        activation_bits=16,
        quantized_weight_fraction=0.0,
        min_compute_capability=7.0,
        calibration="none",
        package="Transformers, vLLM, TensorRT-LLM",
        serving="baseline serving",
        summary=(
            "Use as the quality reference and for high-throughput batches when memory "
            "is not the bottleneck."
        ),
    ),
    "w8a16": QuantizationScheme(
        key="w8a16",
        label="W8A16 weight-only",
        weight_bits=8,
        activation_bits=16,
        quantized_weight_fraction=0.90,
        min_compute_capability=8.0,
        calibration="optional",
        package="llm-compressor",
        serving="vLLM compressed-tensors",
        summary="Low-risk memory reduction for older Ampere-class production hardware.",
    ),
    "w4a16": QuantizationScheme(
        key="w4a16",
        label="W4A16 weight-only",
        weight_bits=4,
        activation_bits=16,
        quantized_weight_fraction=0.88,
        min_compute_capability=8.0,
        calibration="recommended",
        package="llm-compressor, GPTQModel",
        serving="vLLM compressed-tensors, Transformers, SGLang",
        summary=(
            "The usual first choice when fitting a larger model into a smaller GPU is "
            "the main constraint."
        ),
    ),
    "fp8-dynamic": QuantizationScheme(
        key="fp8-dynamic",
        label="W8A8 FP8 dynamic",
        weight_bits=8,
        activation_bits=8,
        quantized_weight_fraction=0.95,
        min_compute_capability=8.9,
        calibration="none or light",
        package="llm-compressor, NVIDIA ModelOpt",
        serving="vLLM, TensorRT-LLM",
        summary="Best default for high-throughput serving on Ada, Hopper, and newer GPUs.",
    ),
}


ALGORITHMS: dict[str, CompressionAlgorithm] = {
    "rtn-w8a16": CompressionAlgorithm(
        key="rtn-w8a16",
        name="Round-to-nearest W8A16",
        family="post-training quantization",
        package="llm-compressor QuantizationModifier",
        scheme_key="w8a16",
        calibration="none",
        best_for="Fast baseline compression and sanity checks before heavier calibration.",
        avoid_when="You need maximum compression or activation quantization.",
        status="production baseline",
    ),
    "gptq-w4a16": CompressionAlgorithm(
        key="gptq-w4a16",
        name="GPTQ W4A16",
        family="second-order post-training quantization",
        package="llm-compressor GPTQModifier",
        scheme_key="w4a16",
        calibration="small representative text set",
        best_for="Production weight-only int4 checkpoints for vLLM when memory is the bottleneck.",
        avoid_when=(
            "Compression GPU memory is extremely tight or the deployment is "
            "high-batch compute-bound."
        ),
        status="production default",
    ),
    "fp8-dynamic": CompressionAlgorithm(
        key="fp8-dynamic",
        name="Dynamic FP8 W8A8",
        family="floating-point post-training quantization",
        package="llm-compressor QuantizationModifier",
        scheme_key="fp8-dynamic",
        calibration="none or light",
        best_for="High-throughput modern NVIDIA serving where FP8 tensor cores are available.",
        avoid_when=(
            "The target GPU is older than Ada or Hopper, or your serving runtime lacks FP8 kernels."
        ),
        status="production default on modern GPUs",
    ),
}


GPU_INSTANCES: tuple[GPUInstance, ...] = (
    GPUInstance(
        "CPU / Apple Silicon",
        0,
        0.0,
        "local workstation",
        "Use llama.cpp, Ollama, or MLX-LM rather than CUDA-only stacks.",
    ),
    GPUInstance(
        "NVIDIA T4 16GB",
        16,
        7.5,
        "AWS g4dn, GCP T4",
        "Economy int8 and small-model experiments; tight for modern serving.",
    ),
    GPUInstance(
        "NVIDIA L4 24GB",
        24,
        8.9,
        "AWS g6, GCP L4",
        "Good small-to-mid model serving and FP8-capable Ada hardware.",
    ),
    GPUInstance(
        "NVIDIA A10G 24GB",
        24,
        8.6,
        "AWS g5",
        "Common Ampere serving GPU; good W4A16 target, no native FP8 path.",
    ),
    GPUInstance(
        "RTX 4090 24GB",
        24,
        8.9,
        "local workstation",
        "Excellent local Ada card; not a data-center deployment target.",
    ),
    GPUInstance(
        "NVIDIA L40S 48GB",
        48,
        8.9,
        "AWS g6e, GCP G2",
        "Ada serving with more headroom for context and concurrency.",
    ),
    GPUInstance(
        "NVIDIA A100 40GB",
        40,
        8.0,
        "AWS p4d, Azure ND A100",
        "Ampere production GPU for W4A16/W8A16 and tensor parallel serving.",
    ),
    GPUInstance(
        "NVIDIA A100 80GB",
        80,
        8.0,
        "AWS p4de, Azure NDm A100",
        "Large context or larger models without FP8 tensor cores.",
    ),
    GPUInstance(
        "NVIDIA H100 80GB",
        80,
        9.0,
        "AWS p5, GCP A3, Azure ND H100",
        "Strong default for FP8 high-throughput serving.",
    ),
    GPUInstance(
        "NVIDIA H200 141GB",
        141,
        9.0,
        "selected cloud bare metal",
        "Large-memory Hopper for long-context and bigger dense models.",
    ),
    GPUInstance(
        "NVIDIA B200 180GB",
        180,
        10.0,
        "Blackwell systems",
        "Target for NVFP4/MXFP4 and newest low-precision formats.",
    ),
)


ARCH_TO_COMPUTE_CAPABILITY = {
    "cpu": 0.0,
    "apple": 0.0,
    "pascal": 6.0,
    "volta": 7.0,
    "turing": 7.5,
    "ampere": 8.0,
    "ada": 8.9,
    "hopper": 9.0,
    "blackwell": 10.0,
}

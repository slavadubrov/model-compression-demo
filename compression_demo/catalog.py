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
    "w8a8-int8": QuantizationScheme(
        key="w8a8-int8",
        label="W8A8 INT8",
        weight_bits=8,
        activation_bits=8,
        quantized_weight_fraction=0.92,
        min_compute_capability=7.5,
        calibration="recommended",
        package="llm-compressor",
        serving="vLLM compressed-tensors",
        summary=(
            "Balanced compatibility path for Turing and newer GPUs when activations "
            "should also be quantized."
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
    "w4afp8": QuantizationScheme(
        key="w4afp8",
        label="W4A(FP8)",
        weight_bits=4,
        activation_bits=8,
        quantized_weight_fraction=0.92,
        min_compute_capability=8.9,
        calibration="recommended",
        package="llm-compressor",
        serving="vLLM support depends on exact compressed-tensors format",
        summary="Aggressive memory reduction with dynamic FP8 activations for modern NVIDIA GPUs.",
    ),
    "nvfp4-mxfp4": QuantizationScheme(
        key="nvfp4-mxfp4",
        label="NVFP4/MXFP4",
        weight_bits=4,
        activation_bits=4,
        quantized_weight_fraction=0.95,
        min_compute_capability=10.0,
        calibration="recommended",
        package="llm-compressor, NVIDIA ModelOpt",
        serving="Blackwell-class runtimes",
        summary=(
            "Newest maximum-compression path for Blackwell-class hardware. Treat as "
            "hardware-specific."
        ),
    ),
    "bnb-int8": QuantizationScheme(
        key="bnb-int8",
        label="bitsandbytes LLM.int8",
        weight_bits=8,
        activation_bits=16,
        quantized_weight_fraction=0.90,
        min_compute_capability=7.5,
        calibration="none",
        package="bitsandbytes + Transformers",
        serving="Transformers runtime, prototyping",
        summary=(
            "Simple runtime quantization when you do not need to export a production "
            "compressed checkpoint."
        ),
    ),
    "bnb-nf4": QuantizationScheme(
        key="bnb-nf4",
        label="bitsandbytes NF4/QLoRA",
        weight_bits=4,
        activation_bits=16,
        quantized_weight_fraction=0.90,
        min_compute_capability=6.0,
        calibration="none",
        package="bitsandbytes + PEFT",
        serving="fine-tuning and experiments",
        summary=(
            "Best known as the QLoRA path for low-memory fine-tuning, not as a vLLM "
            "production export."
        ),
    ),
    "gguf-q4": QuantizationScheme(
        key="gguf-q4",
        label="GGUF Q4/K-quants",
        weight_bits=4.5,
        activation_bits=16,
        quantized_weight_fraction=0.95,
        min_compute_capability=0.0,
        calibration="conversion-time",
        package="llama.cpp, Ollama, MLX-LM on Apple",
        serving="edge, CPU, Apple Silicon",
        summary=(
            "Preferred outside CUDA server stacks, especially local CPU or Apple "
            "Silicon deployment."
        ),
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
    "awq-w4a16": CompressionAlgorithm(
        key="awq-w4a16",
        name="AWQ W4A16",
        family="activation-aware weight quantization",
        package="llm-compressor AWQModifier + QuantizationModifier",
        scheme_key="w4a16",
        calibration="small representative text set",
        best_for="Instruction-tuned and multimodal-friendly int4 weight-only deployment.",
        avoid_when="The model family has no AWQ mappings and you cannot validate quality.",
        status="production option",
    ),
    "smoothquant-w8a8": CompressionAlgorithm(
        key="smoothquant-w8a8",
        name="SmoothQuant + INT8 W8A8",
        family="activation smoothing plus quantization",
        package="llm-compressor SmoothQuantModifier + QuantizationModifier",
        scheme_key="w8a8-int8",
        calibration="representative activation samples",
        best_for="Weight and activation int8 on broad GPU support, with better outlier handling.",
        avoid_when=(
            "You only need checkpoint size reduction; activation quantization adds validation work."
        ),
        status="production option",
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
    "autoround-w4a16": CompressionAlgorithm(
        key="autoround-w4a16",
        name="AutoRound W4A16",
        family="learned weight rounding",
        package="AutoRound, llm-compressor integrations where available",
        scheme_key="w4a16",
        calibration="representative text set",
        best_for="Researching learned rounding as an alternative to GPTQ/AWQ on supported stacks.",
        avoid_when="You need the shortest stable production path in this demo today.",
        status="recipe stub only; not executable in this demo",
    ),
    "kv-cache-fp8": CompressionAlgorithm(
        key="kv-cache-fp8",
        name="FP8 KV cache quantization",
        family="runtime cache compression",
        package="vLLM, llm-compressor experimental checkpoints",
        scheme_key="fp8-dynamic",
        calibration="runtime or checkpoint scales",
        best_for="Long-context or high-concurrency serving where KV cache dominates memory.",
        avoid_when=(
            "Your workload is short prompt, low concurrency, or quality regression is untested."
        ),
        status="advanced production/experimental by format",
    ),
    "nvfp4-mxfp4": CompressionAlgorithm(
        key="nvfp4-mxfp4",
        name="NVFP4/MXFP4 low-precision path",
        family="hardware-specific low precision",
        package="llm-compressor, NVIDIA ModelOpt, Blackwell-class runtimes",
        scheme_key="nvfp4-mxfp4",
        calibration="representative text set",
        best_for=(
            "Blackwell experiments where the runtime and checkpoint format explicitly support FP4."
        ),
        avoid_when="You are not on a verified Blackwell software and hardware stack.",
        status="recipe stub only; hardware-specific",
    ),
    "svdquant-nunchaku": CompressionAlgorithm(
        key="svdquant-nunchaku",
        name="SVDQuant / Nunchaku image-model path",
        family="diffusion and image-model compression",
        package="SVDQuant, Nunchaku",
        scheme_key="w4afp8",
        calibration="image-generation prompts and model-specific calibration",
        best_for="Image-model and diffusion compression outside the LLM serving workflow.",
        avoid_when="You are quantizing text-only LLMs for vLLM serving.",
        status="non-executable roadmap stub; separate runtime family",
    ),
    "bnb-nf4": CompressionAlgorithm(
        key="bnb-nf4",
        name="bitsandbytes NF4 / QLoRA",
        family="runtime quantization and fine-tuning",
        package="bitsandbytes + Transformers + PEFT",
        scheme_key="bnb-nf4",
        calibration="none",
        best_for="Low-memory fine-tuning and quick local experiments.",
        avoid_when="You need a portable vLLM compressed checkpoint.",
        status="popular alternative",
    ),
    "gptqmodel-w4a16": CompressionAlgorithm(
        key="gptqmodel-w4a16",
        name="GPTQModel GPTQ/AWQ W4A16",
        family="checkpoint quantization toolkit",
        package="GPTQModel",
        scheme_key="w4a16",
        calibration="small representative text set",
        best_for=(
            "Transformers, Optimum, PEFT, vLLM, and SGLang compatibility when not "
            "using llm-compressor."
        ),
        avoid_when="You want the vLLM project's compressed-tensors-first workflow.",
        status="current alternative to AutoGPTQ and AutoAWQ",
    ),
    "gguf-q4": CompressionAlgorithm(
        key="gguf-q4",
        name="GGUF Q4/K-quants",
        family="edge quantized file format",
        package="llama.cpp, Ollama, MLX-LM",
        scheme_key="gguf-q4",
        calibration="conversion-time",
        best_for="CPU, Apple Silicon, desktop, and edge deployment.",
        avoid_when="You require vLLM continuous batching on server GPUs.",
        status="popular edge alternative",
    ),
    "distillation": CompressionAlgorithm(
        key="distillation",
        name="Knowledge distillation",
        family="model replacement",
        package="training pipeline dependent",
        scheme_key="bf16",
        calibration="task data and teacher outputs",
        best_for=(
            "Replacing a large model with a smaller trained student for stable recurring tasks."
        ),
        avoid_when="You need a quick checkpoint-only compression pass.",
        status="industrial but project-specific",
    ),
}


GPU_INSTANCES: tuple[GPUInstance, ...] = (
    GPUInstance(
        "CPU / Apple Silicon",
        0,
        0.0,
        "local workstation",
        "Use GGUF, llama.cpp, Ollama, or MLX-LM rather than CUDA-only stacks.",
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

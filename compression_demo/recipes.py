"""Recipe snippets and optional quantization runners."""

from __future__ import annotations

from textwrap import dedent

from .catalog import ALGORITHMS


def recipe_snippet(algorithm_key: str) -> str:
    """Return a copy-pasteable recipe snippet for an algorithm."""

    if algorithm_key not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_key}")

    snippets = {
        "gptq-w4a16": """
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import GPTQModifier

            recipe = GPTQModifier(
                scheme="W4A16",
                targets="Linear",
                ignore=["lm_head"],
            )

            oneshot(
                model="Qwen/Qwen3-0.6B",
                dataset="wikitext",
                dataset_config_name="wikitext-2-raw-v1",
                recipe=recipe,
                output_dir="outputs/Qwen3-0.6B-W4A16",
                max_seq_length=4096,
                num_calibration_samples=256,
            )
        """,
        "rtn-w8a16": """
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import QuantizationModifier

            recipe = QuantizationModifier(
                scheme="W8A16",
                targets="Linear",
                ignore=["lm_head"],
            )

            oneshot(
                model="Qwen/Qwen3-0.6B",
                recipe=recipe,
                output_dir="outputs/Qwen3-0.6B-W8A16",
            )
        """,
        "awq-w4a16": """
            from llmcompressor import oneshot
            from llmcompressor.modifiers.awq import AWQModifier
            from llmcompressor.modifiers.quantization import QuantizationModifier

            recipe = [
                AWQModifier(),
                QuantizationModifier(
                    scheme="W4A16_ASYM",
                    targets=["Linear"],
                    ignore=["lm_head"],
                ),
            ]

            oneshot(
                model="Qwen/Qwen3-0.6B",
                dataset="wikitext",
                dataset_config_name="wikitext-2-raw-v1",
                recipe=recipe,
                output_dir="outputs/Qwen3-0.6B-AWQ-W4A16",
                max_seq_length=4096,
                num_calibration_samples=256,
            )
        """,
        "smoothquant-w8a8": """
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import QuantizationModifier
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

            recipe = [
                SmoothQuantModifier(smoothing_strength=0.8),
                QuantizationModifier(
                    scheme="W8A8",
                    targets="Linear",
                    ignore=["lm_head"],
                ),
            ]

            oneshot(
                model="Qwen/Qwen3-0.6B",
                dataset="wikitext",
                dataset_config_name="wikitext-2-raw-v1",
                recipe=recipe,
                output_dir="outputs/Qwen3-0.6B-W8A8",
                max_seq_length=4096,
                num_calibration_samples=256,
            )
        """,
        "fp8-dynamic": """
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import QuantizationModifier
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "Qwen/Qwen3-0.6B"
            model = AutoModelForCausalLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            recipe = QuantizationModifier(
                targets="Linear",
                scheme="FP8_DYNAMIC",
                ignore=["lm_head"],
            )

            oneshot(model=model, recipe=recipe)
            model.save_pretrained("outputs/Qwen3-0.6B-FP8-Dynamic")
            tokenizer.save_pretrained("outputs/Qwen3-0.6B-FP8-Dynamic")
        """,
        "bnb-nf4": """
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            model_id = "Qwen/Qwen3-0.6B"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=quantization_config,
            )
        """,
        "gptqmodel-w4a16": """
            from datasets import load_dataset
            from gptqmodel import GPTQModel, QuantizeConfig

            model_id = "Qwen/Qwen3-0.6B"
            calibration = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:256]")
            quant_config = QuantizeConfig(bits=4, group_size=128)
            model = GPTQModel.load(model_id, quant_config)
            model.quantize(calibration)
            model.save("outputs/Qwen3-0.6B-GPTQModel-W4A16")
        """,
        "gguf-q4": """
            # llama.cpp example after building llama.cpp tools:
            python convert_hf_to_gguf.py Qwen/Qwen3-0.6B --outfile Qwen3-0.6B-f16.gguf
            ./llama-quantize Qwen3-0.6B-f16.gguf Qwen3-0.6B-Q4_K_M.gguf Q4_K_M
            ./llama-server -m Qwen3-0.6B-Q4_K_M.gguf -c 4096
        """,
        "kv-cache-fp8": """
            # vLLM serving-side example. Confirm the exact flag names for your vLLM version.
            vllm serve ./outputs/Qwen3-0.6B-FP8-Dynamic \\
              --kv-cache-dtype fp8 \\
              --max-model-len 32768
        """,
        "distillation": """
            # Distillation is a training workflow, not a checkpoint-only compression recipe.
            # Typical loop:
            # 1. Select task data and teacher model.
            # 2. Generate teacher logits or rationales.
            # 3. Fine-tune a smaller student.
            # 4. Evaluate the student against task metrics, latency, and cost.
        """,
    }
    return dedent(snippets[algorithm_key]).strip() + "\n"


def dry_run_quantization_command(
    *,
    algorithm_key: str,
    model: str,
    output_dir: str,
    dataset: str,
    dataset_config_name: str,
    num_calibration_samples: int,
    max_seq_length: int,
) -> str:
    """Return the intended quantization run configuration."""

    algorithm = ALGORITHMS[algorithm_key]
    return dedent(
        f"""
        Quantization dry run
        --------------------
        Algorithm: {algorithm.name}
        Package:   {algorithm.package}
        Model:     {model}
        Output:    {output_dir}
        Dataset:   {dataset} / {dataset_config_name}
        Samples:   {num_calibration_samples}
        Sequence:  {max_seq_length}

        Re-run without --dry-run in an environment with the GPU stack installed.
        """
    ).strip()


def run_llmcompressor_quantization(
    *,
    algorithm_key: str,
    model: str,
    output_dir: str,
    dataset: str,
    dataset_config_name: str,
    num_calibration_samples: int,
    max_seq_length: int,
) -> None:
    """Run a supported llm-compressor quantization job.

    Heavy dependencies are imported lazily so the planner and tests work on a
    normal Python installation.
    """

    if algorithm_key == "gptq-w4a16":
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier

        recipe = GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])
        oneshot(
            model=model,
            dataset=dataset,
            dataset_config_name=dataset_config_name,
            recipe=recipe,
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            num_calibration_samples=num_calibration_samples,
        )
        return

    if algorithm_key == "rtn-w8a16":
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier

        recipe = QuantizationModifier(scheme="W8A16", targets="Linear", ignore=["lm_head"])
        oneshot(model=model, recipe=recipe, output_dir=output_dir)
        return

    if algorithm_key == "fp8-dynamic":
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained(model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
        oneshot(model=hf_model, recipe=recipe)
        hf_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return

    raise NotImplementedError(
        f"Direct execution for {algorithm_key} is not wired into this demo. "
        "Use the recipe subcommand for the reference implementation."
    )

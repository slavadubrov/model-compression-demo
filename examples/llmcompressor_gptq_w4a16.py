"""L4-style GPTQ W4A16 quantization with llm-compressor.

Run this in a CUDA-capable environment after `uv sync --extra gpu`.
"""

from __future__ import annotations

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "outputs/Qwen3-0.6B-W4A16"


def main() -> None:
    recipe = GPTQModifier(
        scheme="W4A16",
        targets="Linear",
        ignore=["lm_head"],
    )
    oneshot(
        model=MODEL,
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        recipe=recipe,
        output_dir=OUTPUT_DIR,
        max_seq_length=4096,
        num_calibration_samples=256,
    )
    print(f"Quantized model written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

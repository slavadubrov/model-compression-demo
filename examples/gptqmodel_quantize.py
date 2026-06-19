"""GPTQModel alternative for GPTQ/AWQ-style exported checkpoints."""

from __future__ import annotations

from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

MODEL = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "outputs/Qwen3-0.6B-GPTQModel-W4A16"


def main() -> None:
    calibration = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:256]")
    quant_config = QuantizeConfig(bits=4, group_size=128)
    model = GPTQModel.load(MODEL, quant_config)
    model.quantize(calibration)
    model.save(OUTPUT_DIR)
    print(f"Quantized model written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

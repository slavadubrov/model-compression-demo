"""GPTQ W4A16 quantization with llm-compressor.

Run this in a CUDA-capable environment after installing the compression stack
listed in the README.
"""

from __future__ import annotations

import json
import pathlib

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "outputs/Qwen3-0.6B-W4A16"
CALIBRATION_FILE = pathlib.Path(__file__).with_name("representative_calibration.jsonl")


def load_calibration() -> Dataset:
    rows = [
        json.loads(line)
        for line in CALIBRATION_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return Dataset.from_list(rows)


def main() -> None:
    recipe = GPTQModifier(
        scheme="W4A16",
        targets="Linear",
        ignore=["lm_head"],
    )
    oneshot(
        model=MODEL,
        dataset=load_calibration(),
        recipe=recipe,
        output_dir=OUTPUT_DIR,
        max_seq_length=4096,
        num_calibration_samples=5,
    )
    print(f"Quantized model written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

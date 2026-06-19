"""Runtime 4-bit loading with bitsandbytes.

This is the practical alternative for experiments and QLoRA-style fine-tuning.
It does not create the same kind of vLLM production checkpoint as llm-compressor.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        quantization_config=quantization_config,
    )
    prompt = tokenizer("Model quantization is", return_tensors="pt").to(model.device)
    output = model.generate(**prompt, max_new_tokens=40, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

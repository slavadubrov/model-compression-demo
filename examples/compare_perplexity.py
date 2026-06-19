"""Perplexity comparison for base and quantized models.

This follows the L4 notebook pattern. Keep the token limit small for smoke tests
and raise it for a real evaluation.
"""

from __future__ import annotations

import argparse
import math

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_perplexity(model, tokenizer, dataset, max_tokens: int, stride: int) -> float:
    encodings = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = encodings.input_ids.to(model.device)
    nlls = []
    prev_end = 0
    for begin_loc in range(0, input_ids.size(1), stride):
        end_loc = min(begin_loc + stride, input_ids.size(1))
        trg_len = end_loc - prev_end
        input_slice = input_ids[:, begin_loc:end_loc]
        target_slice = input_slice.clone()
        target_slice[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(input_slice, labels=target_slice).loss
        nlls.append(loss * trg_len)
        prev_end = end_loc
    return math.exp(torch.stack(nlls).sum() / prev_end)


def load_model(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--compressed", required=True)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    base_model, tokenizer = load_model(args.base)
    compressed_model, compressed_tokenizer = load_model(args.compressed)
    base_ppl = calculate_perplexity(base_model, tokenizer, data, args.max_tokens, args.stride)
    compressed_ppl = calculate_perplexity(
        compressed_model, compressed_tokenizer, data, args.max_tokens, args.stride
    )
    print(f"Base perplexity:       {base_ppl:.2f}")
    print(f"Compressed perplexity: {compressed_ppl:.2f}")
    print(f"Delta:                 {compressed_ppl - base_ppl:+.2f}")


if __name__ == "__main__":
    main()

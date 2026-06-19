"""Quality evaluation helpers for compressed language models.

The public planning functions are dependency-free. Heavy ML imports happen only
inside execution functions so the CLI and tests stay usable on a normal Python
installation.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = (
    "Machine learning is a branch of",
    "Explain model quantization to an infrastructure engineer in two sentences.",
    "Summarize why KV cache memory matters for long-context LLM serving.",
)

QUALITY_EVAL_INSTALL_COMMAND = "uv pip install torch transformers datasets lm_eval"


@dataclass(frozen=True)
class QualityEvalPlan:
    base_model: str
    compressed_model: str
    mode: str
    checks: tuple[str, ...]
    prompts: tuple[str, ...]
    dataset: str
    dataset_config_name: str
    dataset_split: str
    lm_eval_task: str | None
    long_context_tokens: int
    required_modules: tuple[str, ...]
    output_json: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "compressed_model": self.compressed_model,
            "mode": self.mode,
            "checks": list(self.checks),
            "prompts": list(self.prompts),
            "dataset": self.dataset,
            "dataset_config_name": self.dataset_config_name,
            "dataset_split": self.dataset_split,
            "lm_eval_task": self.lm_eval_task,
            "long_context_tokens": self.long_context_tokens,
            "required_modules": list(self.required_modules),
            "output_json": self.output_json,
        }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _checks_for_mode(
    mode: str, lm_eval_task: str | None, long_context_tokens: int
) -> tuple[str, ...]:
    if mode == "generation":
        return ("generation comparison",)
    if mode == "perplexity":
        return ("perplexity comparison",)
    if mode == "long-context":
        return ("long-context anchor probe",)
    if mode == "lm-eval":
        return ("task metrics via lm_eval",)

    checks = ["generation comparison", "perplexity comparison"]
    if long_context_tokens > 0:
        checks.append("long-context anchor probe")
    if lm_eval_task:
        checks.append("task metrics via lm_eval")
    return tuple(checks)


def _required_modules(checks: tuple[str, ...]) -> tuple[str, ...]:
    modules = {"torch", "transformers"}
    if "perplexity comparison" in checks:
        modules.add("datasets")
    if "task metrics via lm_eval" in checks:
        modules.add("lm_eval")
    return tuple(sorted(modules))


def build_quality_eval_plan(
    *,
    base_model: str,
    compressed_model: str,
    mode: str = "all",
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    dataset: str = "wikitext",
    dataset_config_name: str = "wikitext-2-raw-v1",
    dataset_split: str = "test",
    lm_eval_task: str | None = None,
    long_context_tokens: int = 4096,
    output_json: str | None = None,
) -> QualityEvalPlan:
    checks = _checks_for_mode(mode, lm_eval_task, long_context_tokens)
    return QualityEvalPlan(
        base_model=base_model,
        compressed_model=compressed_model,
        mode=mode,
        checks=checks,
        prompts=prompts,
        dataset=dataset,
        dataset_config_name=dataset_config_name,
        dataset_split=dataset_split,
        lm_eval_task=lm_eval_task,
        long_context_tokens=long_context_tokens,
        required_modules=_required_modules(checks),
        output_json=output_json,
    )


def format_quality_eval_plan(plan: QualityEvalPlan) -> str:
    missing = [module for module in plan.required_modules if not _module_available(module)]
    lines = [
        "Quality evaluation plan",
        "-----------------------",
        f"Base model:       {plan.base_model}",
        f"Compressed model: {plan.compressed_model}",
        f"Mode:             {plan.mode}",
        "Checks:",
    ]
    lines.extend(f"  - {check}" for check in plan.checks)
    lines.extend(
        [
            "Prompts:",
            *[f"  - {prompt}" for prompt in plan.prompts],
            f"Dataset:          {plan.dataset} / {plan.dataset_config_name} / {plan.dataset_split}",
            f"Long context:     {plan.long_context_tokens} tokens",
            f"Output JSON:      {plan.output_json or '(not requested)'}",
            "Required modules:",
            *[f"  - {module}" for module in plan.required_modules],
        ]
    )
    if plan.lm_eval_task:
        lines.append(f"lm_eval task:     {plan.lm_eval_task}")
    if missing:
        lines.extend(
            [
                "Missing modules:",
                *[f"  - {module}" for module in missing],
                f"Install with: {QUALITY_EVAL_INSTALL_COMMAND}",
            ]
        )
    return "\n".join(lines)


def _require_modules(*modules: str) -> None:
    missing = [module for module in modules if not _module_available(module)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing quality-eval dependencies: {joined}. "
            f"Install them with `{QUALITY_EVAL_INSTALL_COMMAND}`."
        )


def _load_causal_lm(model_path: str):
    _require_modules("torch", "transformers")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def compare_generations(
    *,
    base_model: str,
    compressed_model: str,
    prompts: tuple[str, ...],
    max_new_tokens: int,
) -> list[dict[str, str]]:
    base, base_tokenizer = _load_causal_lm(base_model)
    compressed, compressed_tokenizer = _load_causal_lm(compressed_model)
    rows = []
    for prompt in prompts:
        rows.append(
            {
                "prompt": prompt,
                "base_response": _generate(base, base_tokenizer, prompt, max_new_tokens),
                "compressed_response": _generate(
                    compressed,
                    compressed_tokenizer,
                    prompt,
                    max_new_tokens,
                ),
            }
        )
    return rows


def calculate_perplexity(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_tokens: int,
    stride: int,
) -> float:
    _require_modules("torch")
    import math

    import torch

    encodings = tokenizer(
        "\n\n".join(texts),
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


def compare_perplexity(
    *,
    base_model: str,
    compressed_model: str,
    dataset: str,
    dataset_config_name: str,
    dataset_split: str,
    max_tokens: int,
    stride: int,
) -> dict[str, float]:
    _require_modules("datasets")
    from datasets import load_dataset

    data = load_dataset(dataset, dataset_config_name, split=dataset_split)
    texts = list(data["text"])
    base, base_tokenizer = _load_causal_lm(base_model)
    compressed, compressed_tokenizer = _load_causal_lm(compressed_model)
    base_ppl = calculate_perplexity(
        base, base_tokenizer, texts, max_tokens=max_tokens, stride=stride
    )
    compressed_ppl = calculate_perplexity(
        compressed,
        compressed_tokenizer,
        texts,
        max_tokens=max_tokens,
        stride=stride,
    )
    return {
        "base_perplexity": base_ppl,
        "compressed_perplexity": compressed_ppl,
        "delta": compressed_ppl - base_ppl,
        "relative_delta_pct": (compressed_ppl / base_ppl - 1) * 100,
    }


def build_long_context_prompt(target_tokens: int, tokenizer_name: str) -> str:
    _require_modules("transformers")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    anchor = "ANCHOR_FACT: compressed models must preserve long-context retrieval."
    filler = "The deployment report discusses memory, quantization, latency, and validation. "
    prompt = anchor + "\n\n"
    while len(tokenizer(prompt + filler, add_special_tokens=False)["input_ids"]) < target_tokens:
        prompt += filler
    prompt += "\n\nQuestion: Repeat the exact ANCHOR_FACT from the beginning."
    return prompt


def compare_long_context(
    *,
    base_model: str,
    compressed_model: str,
    long_context_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = build_long_context_prompt(long_context_tokens, base_model)
    rows = compare_generations(
        base_model=base_model,
        compressed_model=compressed_model,
        prompts=(prompt,),
        max_new_tokens=max_new_tokens,
    )
    anchor = "compressed models must preserve long-context retrieval"
    row = rows[0]
    return {
        "target_tokens": long_context_tokens,
        "base_contains_anchor": anchor in row["base_response"],
        "compressed_contains_anchor": anchor in row["compressed_response"],
        "base_response": row["base_response"],
        "compressed_response": row["compressed_response"],
    }


def build_lm_eval_command(*, model: str, task: str, limit: int | None = None) -> list[str]:
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model}",
        "--tasks",
        task,
        "--batch_size",
        "auto",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def run_lm_eval_pair(
    *,
    base_model: str,
    compressed_model: str,
    task: str,
    limit: int | None,
) -> dict[str, Any]:
    _require_modules("lm_eval")
    results = {}
    for label, model in (("base", base_model), ("compressed", compressed_model)):
        command = build_lm_eval_command(model=model, task=task, limit=limit)
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        results[label] = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return results


def run_quality_eval(
    *,
    plan: QualityEvalPlan,
    max_new_tokens: int,
    max_tokens: int,
    stride: int,
    lm_eval_limit: int | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {"plan": plan.to_dict()}
    if "generation comparison" in plan.checks:
        results["generation"] = compare_generations(
            base_model=plan.base_model,
            compressed_model=plan.compressed_model,
            prompts=plan.prompts,
            max_new_tokens=max_new_tokens,
        )
    if "perplexity comparison" in plan.checks:
        results["perplexity"] = compare_perplexity(
            base_model=plan.base_model,
            compressed_model=plan.compressed_model,
            dataset=plan.dataset,
            dataset_config_name=plan.dataset_config_name,
            dataset_split=plan.dataset_split,
            max_tokens=max_tokens,
            stride=stride,
        )
    if "long-context anchor probe" in plan.checks:
        results["long_context"] = compare_long_context(
            base_model=plan.base_model,
            compressed_model=plan.compressed_model,
            long_context_tokens=plan.long_context_tokens,
            max_new_tokens=max_new_tokens,
        )
    if "task metrics via lm_eval" in plan.checks:
        if not plan.lm_eval_task:
            raise ValueError("lm_eval task is required for task metrics")
        results["lm_eval"] = run_lm_eval_pair(
            base_model=plan.base_model,
            compressed_model=plan.compressed_model,
            task=plan.lm_eval_task,
            limit=lm_eval_limit,
        )
    if plan.output_json:
        output_path = Path(plan.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return results

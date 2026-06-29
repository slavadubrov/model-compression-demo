"""Quality evaluation helpers for compressed language models.

The public planning functions are dependency-free. Heavy ML imports happen only
inside execution functions so the CLI and tests stay usable on a normal Python
installation.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = (
    "Machine learning is a branch of",
    "Explain model quantization to an infrastructure engineer in two sentences.",
    "Summarize why KV cache memory matters for long-context LLM serving.",
)

QUALITY_EVAL_INSTALL_COMMAND = "uv sync --group quality"
DEFAULT_LM_EVAL_TASK = "hellaswag"
DEFAULT_LM_EVAL_LIMIT = 50
DEFAULT_MAX_PERPLEXITY_DELTA_PCT = 5.0
DEFAULT_MAX_TASK_REGRESSION = 0.02
QUALITY_EVAL_MODES = frozenset({"all", "generation", "perplexity", "long-context", "lm-eval"})


class QualityGateError(RuntimeError):
    """Raised when a quality evaluation proves a deployment gate failed."""

    def __init__(self, message: str, results: dict[str, Any]) -> None:
        super().__init__(message)
        self.results = results


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
    lm_eval_limit: int | None
    long_context_tokens: int
    max_perplexity_delta_pct: float | None
    max_task_regression: float | None
    require_long_context_anchor: bool
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
            "lm_eval_limit": self.lm_eval_limit,
            "long_context_tokens": self.long_context_tokens,
            "max_perplexity_delta_pct": self.max_perplexity_delta_pct,
            "max_task_regression": self.max_task_regression,
            "require_long_context_anchor": self.require_long_context_anchor,
            "required_modules": list(self.required_modules),
            "output_json": self.output_json,
        }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _local_model_uses_compressed_tensors(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    quantization_config = config.get("quantization_config")
    return (
        isinstance(quantization_config, dict)
        and quantization_config.get("quant_method") == "compressed-tensors"
    )


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")


def _validate_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")


def validate_quality_runtime_args(*, max_new_tokens: int, max_tokens: int, stride: int) -> None:
    """Validate runtime-only quality evaluation numeric settings."""

    _validate_positive_int("max_new_tokens", max_new_tokens)
    _validate_positive_int("max_tokens", max_tokens)
    _validate_positive_int("stride", stride)


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


def _required_modules(checks: tuple[str, ...], compressed_model: str | None) -> tuple[str, ...]:
    modules = {"accelerate", "torch", "transformers"}
    if compressed_model and _local_model_uses_compressed_tensors(compressed_model):
        modules.add("compressed_tensors")
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
    lm_eval_limit: int | None = DEFAULT_LM_EVAL_LIMIT,
    long_context_tokens: int = 4096,
    max_perplexity_delta_pct: float | None = DEFAULT_MAX_PERPLEXITY_DELTA_PCT,
    max_task_regression: float | None = DEFAULT_MAX_TASK_REGRESSION,
    require_long_context_anchor: bool = True,
    output_json: str | None = None,
) -> QualityEvalPlan:
    if mode not in QUALITY_EVAL_MODES:
        expected = ", ".join(sorted(QUALITY_EVAL_MODES))
        raise ValueError(f"mode must be one of: {expected}")
    _validate_non_negative_int("long_context_tokens", long_context_tokens)
    if lm_eval_limit is not None:
        _validate_positive_int("lm_eval_limit", lm_eval_limit)
    if max_perplexity_delta_pct is not None:
        _validate_non_negative_float("max_perplexity_delta_pct", max_perplexity_delta_pct)
    if max_task_regression is not None:
        _validate_non_negative_float("max_task_regression", max_task_regression)

    if mode in {"all", "lm-eval"} and lm_eval_task is None:
        lm_eval_task = DEFAULT_LM_EVAL_TASK
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
        lm_eval_limit=lm_eval_limit,
        long_context_tokens=long_context_tokens,
        max_perplexity_delta_pct=max_perplexity_delta_pct,
        max_task_regression=max_task_regression,
        require_long_context_anchor=require_long_context_anchor,
        required_modules=_required_modules(checks, compressed_model),
        output_json=output_json,
    )


def format_quality_eval_plan(plan: QualityEvalPlan) -> str:
    missing = [module for module in plan.required_modules if not _module_available(module)]
    ppl_delta = (
        f"{plan.max_perplexity_delta_pct}%"
        if plan.max_perplexity_delta_pct is not None
        else "disabled"
    )
    task_drop = (
        str(plan.max_task_regression) if plan.max_task_regression is not None else "disabled"
    )
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
            f"Max PPL delta:    {ppl_delta}",
            f"Max task drop:    {task_drop}",
            f"Require anchor:   {plan.require_long_context_anchor}",
            "Required modules:",
            *[f"  - {module}" for module in plan.required_modules],
        ]
    )
    if plan.lm_eval_task:
        lines.append(f"lm_eval task:     {plan.lm_eval_task}")
        lines.append(f"lm_eval limit:    {plan.lm_eval_limit or '(no limit)'}")
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
    modules = ["accelerate", "torch", "transformers"]
    if _local_model_uses_compressed_tensors(model_path):
        modules.append("compressed_tensors")
    _require_modules(*modules)
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


def _clear_model_cache() -> None:
    gc.collect()
    if _module_available("torch"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    sequential_models: bool = True,
) -> list[dict[str, str]]:
    if sequential_models:
        base, base_tokenizer = _load_causal_lm(base_model)
        base_responses = [
            _generate(base, base_tokenizer, prompt, max_new_tokens) for prompt in prompts
        ]
        del base, base_tokenizer
        _clear_model_cache()

        compressed, compressed_tokenizer = _load_causal_lm(compressed_model)
        compressed_responses = [
            _generate(compressed, compressed_tokenizer, prompt, max_new_tokens)
            for prompt in prompts
        ]
        del compressed, compressed_tokenizer
        _clear_model_cache()
    else:
        base, base_tokenizer = _load_causal_lm(base_model)
        compressed, compressed_tokenizer = _load_causal_lm(compressed_model)
        base_responses = [
            _generate(base, base_tokenizer, prompt, max_new_tokens) for prompt in prompts
        ]
        compressed_responses = [
            _generate(compressed, compressed_tokenizer, prompt, max_new_tokens)
            for prompt in prompts
        ]

    return [
        {
            "prompt": prompt,
            "base_response": base_response,
            "compressed_response": compressed_response,
        }
        for prompt, base_response, compressed_response in zip(
            prompts, base_responses, compressed_responses, strict=True
        )
    ]


def calculate_perplexity(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_tokens: int,
    stride: int,
) -> float:
    _validate_positive_int("max_tokens", max_tokens)
    _validate_positive_int("stride", stride)
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
    sequential_models: bool = True,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    _require_modules("datasets")
    from datasets import load_dataset

    data = load_dataset(dataset, dataset_config_name, split=dataset_split)
    texts = list(data["text"])
    base, base_tokenizer = _load_causal_lm(base_model)
    base_ppl = calculate_perplexity(
        base, base_tokenizer, texts, max_tokens=max_tokens, stride=stride
    )
    if checkpoint:
        checkpoint({"perplexity": {"base_perplexity": base_ppl, "status": "base_complete"}})
    if sequential_models:
        del base, base_tokenizer
        _clear_model_cache()

    compressed, compressed_tokenizer = _load_causal_lm(compressed_model)
    compressed_ppl = calculate_perplexity(
        compressed,
        compressed_tokenizer,
        texts,
        max_tokens=max_tokens,
        stride=stride,
    )
    if sequential_models:
        del compressed, compressed_tokenizer
        _clear_model_cache()
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
    sequential_models: bool = True,
) -> dict[str, Any]:
    prompt = build_long_context_prompt(long_context_tokens, base_model)
    rows = compare_generations(
        base_model=base_model,
        compressed_model=compressed_model,
        prompts=(prompt,),
        max_new_tokens=max_new_tokens,
        sequential_models=sequential_models,
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


def _write_results_if_requested(results: dict[str, Any], output_json: str | None) -> None:
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_primary_lm_eval_score(stdout: str) -> float | None:
    payload = _extract_json_object(stdout)
    if not payload:
        return None
    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        return None
    first_task = next(iter(results.values()))
    if not isinstance(first_task, dict):
        return None
    preferred = (
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "f1,none",
        "acc_norm",
        "acc",
        "exact_match",
        "f1",
    )
    for key in preferred:
        value = first_task.get(key)
        if isinstance(value, int | float):
            return float(value)
    for key, value in first_task.items():
        if re.search(r"(acc|exact|f1)", key) and isinstance(value, int | float):
            return float(value)
    return None


def summarize_quality_results(
    results: dict[str, Any],
    *,
    max_perplexity_delta_pct: float | None,
    max_task_regression: float | None,
    require_long_context_anchor: bool,
) -> dict[str, Any]:
    """Return a compact pass/fail/needs_review verdict for quality results."""

    failures: list[str] = []
    warnings: list[str] = []

    perplexity = results.get("perplexity")
    if isinstance(perplexity, dict) and max_perplexity_delta_pct is not None:
        delta_pct = perplexity.get("relative_delta_pct")
        if isinstance(delta_pct, int | float) and delta_pct > max_perplexity_delta_pct:
            failures.append(
                f"perplexity regression {delta_pct:.2f}% exceeds {max_perplexity_delta_pct:.2f}%"
            )

    long_context = results.get("long_context")
    if isinstance(long_context, dict):
        base_anchor = bool(long_context.get("base_contains_anchor"))
        compressed_anchor = bool(long_context.get("compressed_contains_anchor"))
        if require_long_context_anchor and not compressed_anchor:
            failures.append("compressed model missed the long-context anchor")
        if not base_anchor:
            warnings.append("base model missed the long-context anchor; review the probe")

    lm_eval = results.get("lm_eval")
    if isinstance(lm_eval, dict):
        for label in ("base", "compressed"):
            run = lm_eval.get(label)
            if isinstance(run, dict) and run.get("returncode") != 0:
                failures.append(f"lm_eval {label} run exited with {run.get('returncode')}")

        if max_task_regression is not None:
            base_run = lm_eval.get("base")
            compressed_run = lm_eval.get("compressed")
            if isinstance(base_run, dict) and isinstance(compressed_run, dict):
                base_score = _extract_primary_lm_eval_score(str(base_run.get("stdout", "")))
                compressed_score = _extract_primary_lm_eval_score(
                    str(compressed_run.get("stdout", ""))
                )
                if base_score is not None and compressed_score is not None:
                    regression = base_score - compressed_score
                    if regression > max_task_regression:
                        failures.append(
                            f"task metric regression {regression:.4f} exceeds "
                            f"{max_task_regression:.4f}"
                        )
                elif not failures:
                    warnings.append("could not parse lm_eval scores; review task metrics manually")

    verdict = "fail" if failures else "needs_review" if warnings else "pass"
    return {"verdict": verdict, "failures": failures, "warnings": warnings}


def run_quality_eval(
    *,
    plan: QualityEvalPlan,
    max_new_tokens: int,
    max_tokens: int,
    stride: int,
    lm_eval_limit: int | None = None,
    sequential_models: bool = True,
) -> dict[str, Any]:
    validate_quality_runtime_args(
        max_new_tokens=max_new_tokens,
        max_tokens=max_tokens,
        stride=stride,
    )
    results: dict[str, Any] = {"plan": plan.to_dict()}
    output_json = plan.output_json

    def checkpoint(partial: dict[str, Any]) -> None:
        results.update(partial)
        _write_results_if_requested(results, output_json)

    if "generation comparison" in plan.checks:
        results["generation"] = compare_generations(
            base_model=plan.base_model,
            compressed_model=plan.compressed_model,
            prompts=plan.prompts,
            max_new_tokens=max_new_tokens,
            sequential_models=sequential_models,
        )
        _write_results_if_requested(results, output_json)
    if "perplexity comparison" in plan.checks:
        results["perplexity"] = compare_perplexity(
            base_model=plan.base_model,
            compressed_model=plan.compressed_model,
            dataset=plan.dataset,
            dataset_config_name=plan.dataset_config_name,
            dataset_split=plan.dataset_split,
            max_tokens=max_tokens,
            stride=stride,
            sequential_models=sequential_models,
            checkpoint=checkpoint,
        )
        _write_results_if_requested(results, output_json)
    if "long-context anchor probe" in plan.checks:
        results["long_context"] = compare_long_context(
            base_model=plan.base_model,
            compressed_model=plan.compressed_model,
            long_context_tokens=plan.long_context_tokens,
            max_new_tokens=max_new_tokens,
            sequential_models=sequential_models,
        )
        _write_results_if_requested(results, output_json)
    if "task metrics via lm_eval" in plan.checks:
        if not plan.lm_eval_task:
            raise ValueError("lm_eval task is required for task metrics")
        results["lm_eval"] = run_lm_eval_pair(
            base_model=plan.base_model,
            compressed_model=plan.compressed_model,
            task=plan.lm_eval_task,
            limit=lm_eval_limit if lm_eval_limit is not None else plan.lm_eval_limit,
        )
        _write_results_if_requested(results, output_json)

    results["summary"] = summarize_quality_results(
        results,
        max_perplexity_delta_pct=plan.max_perplexity_delta_pct,
        max_task_regression=plan.max_task_regression,
        require_long_context_anchor=plan.require_long_context_anchor,
    )
    _write_results_if_requested(results, output_json)
    if results["summary"]["verdict"] == "fail":
        raise QualityGateError("quality evaluation failed deployment gates", results)
    return results

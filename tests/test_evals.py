from __future__ import annotations

import pytest

from compression_demo.evals import (
    DEFAULT_PROMPTS,
    build_lm_eval_command,
    build_quality_eval_plan,
    format_quality_eval_plan,
    summarize_quality_results,
    validate_quality_runtime_args,
)


def test_quality_plan_covers_article_eval_checks() -> None:
    plan = build_quality_eval_plan(
        base_model="base",
        compressed_model="compressed",
        long_context_tokens=8192,
    )

    assert "generation comparison" in plan.checks
    assert "perplexity comparison" in plan.checks
    assert "task metrics via lm_eval" in plan.checks
    assert "long-context anchor probe" in plan.checks
    assert "accelerate" in plan.required_modules
    assert "datasets" in plan.required_modules
    assert "lm_eval" in plan.required_modules
    assert plan.lm_eval_task == "hellaswag"
    assert plan.lm_eval_limit == 50


def test_single_mode_quality_plan() -> None:
    plan = build_quality_eval_plan(
        base_model="base",
        compressed_model="compressed",
        mode="generation",
    )

    assert plan.checks == ("generation comparison",)
    assert plan.prompts == DEFAULT_PROMPTS


def test_format_quality_eval_plan_lists_missing_install_command(monkeypatch) -> None:
    monkeypatch.setattr("compression_demo.evals._module_available", lambda name: False)
    plan = build_quality_eval_plan(
        base_model="base",
        compressed_model="compressed",
        mode="perplexity",
    )

    formatted = format_quality_eval_plan(plan)

    assert "Quality evaluation plan" in formatted
    assert "perplexity comparison" in formatted
    assert "uv sync --group quality" in formatted


def test_quality_plan_requires_compressed_tensors_for_local_quantized_model(tmp_path) -> None:
    compressed = tmp_path / "compressed"
    compressed.mkdir()
    (compressed / "config.json").write_text(
        '{"quantization_config": {"quant_method": "compressed-tensors"}}',
        encoding="utf-8",
    )

    plan = build_quality_eval_plan(
        base_model="base",
        compressed_model=str(compressed),
        mode="generation",
    )

    assert "compressed_tensors" in plan.required_modules


def test_lm_eval_command_uses_hf_model() -> None:
    command = build_lm_eval_command(model="compressed", task="hellaswag", limit=10)

    assert command[:4] == ["lm_eval", "--model", "hf", "--model_args"]
    assert "pretrained=compressed" in command
    assert "hellaswag" in command
    assert "10" in command


def test_quality_summary_fails_on_perplexity_regression() -> None:
    results = {"perplexity": {"relative_delta_pct": 6.0}}

    summary = summarize_quality_results(
        results,
        max_perplexity_delta_pct=5.0,
        max_task_regression=0.02,
        require_long_context_anchor=True,
    )

    assert summary["verdict"] == "fail"
    assert "perplexity regression" in summary["failures"][0]


def test_quality_summary_fails_on_lm_eval_nonzero() -> None:
    results = {
        "lm_eval": {
            "base": {"returncode": 0, "stdout": ""},
            "compressed": {"returncode": 1, "stdout": ""},
        }
    }

    summary = summarize_quality_results(
        results,
        max_perplexity_delta_pct=5.0,
        max_task_regression=0.02,
        require_long_context_anchor=True,
    )

    assert summary["verdict"] == "fail"
    assert "lm_eval compressed run exited with 1" in summary["failures"]


def test_quality_summary_passes_when_gates_hold() -> None:
    stdout = '{"results": {"hellaswag": {"acc,none": 0.8}}}'
    results = {
        "perplexity": {"relative_delta_pct": 1.5},
        "long_context": {
            "base_contains_anchor": True,
            "compressed_contains_anchor": True,
        },
        "lm_eval": {
            "base": {"returncode": 0, "stdout": stdout},
            "compressed": {"returncode": 0, "stdout": stdout},
        },
    }

    summary = summarize_quality_results(
        results,
        max_perplexity_delta_pct=5.0,
        max_task_regression=0.02,
        require_long_context_anchor=True,
    )

    assert summary == {"verdict": "pass", "failures": [], "warnings": []}


def test_quality_plan_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        build_quality_eval_plan(
            base_model="base",
            compressed_model="compressed",
            mode="surprise",
        )


def test_quality_runtime_args_must_be_positive() -> None:
    with pytest.raises(ValueError, match="stride must be a positive integer"):
        validate_quality_runtime_args(max_new_tokens=8, max_tokens=128, stride=0)

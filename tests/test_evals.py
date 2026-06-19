from __future__ import annotations

from compression_demo.evals import (
    DEFAULT_PROMPTS,
    build_lm_eval_command,
    build_quality_eval_plan,
    format_quality_eval_plan,
)


def test_quality_plan_covers_article_eval_checks() -> None:
    plan = build_quality_eval_plan(
        base_model="base",
        compressed_model="compressed",
        lm_eval_task="hellaswag",
        long_context_tokens=8192,
    )

    assert "generation comparison" in plan.checks
    assert "perplexity comparison" in plan.checks
    assert "task metrics via lm_eval" in plan.checks
    assert "long-context anchor probe" in plan.checks
    assert "datasets" in plan.required_modules
    assert "lm_eval" in plan.required_modules


def test_single_mode_quality_plan() -> None:
    plan = build_quality_eval_plan(
        base_model="base",
        compressed_model="compressed",
        mode="generation",
    )

    assert plan.checks == ("generation comparison",)
    assert plan.prompts == DEFAULT_PROMPTS


def test_format_quality_eval_plan_lists_missing_install_command() -> None:
    plan = build_quality_eval_plan(
        base_model="base",
        compressed_model="compressed",
        mode="perplexity",
    )

    formatted = format_quality_eval_plan(plan)

    assert "Quality evaluation plan" in formatted
    assert "perplexity comparison" in formatted
    assert "uv pip install torch transformers datasets lm_eval" in formatted


def test_lm_eval_command_uses_hf_model() -> None:
    command = build_lm_eval_command(model="compressed", task="hellaswag", limit=10)

    assert command[:4] == ["lm_eval", "--model", "hf", "--model_args"]
    assert "pretrained=compressed" in command
    assert "hellaswag" in command
    assert "10" in command

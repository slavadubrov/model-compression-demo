from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from compression_demo.evals import (
    DEFAULT_PROMPTS,
    build_lm_eval_command,
    build_quality_eval_plan,
    format_quality_eval_plan,
)


class EvalPlanTests(unittest.TestCase):
    def test_quality_plan_covers_article_eval_checks(self) -> None:
        plan = build_quality_eval_plan(
            base_model="base",
            compressed_model="compressed",
            lm_eval_task="hellaswag",
            long_context_tokens=8192,
        )

        self.assertIn("generation comparison", plan.checks)
        self.assertIn("perplexity comparison", plan.checks)
        self.assertIn("task metrics via lm_eval", plan.checks)
        self.assertIn("long-context anchor probe", plan.checks)
        self.assertIn("datasets", plan.required_modules)
        self.assertIn("lm_eval", plan.required_modules)

    def test_single_mode_quality_plan(self) -> None:
        plan = build_quality_eval_plan(
            base_model="base",
            compressed_model="compressed",
            mode="generation",
        )

        self.assertEqual(plan.checks, ("generation comparison",))
        self.assertEqual(plan.prompts, DEFAULT_PROMPTS)

    def test_format_quality_eval_plan_lists_missing_install_command(self) -> None:
        plan = build_quality_eval_plan(
            base_model="base",
            compressed_model="compressed",
            mode="perplexity",
        )

        formatted = format_quality_eval_plan(plan)

        self.assertIn("Quality evaluation plan", formatted)
        self.assertIn("perplexity comparison", formatted)
        self.assertIn("uv sync --extra gpu --extra alternatives", formatted)

    def test_lm_eval_command_uses_hf_model(self) -> None:
        command = build_lm_eval_command(model="compressed", task="hellaswag", limit=10)

        self.assertEqual(command[:4], ["lm_eval", "--model", "hf", "--model_args"])
        self.assertIn("pretrained=compressed", command)
        self.assertIn("hellaswag", command)
        self.assertIn("10", command)


if __name__ == "__main__":
    unittest.main()

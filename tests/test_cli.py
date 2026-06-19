from __future__ import annotations

import io
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from compression_demo.cli import main


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(args))
        self.assertEqual(code, 0)
        return buf.getvalue()

    def test_recipe_contains_llmcompressor_gptq(self) -> None:
        out = self.run_cli("recipe", "--algorithm", "gptq-w4a16")
        self.assertIn("GPTQModifier", out)
        self.assertIn("oneshot", out)

    def test_estimate_json(self) -> None:
        out = self.run_cli(
            "estimate",
            "--params-b",
            "7",
            "--scheme",
            "w4a16",
            "--json",
        )
        self.assertIn('"total_gib"', out)
        self.assertIn('"scheme_key": "w4a16"', out)

    def test_quantize_dry_run(self) -> None:
        out = self.run_cli("quantize", "--dry-run")
        self.assertIn("Quantization dry run", out)
        self.assertIn("Qwen/Qwen3-0.6B", out)

    def test_quality_eval_dry_run(self) -> None:
        out = self.run_cli(
            "quality-eval",
            "--base-model",
            "base",
            "--compressed-model",
            "compressed",
            "--lm-eval-task",
            "hellaswag",
            "--dry-run",
        )
        self.assertIn("generation comparison", out)
        self.assertIn("perplexity comparison", out)
        self.assertIn("task metrics via lm_eval", out)
        self.assertIn("long-context anchor probe", out)

    def test_compare_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base"
            quant = root / "quant"
            base.mkdir()
            quant.mkdir()
            (base / "weights.bin").write_bytes(b"0" * 100)
            (quant / "weights.bin").write_bytes(b"0" * 25)
            out = self.run_cli(
                "compare-size", "--base-dir", str(base), "--compressed-dir", str(quant)
            )
        self.assertIn("Reduction:  75.0%", out)

    def test_html_smoke(self) -> None:
        guide = pathlib.Path(__file__).resolve().parents[1] / "index.html"
        out = self.run_cli("smoke-html", "--path", str(guide))
        self.assertIn("HTML guide OK", out)


if __name__ == "__main__":
    unittest.main()

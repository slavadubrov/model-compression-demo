from __future__ import annotations

import io
import pathlib
from contextlib import redirect_stdout

from compression_demo.cli import main


def run_cli(*args: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(list(args))
    assert code == 0
    return buf.getvalue()


def test_recipe_contains_llmcompressor_gptq() -> None:
    out = run_cli("recipe", "--algorithm", "gptq-w4a16")
    assert "GPTQModifier" in out
    assert "oneshot" in out


def test_estimate_json() -> None:
    out = run_cli(
        "estimate",
        "--params-b",
        "7",
        "--scheme",
        "w4a16",
        "--json",
    )
    assert '"total_gib"' in out
    assert '"scheme_key": "w4a16"' in out


def test_quantize_dry_run() -> None:
    out = run_cli("quantize", "--dry-run")
    assert "Quantization dry run" in out
    assert "Qwen/Qwen3-0.6B" in out


def test_quality_eval_dry_run() -> None:
    out = run_cli(
        "quality-eval",
        "--base-model",
        "base",
        "--compressed-model",
        "compressed",
        "--lm-eval-task",
        "hellaswag",
        "--dry-run",
    )
    assert "generation comparison" in out
    assert "perplexity comparison" in out
    assert "task metrics via lm_eval" in out
    assert "long-context anchor probe" in out


def test_compare_size(tmp_path: pathlib.Path) -> None:
    base = tmp_path / "base"
    quant = tmp_path / "quant"
    base.mkdir()
    quant.mkdir()
    (base / "weights.bin").write_bytes(b"0" * 100)
    (quant / "weights.bin").write_bytes(b"0" * 25)
    out = run_cli("compare-size", "--base-dir", str(base), "--compressed-dir", str(quant))
    assert "Reduction:  75.0%" in out


def test_html_smoke() -> None:
    guide = pathlib.Path(__file__).resolve().parents[1] / "index.html"
    out = run_cli("smoke-html", "--path", str(guide))
    assert "HTML guide OK" in out

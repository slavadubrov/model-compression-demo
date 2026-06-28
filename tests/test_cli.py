from __future__ import annotations

import io
import json
import pathlib
from contextlib import redirect_stdout

import pytest

from compression_demo.cli import main


def run_cli(*args: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(list(args))
    assert code == 0
    return buf.getvalue()


def assert_cli_usage_error(*args: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(list(args))
    assert exc_info.value.code == 2


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
    assert "outputs/Qwen3-0.6B-W4A16" in out
    assert "Exact command:" in out


def test_quantize_dry_run_uses_algorithm_specific_default_outputs() -> None:
    cases = {
        "gptq-w4a16": "outputs/Qwen3-0.6B-W4A16",
        "rtn-w8a16": "outputs/Qwen3-0.6B-W8A16",
        "fp8-dynamic": "outputs/Qwen3-0.6B-FP8-Dynamic",
    }
    for algorithm, expected_output in cases.items():
        out = run_cli("quantize", "--algorithm", algorithm, "--dry-run")
        assert f"Output:    {expected_output}" in out


def test_quantize_dry_run_describes_representative_calibration_file() -> None:
    out = run_cli(
        "quantize",
        "--calibration-file",
        "examples/representative_calibration.jsonl",
        "--dry-run",
    )
    assert "representative local file examples/representative_calibration.jsonl" in out
    assert "--calibration-file examples/representative_calibration.jsonl" in out


def test_quality_eval_dry_run() -> None:
    out = run_cli(
        "quality-eval",
        "--base-model",
        "base",
        "--compressed-model",
        "compressed",
        "--dry-run",
    )
    assert "generation comparison" in out
    assert "perplexity comparison" in out
    assert "task metrics via lm_eval" in out
    assert "long-context anchor probe" in out
    assert "lm_eval task:     hellaswag" in out
    assert "Max PPL delta:    5.0%" in out


def test_plan_for_cpu_reports_local_runtime_not_gpu() -> None:
    out = run_cli(
        "plan",
        "--params-b",
        "7",
        "--goal",
        "fit-memory",
        "--hardware",
        "cpu",
    )
    assert "Algorithm:         GGUF Q4/K-quants" in out
    assert "RAM / unified memory target" in out
    assert "Recommended local runtimes:" in out
    assert "Recommended GPUs:" not in out
    assert "Compression GPU:   not required" in out


def test_serve_command_for_fp8_kv_cache() -> None:
    out = run_cli(
        "serve-command",
        "--algorithm",
        "fp8-dynamic",
        "--fp8-kv-cache",
        "--enable-prefix-caching",
    )
    assert "--quantization fp8" in out
    assert "--kv-cache-dtype fp8" in out
    assert "--max-model-len 32768" in out
    assert "--enable-prefix-caching" in out


def test_benchmark_plan_outputs_json_shape_and_commands(tmp_path: pathlib.Path) -> None:
    output_json = tmp_path / "benchmark-plan.json"
    out = run_cli(
        "benchmark-plan",
        "--model",
        "Qwen/Qwen2.5-32B-Instruct",
        "--algorithms",
        "gptq-w4a16,awq-w4a16,bnb-nf4,gguf-q4",
        "--dataset-name",
        "sharegpt",
        "--num-prompts",
        "200",
        "--input-len",
        "1024",
        "--output-len",
        "256",
        "--output-json",
        str(output_json),
    )
    payload = json.loads(out)
    file_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload == file_payload
    assert payload["warning"].startswith("This command plan generates")
    assert len(payload["rows"]) == 4
    first = payload["rows"][0]
    assert first["algorithm_key"] == "gptq-w4a16"
    assert "vllm serve" in first["serve_command"]
    assert "vllm bench serve" in first["bench_command"]
    assert "quality-eval" in first["quality_eval_command"]


def test_benchmark_plan_generates_vllm_flags_for_common_variants() -> None:
    out = run_cli(
        "benchmark-plan",
        "--algorithms",
        "awq-w4a16,bnb-nf4,gguf-q4",
    )
    assert "--quantization awq" in out
    assert "--quantization bitsandbytes --load-format bitsandbytes" in out
    assert "--tokenizer Qwen/Qwen2.5-32B-Instruct" in out
    assert "GPU benchmark numbers are environment-specific" in out


def test_readme_mentions_benchmark_plan_workflow() -> None:
    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "benchmark-plan" in text
    assert "vllm_quantization_benchmark.sh" in text


def test_model_preset_supplies_params_and_architecture() -> None:
    out = run_cli("estimate", "--model-preset", "llama3-8b", "--scheme", "w4a16")
    assert "Architecture:     Llama 3/3.1 8B" in out
    assert "Layers/hidden/KV: 32 / 4096 / 0.250" in out


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


def test_quality_eval_rejects_invalid_runtime_numbers() -> None:
    assert_cli_usage_error(
        "quality-eval",
        "--base-model",
        "base",
        "--compressed-model",
        "compressed",
        "--stride",
        "0",
        "--dry-run",
    )


def test_gpu_benchmark_rejects_invalid_repeat_runs() -> None:
    assert_cli_usage_error("gpu-benchmark", "--repeat-runs", "0", "--dry-run")

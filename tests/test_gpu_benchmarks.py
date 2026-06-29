from __future__ import annotations

import json

import pytest

from compression_demo.gpu_benchmarks import (
    GPUBenchmarkRun,
    _parse_nvidia_smi_used_memory_gib,
    _parse_vllm_model_memory_from_lines,
    build_gpu_benchmark_plan,
    format_gpu_benchmark_plan,
    summarize_gpu_benchmark_results,
    validate_gpu_benchmark_numbers,
    write_gpu_benchmark_report,
)


def test_gpu_benchmark_plan_formats_without_heavy_imports() -> None:
    plan = build_gpu_benchmark_plan(
        models=("Qwen/Qwen3-8B",),
        variants=("bf16", "fp8-dynamic"),
        kernels=("sdpa", "eager"),
        prompts=("hello",),
        max_new_tokens=8,
        warmup_runs=1,
        repeat_runs=2,
        output_json="reports/out.json",
        report_html="reports/out.html",
    )

    formatted = format_gpu_benchmark_plan(plan)

    assert plan["total_runs"] == 2
    assert "Qwen/Qwen3-8B" in formatted
    assert "bf16, fp8-dynamic" in formatted
    assert "downloads models" in formatted


def test_gpu_benchmark_plan_routes_fp8_to_vllm_only() -> None:
    plan = build_gpu_benchmark_plan(
        models=("Qwen/Qwen2.5-7B-Instruct",),
        variants=("bf16", "fp8-dynamic", "fp8-dynamic-kv"),
        kernels=("sdpa", "vllm"),
        prompts=("hello",),
        max_new_tokens=8,
        warmup_runs=1,
        repeat_runs=1,
        output_json="reports/out.json",
        report_html="reports/out.html",
        vllm_max_model_len=2048,
        vllm_gpu_memory_utilization=0.8,
    )

    formatted = format_gpu_benchmark_plan(plan)

    assert plan["total_runs"] == 4
    assert len(plan["skipped_runs"]) == 2
    assert any(item["variant"] == "fp8-dynamic" for item in plan["skipped_runs"])
    assert "vLLM max len:2048" in formatted
    assert "Skipped:      2" in formatted


def test_gpu_benchmark_summary_picks_fastest_and_compressed() -> None:
    runs = [
        GPUBenchmarkRun(
            model="model-a",
            variant="bf16",
            variant_label="BF16 baseline",
            kernel="sdpa",
            kernel_label="PyTorch SDPA default",
            status="ok",
            generated_tokens_per_second=10,
            peak_allocated_gib=4,
            model_memory_gib=2,
            compression_ratio_vs_bf16=1,
        ),
        GPUBenchmarkRun(
            model="model-a",
            variant="fp8-dynamic",
            variant_label="FP8 dynamic",
            kernel="vllm",
            kernel_label="vLLM",
            status="ok",
            generated_tokens_per_second=12,
            peak_allocated_gib=2,
            model_memory_gib=0.6,
            compression_ratio_vs_bf16=3.3,
        ),
    ]

    summary = summarize_gpu_benchmark_results(runs)

    assert summary["ok_runs"] == 2
    assert summary["fastest"]["variant"] == "fp8-dynamic"
    assert summary["lowest_memory"]["variant"] == "fp8-dynamic"
    assert summary["lowest_memory"]["memory_basis_gib"] == 0.6
    assert summary["best_compression"]["compression_ratio_vs_bf16"] == 3.3


def test_gpu_benchmark_report_writes_html(tmp_path) -> None:
    payload = {
        "environment": {
            "created_at": "2026-06-28T00:00:00+00:00",
            "torch": {
                "device_name": "RTX 4090",
                "compute_capability": "8.9",
                "version": "test",
                "cuda_version": "test",
            },
        },
        "config": {"output_json": "out.json", "report_html": "out.html"},
        "summary": {
            "ok_runs": 2,
            "skipped_runs": 0,
            "failed_runs": 0,
            "fastest": {
                "model": "Qwen/Qwen3.5-9B",
                "variant": "fp8-dynamic",
                "kernel": "vllm",
                "generated_tokens_per_second": 18.0,
            },
            "lowest_memory": {
                "model": "Qwen/Qwen3.5-9B",
                "variant": "fp8-dynamic",
                "kernel": "vllm",
                "model_memory_gib": 8.9,
                "memory_basis_gib": 8.9,
            },
            "best_compression": {
                "model": "Qwen/Qwen3.5-9B",
                "variant": "fp8-dynamic",
                "kernel": "vllm",
                "compression_ratio_vs_bf16": 1.8,
            },
        },
        "runs": [
            {
                "model": "Qwen/Qwen3.5-9B",
                "variant": "bf16",
                "kernel": "vllm",
                "status": "ok",
                "generated_tokens_per_second": 10.0,
                "peak_allocated_gib": None,
                "peak_reserved_gib": None,
                "model_memory_gib": 16.0,
                "gpu_memory_delta_gib": 23.0,
                "compression_ratio_vs_bf16": 1.0,
                "error": None,
            },
            {
                "model": "Qwen/Qwen3.5-9B",
                "variant": "fp8-dynamic",
                "kernel": "vllm",
                "status": "ok",
                "generated_tokens_per_second": 18.0,
                "peak_allocated_gib": None,
                "peak_reserved_gib": None,
                "model_memory_gib": 8.9,
                "gpu_memory_delta_gib": 22.9,
                "compression_ratio_vs_bf16": 1.8,
                "error": None,
            },
        ],
    }
    report = tmp_path / "report.html"

    write_gpu_benchmark_report(payload, str(report))

    html = report.read_text(encoding="utf-8")
    assert "GPU Compression Benchmark Report" in html
    assert "Conclusions" in html
    assert "Best FP8 throughput comparison" in html
    assert "nvidia-smi used-memory delta" in html
    assert "Generation Throughput" in html
    assert "Model Memory Footprint" in html
    assert "Total GPU Memory Delta" in html
    assert "GPU Delta GiB" in html
    assert "Compression Ratio vs BF16 Memory Footprint" in html
    assert "No successful measurements." not in html
    assert "Qwen3.5-9B" in html


def test_cli_gpu_benchmark_can_be_monkeypatched(tmp_path, monkeypatch) -> None:
    from compression_demo import cli

    def fake_run_gpu_benchmarks(**kwargs):
        output_json = kwargs["output_json"]
        report_html = kwargs["report_html"]
        payload = {
            "environment": {},
            "config": {
                "output_json": output_json,
                "report_html": report_html,
            },
            "summary": {
                "ok_runs": 1,
                "skipped_runs": 0,
                "failed_runs": 0,
                "fastest": None,
                "lowest_memory": None,
                "best_compression": None,
            },
            "runs": [],
        }
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        with open(report_html, "w", encoding="utf-8") as handle:
            handle.write("<html></html>")
        return payload

    monkeypatch.setattr(cli, "run_gpu_benchmarks", fake_run_gpu_benchmarks)
    output_json = tmp_path / "bench.json"
    report_html = tmp_path / "bench.html"

    code = cli.main(
        [
            "gpu-benchmark",
            "--models",
            "model-a",
            "--variants",
            "bf16",
            "--kernels",
            "sdpa",
            "--output-json",
            str(output_json),
            "--report-html",
            str(report_html),
        ]
    )

    assert code == 0
    assert output_json.exists()
    assert report_html.exists()


def test_gpu_benchmark_numbers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="repeat_runs must be a positive integer"):
        validate_gpu_benchmark_numbers(max_new_tokens=8, warmup_runs=1, repeat_runs=0)


def test_nvidia_smi_memory_parser_reads_mib_values() -> None:
    assert _parse_nvidia_smi_used_memory_gib("1024\n") == 1.0
    assert _parse_nvidia_smi_used_memory_gib("2048 MiB\n") == 2.0
    assert _parse_nvidia_smi_used_memory_gib("not available\n") is None


def test_vllm_model_memory_parser_reads_current_log_formats() -> None:
    assert (
        _parse_vllm_model_memory_from_lines(
            ["INFO model_runner.py:320] Model loading took 1.23 GiB and 2.000000 seconds"]
        )
        == 1.23
    )
    assert (
        _parse_vllm_model_memory_from_lines(
            ["INFO gpu_model_runner.py:5188] Model loading took 4.56 GiB memory and 7s"]
        )
        == 4.56
    )
    assert _parse_vllm_model_memory_from_lines(["no memory line"]) is None

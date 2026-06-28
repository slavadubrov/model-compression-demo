from __future__ import annotations

import json

from compression_demo.gpu_benchmarks import (
    GPUBenchmarkRun,
    build_gpu_benchmark_plan,
    format_gpu_benchmark_plan,
    summarize_gpu_benchmark_results,
    write_gpu_benchmark_report,
)


def test_gpu_benchmark_plan_formats_without_heavy_imports() -> None:
    plan = build_gpu_benchmark_plan(
        models=("Qwen/Qwen3-0.6B",),
        variants=("bf16", "bnb-nf4"),
        kernels=("sdpa", "eager"),
        prompts=("hello",),
        max_new_tokens=8,
        warmup_runs=1,
        repeat_runs=2,
        output_json="reports/out.json",
        report_html="reports/out.html",
    )

    formatted = format_gpu_benchmark_plan(plan)

    assert plan["total_runs"] == 4
    assert "Qwen/Qwen3-0.6B" in formatted
    assert "bf16, bnb-nf4" in formatted
    assert "downloads models" in formatted


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
            variant="bnb-nf4",
            variant_label="bitsandbytes NF4 4-bit",
            kernel="sdpa",
            kernel_label="PyTorch SDPA default",
            status="ok",
            generated_tokens_per_second=12,
            peak_allocated_gib=2,
            model_memory_gib=0.6,
            compression_ratio_vs_bf16=3.3,
        ),
    ]

    summary = summarize_gpu_benchmark_results(runs)

    assert summary["ok_runs"] == 2
    assert summary["fastest"]["variant"] == "bnb-nf4"
    assert summary["lowest_memory"]["variant"] == "bnb-nf4"
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
        "summary": {},
        "runs": [
            {
                "model": "Qwen/Qwen3-0.6B",
                "variant": "bf16",
                "kernel": "sdpa",
                "status": "ok",
                "generated_tokens_per_second": 10.0,
                "peak_allocated_gib": 1.2,
                "model_memory_gib": 1.1,
                "compression_ratio_vs_bf16": 1.0,
                "error": None,
            }
        ],
    }
    report = tmp_path / "report.html"

    write_gpu_benchmark_report(payload, str(report))

    html = report.read_text(encoding="utf-8")
    assert "GPU Compression Benchmark Report" in html
    assert "Generation Throughput" in html
    assert "Qwen3-0.6B" in html


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

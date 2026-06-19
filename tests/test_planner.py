from __future__ import annotations

import pytest

from compression_demo.catalog import SCHEMES
from compression_demo.planner import (
    estimate_kv_cache_gib,
    estimate_serving_memory,
    recommend_instances,
    select_algorithm,
)


def test_w4a16_is_smaller_than_bf16() -> None:
    bf16 = estimate_serving_memory(
        params_b=7,
        scheme_key="bf16",
        layers=32,
        hidden_size=4096,
        context_tokens=2048,
        concurrency=1,
    )
    w4a16 = estimate_serving_memory(
        params_b=7,
        scheme_key="w4a16",
        layers=32,
        hidden_size=4096,
        context_tokens=2048,
        concurrency=1,
    )
    assert w4a16.weight_gib < bf16.weight_gib
    assert SCHEMES["w4a16"].effective_weight_bits < SCHEMES["bf16"].effective_weight_bits


def test_kv_cache_scales_with_concurrency() -> None:
    one = estimate_kv_cache_gib(
        layers=32,
        hidden_size=4096,
        context_tokens=4096,
        concurrency=1,
    )
    four = estimate_kv_cache_gib(
        layers=32,
        hidden_size=4096,
        context_tokens=4096,
        concurrency=4,
    )
    assert four == pytest.approx(one * 4)


def test_select_algorithm_prefers_edge_for_cpu() -> None:
    assert select_algorithm(goal="fit-memory", hardware="cpu") == "gguf-q4"


def test_select_algorithm_prefers_fp8_for_hopper_throughput() -> None:
    assert select_algorithm(goal="throughput", hardware="hopper") == "fp8-dynamic"


def test_recommendations_respect_compute_capability() -> None:
    recs = recommend_instances(required_gib=20, min_compute_capability=8.9)
    assert recs
    assert all(rec.instance.compute_capability >= 8.9 for rec in recs)
    assert any("L4" in rec.instance.name or "4090" in rec.instance.name for rec in recs)


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        estimate_serving_memory(
            params_b=0,
            scheme_key="w4a16",
            layers=32,
            hidden_size=4096,
            context_tokens=2048,
            concurrency=1,
        )

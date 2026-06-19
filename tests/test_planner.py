from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from compression_demo.catalog import SCHEMES
from compression_demo.planner import (
    estimate_kv_cache_gib,
    estimate_serving_memory,
    recommend_instances,
    select_algorithm,
)


class PlannerTests(unittest.TestCase):
    def test_w4a16_is_smaller_than_bf16(self) -> None:
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
        self.assertLess(w4a16.weight_gib, bf16.weight_gib)
        self.assertLess(
            SCHEMES["w4a16"].effective_weight_bits, SCHEMES["bf16"].effective_weight_bits
        )

    def test_kv_cache_scales_with_concurrency(self) -> None:
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
        self.assertAlmostEqual(four, one * 4, places=6)

    def test_select_algorithm_prefers_edge_for_cpu(self) -> None:
        self.assertEqual(select_algorithm(goal="fit-memory", hardware="cpu"), "gguf-q4")

    def test_select_algorithm_prefers_fp8_for_hopper_throughput(self) -> None:
        self.assertEqual(select_algorithm(goal="throughput", hardware="hopper"), "fp8-dynamic")

    def test_recommendations_respect_compute_capability(self) -> None:
        recs = recommend_instances(required_gib=20, min_compute_capability=8.9)
        self.assertTrue(recs)
        self.assertTrue(all(rec.instance.compute_capability >= 8.9 for rec in recs))
        self.assertTrue(
            any("L4" in rec.instance.name or "4090" in rec.instance.name for rec in recs)
        )

    def test_invalid_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            estimate_serving_memory(
                params_b=0,
                scheme_key="w4a16",
                layers=32,
                hidden_size=4096,
                context_tokens=2048,
                concurrency=1,
            )


if __name__ == "__main__":
    unittest.main()

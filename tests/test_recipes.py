from __future__ import annotations

from compression_demo.catalog import ALGORITHMS
from compression_demo.recipes import recipe_snippet


def test_every_catalog_algorithm_has_recipe_or_explicit_stub_status() -> None:
    for key, algorithm in ALGORITHMS.items():
        snippet = recipe_snippet(key)
        assert snippet.strip()
        if "stub" in algorithm.status or "roadmap" in algorithm.status:
            assert "stub" in snippet.lower() or "outside" in snippet.lower()


def test_fp8_recipe_saves_compressed_checkpoint() -> None:
    snippet = recipe_snippet("fp8-dynamic")

    assert 'device_map="auto"' in snippet
    assert "save_compressed=True" in snippet

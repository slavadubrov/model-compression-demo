from __future__ import annotations

import pathlib
import tomllib


def load_pyproject() -> dict:
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def test_uv_project_has_pytest_and_ruff_dev_dependencies_without_heavy_extras() -> None:
    data = load_pyproject()

    assert data["project"]["name"] == "model-compression-demo"
    assert "optional-dependencies" not in data["project"]
    assert "pytest>=8.0.0" in data["dependency-groups"]["dev"]
    assert "ruff>=0.12.0" in data["dependency-groups"]["dev"]


def test_runtime_stacks_are_documented_but_not_project_extras() -> None:
    data = load_pyproject()

    stacks = data["tool"]["model-compression-demo"]["runtime-stacks"]
    assert "llmcompressor" in stacks["compression"]
    assert "gptqmodel" in stacks["alternatives"]
    assert "vllm" in stacks["serving"]


def test_cli_scripts_are_declared() -> None:
    data = load_pyproject()

    assert data["project"]["scripts"]["compression-demo"] == "compression_demo.cli:main"
    assert data["project"]["scripts"]["model-compression-demo"] == "compression_demo.cli:main"

from __future__ import annotations

import pathlib
import tomllib


def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def load_pyproject() -> dict:
    pyproject = project_root() / "pyproject.toml"
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
    assert "llmcompressor==0.6.0.1" in stacks["compression"]
    assert "vllm==0.23.0" in stacks["serving"]


def test_cli_scripts_are_declared() -> None:
    data = load_pyproject()

    assert data["project"]["scripts"]["compression-demo"] == "compression_demo.cli:main"
    assert data["project"]["scripts"]["model-compression-demo"] == "compression_demo.cli:main"


def test_python_floor_matches_ml_ecosystem() -> None:
    data = load_pyproject()

    assert data["project"]["requires-python"] == ">=3.11"
    assert data["tool"]["ruff"]["target-version"] == "py311"


def test_docs_include_architecture_descriptions_and_svg_diagrams() -> None:
    docs = project_root() / "docs"
    architecture = (docs / "architecture.md").read_text(encoding="utf-8")
    reports = (docs / "reports.md").read_text(encoding="utf-8")

    assert "Module Responsibilities" in architecture
    assert "architecture-overview.svg" in architecture
    assert "quality-benchmark-flow.svg" in architecture
    assert "Regeneration Commands" in reports

    for diagram in ("architecture-overview.svg", "quality-benchmark-flow.svg"):
        svg = (docs / diagram).read_text(encoding="utf-8")
        assert svg.startswith("<svg ")
        assert "<title" in svg
        assert "<desc" in svg
        assert "</svg>" in svg

from __future__ import annotations

import pathlib
import tomllib
import unittest


class ProjectMetadataTests(unittest.TestCase):
    def test_uv_project_has_ruff_dev_dependency(self) -> None:
        pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

        self.assertEqual(data["project"]["name"], "model-compression-demo")
        self.assertIn("gpu", data["project"]["optional-dependencies"])
        self.assertIn("alternatives", data["project"]["optional-dependencies"])
        self.assertIn("ruff>=0.12.0", data["dependency-groups"]["dev"])

    def test_cli_scripts_are_declared(self) -> None:
        pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

        self.assertEqual(
            data["project"]["scripts"]["compression-demo"], "compression_demo.cli:main"
        )
        self.assertEqual(
            data["project"]["scripts"]["model-compression-demo"],
            "compression_demo.cli:main",
        )

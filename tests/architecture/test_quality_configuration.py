from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYRIGHT_BASELINE = {
    "models/catalog.py",
    "model_runtime/__init__.py",
    "model_runtime/cli/__init__.py",
    "model_runtime/inspection/__init__.py",
    "model_runtime/packages/__init__.py",
    "model_runtime/packages/configuration.py",
    "model_runtime/packages/identity.py",
    "model_runtime/runs/__init__.py",
}
REQUIRED_GENERATED_EXCLUDES = {
    "**/__pycache__",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "node_modules",
    "torchenv",
    "workbench/.runtime",
    "workbench/frontend/.next",
    "workbench/frontend/node_modules",
}


class QualityConfigurationTests(unittest.TestCase):
    def test_strict_type_baseline_cannot_silently_shrink(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "pyrightconfig.json").read_text(encoding="utf-8")
        )

        self.assertEqual(config["typeCheckingMode"], "strict")
        self.assertEqual(config["pythonVersion"], "3.13")
        self.assertTrue(PYRIGHT_BASELINE.issubset(config["include"]))
        self.assertTrue(REQUIRED_GENERATED_EXCLUDES.issubset(config["exclude"]))
        self.assertNotIn("ignore", config)
        self.assertNotIn("diagnosticSeverityOverrides", config)

        for relative_path in config["include"]:
            source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
            with self.subTest(path=relative_path):
                self.assertNotIn("type: ignore", source)
                self.assertNotIn("pyright: ignore", source)

    def test_repository_versions_match_contracts(self) -> None:
        with (PROJECT_ROOT / "mise.toml").open("rb") as mise_file:
            mise = tomllib.load(mise_file)
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
            project = tomllib.load(project_file)

        self.assertEqual(mise["tools"], {"python": "3.13", "node": "24"})
        self.assertEqual(
            (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip(),
            "3.13",
        )
        self.assertEqual(project["project"]["requires-python"], ">=3.11,<3.14")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import ast
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
EMPEROR_ROOT = SOURCE_ROOT / "emperor"
MODEL_RUNTIME_ROOT = SOURCE_ROOT / "model_runtime"
PROJECT_CLI_ROOT = SOURCE_ROOT / "models" / "project_cli"
PUBLIC_RUNTIME_PACKAGES = ("packages", "inspection", "runs", "cli")


def _imports_under(root: Path) -> list[tuple[Path, str]]:
    imports: list[tuple[Path, str]] = []
    for source_path in sorted(root.rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            source_path.as_posix(),
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((source_path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((source_path, node.module))
    return imports


class ModelRuntimeBoundaryTests(unittest.TestCase):
    def test_project_cli_is_owned_outside_emperor(self) -> None:
        self.assertFalse((EMPEROR_ROOT / "cli").exists())
        self.assertFalse((EMPEROR_ROOT / "__main__.py").exists())
        self.assertTrue((PROJECT_CLI_ROOT / "__main__.py").is_file())

        with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
            project = tomllib.load(project_file)["project"]

        self.assertEqual(
            project["scripts"]["emperor"],
            "models.project_cli:main",
        )
        self.assertNotIn("emperor.project_adapter", project.get("entry-points", {}))

    def test_runtime_public_package_shell_exists(self) -> None:
        self.assertTrue((MODEL_RUNTIME_ROOT / "__init__.py").is_file())
        for package in PUBLIC_RUNTIME_PACKAGES:
            with self.subTest(package=package):
                self.assertTrue(
                    (MODEL_RUNTIME_ROOT / package / "__init__.py").is_file()
                )

    def test_emperor_has_no_outward_project_imports(self) -> None:
        forbidden = [
            (path, module)
            for path, module in _imports_under(EMPEROR_ROOT)
            if module == "model_runtime"
            or module.startswith("model_runtime.")
            or module == "models"
            or module.startswith("models.")
            or module == "workbench"
            or module.startswith("workbench.")
            or module == "emperor_workbench"
            or module.startswith("emperor_workbench.")
        ]

        self.assertEqual(forbidden, [])

    def test_generic_runtime_has_no_project_or_workbench_imports(self) -> None:
        forbidden = [
            (path, module)
            for path, module in _imports_under(MODEL_RUNTIME_ROOT)
            if module == "models"
            or module.startswith("models.")
            or module == "workbench"
            or module.startswith("workbench.")
            or module == "emperor_workbench"
            or module.startswith("emperor_workbench.")
        ]

        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()

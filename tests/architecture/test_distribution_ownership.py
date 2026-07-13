from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_TREES = ("docs", "emperor", "model_runtime", "models")


class DistributionOwnershipTests(unittest.TestCase):
    def test_executable_tests_are_external_to_production_source(self) -> None:
        leaked_tests = sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for tree in PRODUCTION_TREES
            for path in (PROJECT_ROOT / tree).rglob("test*.py")
        )

        self.assertEqual(leaked_tests, [])

    def test_setuptools_discovery_names_the_installable_packages(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
            project = tomllib.load(project_file)

        discovery = project["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(
            discovery["include"],
            [
                "emperor",
                "emperor.*",
                "models",
                "models.*",
                "model_runtime",
                "model_runtime.*",
            ],
        )
        self.assertEqual(discovery["where"], ["."])
        self.assertEqual(
            discovery["exclude"],
            ["docs", "docs.*", "tests", "tests.*", "workbench", "workbench.*"],
        )
        self.assertFalse(discovery["namespaces"])

    def test_workbench_owns_its_web_dependencies_and_package_config(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
            project = tomllib.load(project_file)
        with (PROJECT_ROOT / "workbench" / "pyproject.toml").open(
            "rb"
        ) as project_file:
            workbench = tomllib.load(project_file)

        root_requirements = tuple(project["project"]["dependencies"])
        forbidden_roots = (
            "fastapi",
            "httpx",
            "psutil",
            "pydantic-settings",
            "starlette",
            "uvicorn",
            "pywin32",
        )
        for dependency in forbidden_roots:
            with self.subTest(dependency=dependency):
                self.assertFalse(
                    any(
                        requirement.partition("[")[0]
                        .partition(">=")[0]
                        .partition(";")[0]
                        .strip()
                        == dependency
                        for requirement in root_requirements
                    )
                )

        workbench_requirements = "\n".join(workbench["project"]["dependencies"])
        for dependency in forbidden_roots:
            with self.subTest(workbench_dependency=dependency):
                self.assertIn(dependency, workbench_requirements)
        self.assertEqual(
            workbench["project"]["scripts"]["emperor-workbench"],
            "workbench.backend.launch:main",
        )
        self.assertNotIn(
            "workbench.backend.tests",
            workbench["tool"]["setuptools"]["packages"],
        )
        self.assertNotIn(
            "workbench.backend.inspector",
            workbench["tool"]["setuptools"]["packages"],
        )

    def test_installed_packages_import_outside_the_checkout(self) -> None:
        smoke = """
from importlib.metadata import distribution

distribution('emperor')
import emperor
import model_runtime
import models
"""
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTHONSAFEPATH"] = "1"
        with tempfile.TemporaryDirectory() as outside_checkout:
            completed = subprocess.run(
                [sys.executable, "-P", "-c", smoke],
                cwd=outside_checkout,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

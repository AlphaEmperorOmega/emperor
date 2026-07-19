from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
PRODUCTION_SOURCE_ROOTS = tuple(
    SOURCE_ROOT / package for package in ("emperor", "model_runtime", "models")
)


class DistributionOwnershipTests(unittest.TestCase):
    def test_executable_tests_are_external_to_production_source(self) -> None:
        leaked_tests = sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for source_root in PRODUCTION_SOURCE_ROOTS
            for path in source_root.rglob("test*.py")
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
        self.assertEqual(project["tool"]["setuptools"]["package-dir"], {"": "src"})
        self.assertEqual(discovery["where"], ["src"])
        self.assertNotIn("exclude", discovery)
        self.assertFalse(discovery["namespaces"])

    def test_workbench_api_owns_its_dependencies_and_package_config(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
            project = tomllib.load(project_file)
        with (PROJECT_ROOT / "apps" / "workbench" / "api" / "pyproject.toml").open(
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

        workbench_requirements = "\n".join(
            [
                *workbench["project"]["dependencies"],
                *workbench["project"]["optional-dependencies"]["dev"],
            ]
        )
        for dependency in forbidden_roots:
            with self.subTest(workbench_dependency=dependency):
                self.assertIn(dependency, workbench_requirements)
        self.assertEqual(
            workbench["project"]["scripts"]["emperor-workbench"],
            "emperor_workbench.cli:main",
        )
        self.assertEqual(
            workbench["tool"]["setuptools"]["package-dir"],
            {"": "src"},
        )
        discovery = workbench["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(discovery["where"], ["src"])
        self.assertEqual(
            discovery["include"],
            ["emperor_workbench", "emperor_workbench.*"],
        )
        self.assertFalse(discovery["namespaces"])

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

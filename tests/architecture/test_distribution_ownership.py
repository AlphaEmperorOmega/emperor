from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
PRODUCTION_PACKAGES = ("emperor", "model_runtime", "models")
PRODUCTION_SOURCE_ROOTS = tuple(
    SOURCE_ROOT / package for package in PRODUCTION_PACKAGES
) + (PROJECT_ROOT / "apps" / "workbench" / "api" / "src" / "emperor_workbench",)
DEPENDENCY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _dependency_names(requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        match = DEPENDENCY_NAME.match(requirement)
        if match is None:
            raise AssertionError(f"Invalid dependency requirement: {requirement!r}")
        names.add(re.sub(r"[-_.]+", "-", match.group(0)).lower())
    return names


class DistributionOwnershipTests(unittest.TestCase):
    def test_distribution_manifest_includes_test_contracts(self) -> None:
        manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertIn("recursive-include tests *.json *.py *.toml", manifest)
        interface_manifest = (
            PROJECT_ROOT / "tests" / "architecture" / "emperor_interfaces.toml"
        )
        self.assertTrue(interface_manifest.is_file())

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

        root_runtime = _dependency_names(project["project"]["dependencies"])
        root_dev = _dependency_names(project["project"]["optional-dependencies"]["dev"])
        self.assertEqual(
            root_runtime,
            {
                "datasets",
                "filelock",
                "gymnasium",
                "ipython",
                "lightning",
                "matplotlib",
                "numpy",
                "pillow",
                "tensorboard",
                "tokenizers",
                "torch",
                "torchmetrics",
                "torchtext",
                "torchvision",
            },
        )
        self.assertEqual(
            root_dev,
            {
                "build",
                "coverage",
                "mutmut",
                "psutil",
                "ruff",
            },
        )

        workbench_runtime = _dependency_names(workbench["project"]["dependencies"])
        workbench_dev = _dependency_names(
            workbench["project"]["optional-dependencies"]["dev"]
        )
        self.assertEqual(
            workbench_runtime,
            {
                "emperor",
                "fastapi",
                "pydantic",
                "pydantic-settings",
                "pywin32",
                "starlette",
                "tensorboard",
                "torch",
                "typing-extensions",
                "uvicorn",
            },
        )
        self.assertEqual(
            workbench_dev,
            {
                "httpx",
                "psutil",
                "pytest",
                "ruff",
            },
        )
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

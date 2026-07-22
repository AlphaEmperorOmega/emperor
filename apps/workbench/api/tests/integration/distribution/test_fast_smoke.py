from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FAST_SMOKE_PUBLIC_INTERFACES = (
    "emperor_workbench.failures",
    "emperor_workbench.filesystem",
    "emperor_workbench.log_experiments",
    "emperor_workbench.project_adapter",
    "emperor_workbench.model_packages",
    "emperor_workbench.config_snapshots",
    "emperor_workbench.run_plans",
)

HEAVY_RUNTIME_MODULE_ROOTS = frozenset(
    {
        "fastapi",
        "httpx",
        "lightning",
        "pydantic",
        "pydantic_settings",
        "tensorboard",
        "torch",
    }
)


class FastSmokeSuiteTests(unittest.TestCase):
    def test_public_interfaces_import_without_heavy_runtime_dependencies(self) -> None:
        before_imports = set(sys.modules)

        for module_name in FAST_SMOKE_PUBLIC_INTERFACES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

        imported_roots = {
            module_name.split(".", 1)[0]
            for module_name in set(sys.modules) - before_imports
        }
        self.assertEqual(
            sorted(imported_roots & HEAVY_RUNTIME_MODULE_ROOTS),
            [],
        )

    def test_inspection_facade_import_is_lazy(self) -> None:
        script = """
import sys
import model_runtime.inspection
from models.catalog import discover_model_packages

loaded = [
    package.catalog_key
    for package in discover_model_packages()
    for package_module in (
        f'models.{package.identity.model_type}.{package.identity.model}',
    )
    if any(
        name.startswith(package_module + '.')
        for name in sys.modules
    )
]
if loaded:
    raise SystemExit(f'Inspection facade loaded Model Packages: {loaded}')
if 'torch' in sys.modules:
    raise SystemExit('Inspection facade imported Torch')
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=".",
            env={
                **os.environ,
                "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "matplotlib"),
            },
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_lightweight_app_startup_does_not_load_a_model_package(self) -> None:
        script = """
import sys
from emperor_workbench.api import create_app
from models.catalog import discover_model_packages

create_app()
loaded = [
    package.catalog_key
    for package in discover_model_packages()
    for package_module in (
        f'models.{package.identity.model_type}.{package.identity.model}',
    )
    if any(
        name.startswith(package_module + '.')
        for name in sys.modules
    )
]
if loaded:
    raise SystemExit(f'Workbench startup loaded Model Packages: {loaded}')
if 'torch' in sys.modules:
    raise SystemExit('Workbench startup imported Torch')
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=".",
            env={
                **os.environ,
                "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "matplotlib"),
            },
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

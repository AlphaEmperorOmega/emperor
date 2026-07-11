from __future__ import annotations

import importlib
import os
import subprocess
import sys
import unittest

FAST_SMOKE_MODULES = (
    "workbench.backend.storage.local_files",
    "workbench.backend.training_jobs.status",
    "workbench.backend.config_snapshots",
    "workbench.backend.training_jobs.store",
    "workbench.backend.tests.test_dependency_direction",
    "workbench.backend.tests.test_local_files",
    "workbench.backend.tests.test_job_status",
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
    def test_fast_smoke_modules_import_without_heavy_runtime_dependencies(self) -> None:
        before_imports = set(sys.modules)

        for module_name in FAST_SMOKE_MODULES:
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
import emperor.inspection
from emperor.model_packages import discover_model_packages

loaded = [
    package.module_path
    for package in discover_model_packages()
    if any(
        name == package.module_path or name.startswith(package.module_path + '.')
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
            env={**os.environ, "MPLCONFIGDIR": "/tmp/matplotlib"},
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_lightweight_app_startup_does_not_load_a_model_package(self) -> None:
        script = """
import sys
from workbench.backend.main import create_app
from emperor.model_packages import discover_model_packages

create_app()
loaded = [
    package.module_path
    for package in discover_model_packages()
    if any(
        name == package.module_path or name.startswith(package.module_path + '.')
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
            env={**os.environ, "MPLCONFIGDIR": "/tmp/matplotlib"},
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import sys
import unittest

FAST_SMOKE_MODULES = (
    "viewer.backend.storage.local_files",
    "viewer.backend.runtime.job_status",
    "viewer.backend.config_snapshots",
    "viewer.backend.job_store",
    "viewer.backend.tests.test_dependency_direction",
    "viewer.backend.tests.test_local_files",
    "viewer.backend.tests.test_job_status",
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


if __name__ == "__main__":
    unittest.main()

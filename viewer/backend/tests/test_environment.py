from __future__ import annotations

import importlib
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


REQUIRED_BACKEND_TEST_MODULES = (
    ("torch", "torch"),
    ("fastapi", "fastapi"),
    ("tensorboard", "tensorboard"),
    ("pydantic", "pydantic"),
    ("pydantic-settings", "pydantic_settings"),
    ("httpx", "httpx"),
    ("lightning", "lightning.pytorch"),
)


class BackendTestEnvironmentTests(unittest.TestCase):
    def test_required_backend_test_dependencies_are_importable(self) -> None:
        missing: list[str] = []

        for package_name, module_name in REQUIRED_BACKEND_TEST_MODULES:
            with self.subTest(module=module_name):
                try:
                    importlib.import_module(module_name)
                except ModuleNotFoundError as error:
                    missing.append(
                        f"{package_name} ({module_name}; missing {error.name})"
                    )

        self.assertEqual(
            missing,
            [],
            "Missing backend test dependencies. Install the project dependencies "
            "with `python -m pip install -e .` or run tests from an environment "
            f"where these modules are available: {', '.join(missing)}",
        )


if __name__ == "__main__":
    unittest.main()

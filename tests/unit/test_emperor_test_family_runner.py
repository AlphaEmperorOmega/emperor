from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_emperor_test_family():
    name = "_emperor_test_family_runner_tests"
    spec = importlib.util.spec_from_file_location(
        name,
        PROJECT_ROOT / "tools" / "emperor_test_family.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Emperor test-family runner.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


emperor_test_family = _load_emperor_test_family()
coverage_include_argument = emperor_test_family.coverage_include_argument
load_family = emperor_test_family.load_family
resolve_test_module_names = emperor_test_family.test_module_names


class EmperorTestFamilyRunnerTests(unittest.TestCase):
    def test_python_environment_prefers_src_and_disables_implicit_cwd_imports(
        self,
    ) -> None:
        environment = emperor_test_family._python_environment(PROJECT_ROOT)

        self.assertEqual(environment["PYTHONSAFEPATH"], "1")
        self.assertEqual(
            environment["PYTHONPATH"].split(os.pathsep)[:3],
            [
                str(PROJECT_ROOT / "src"),
                str(PROJECT_ROOT / "tests"),
                str(PROJECT_ROOT),
            ],
        )

    def test_family_modules_include_deduplicated_integration_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            focused = root / "tests" / "unit" / "test_family.py"
            integration = root / "tests" / "integration" / "test_family_runtime.py"
            focused.parent.mkdir(parents=True)
            integration.parent.mkdir(parents=True)
            focused.touch()
            integration.touch()
            family = {
                "focused_tests": ["tests/unit/test_family.py"],
                "integration_tests": [
                    "tests/unit/test_family.py",
                    "tests/integration/test_family_runtime.py",
                ],
            }

            modules = emperor_test_family._family_test_modules(root, family)

        self.assertEqual(
            modules,
            (
                "tests.integration.test_family_runtime",
                "tests.unit.test_family",
            ),
        )

    def test_resolved_test_files_convert_to_unittest_module_names(self) -> None:
        paths = (
            PROJECT_ROOT / "tests" / "unit" / "test_linears.py",
            PROJECT_ROOT / "tests" / "integration" / "test_linear_monitor_lifecycle.py",
        )

        self.assertEqual(
            resolve_test_module_names(PROJECT_ROOT, paths),
            (
                "tests.unit.test_linears",
                "tests.integration.test_linear_monitor_lifecycle",
            ),
        )

    def test_coverage_include_argument_uses_only_registered_family_modules(
        self,
    ) -> None:
        family = load_family(PROJECT_ROOT, "linears")

        include_argument = coverage_include_argument(family["modules"])

        self.assertEqual(
            include_argument,
            "--include=" + ",".join(family["modules"]),
        )
        self.assertNotIn("src/emperor/attention/", include_argument)


if __name__ == "__main__":
    unittest.main()

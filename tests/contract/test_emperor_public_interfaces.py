from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "tests" / "architecture" / "emperor_interfaces.toml"


def _load_manifest() -> dict[str, object]:
    with MANIFEST_PATH.open("rb") as manifest_file:
        return tomllib.load(manifest_file)


class EmperorPublicInterfaceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _load_manifest()
        cls.protected_interfaces = cls.manifest["protected_interfaces"]

    def test_public_interfaces_resolve_only_declared_exports(self) -> None:
        for module_name, contract in self.protected_interfaces.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                expected_exports = tuple(contract["exports"])
                self.assertEqual(tuple(module.__all__), expected_exports)
                for export_name in expected_exports:
                    self.assertIsNotNone(getattr(module, export_name))
                missing_export = "__emperor_undeclared_interface_export__"
                with self.assertRaises(AttributeError):
                    getattr(module, missing_export)

    def test_all_public_interfaces_import_in_a_fresh_interpreter(self) -> None:
        module_names = tuple(self.protected_interfaces)
        script = "\n".join(
            (
                "import importlib",
                f"modules = {module_names!r}",
                "for name in modules:",
                "    module = importlib.import_module(name)",
                "    assert tuple(module.__all__)",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-P", "-c", script],
            cwd=PROJECT_ROOT,
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

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tomllib
import unittest
from operator import attrgetter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "tests" / "architecture" / "emperor_interfaces.toml"
FORBIDDEN_LIGHTWEIGHT_DEPENDENCIES = (
    "torch",
    "lightning",
    "matplotlib",
    "torchvision",
    "torchtext",
    "tokenizers",
)


def _load_manifest() -> dict[str, object]:
    with MANIFEST_PATH.open("rb") as manifest_file:
        return tomllib.load(manifest_file)


class EmperorPublicInterfaceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _load_manifest()
        cls.interfaces = {
            interface["module"]: interface for interface in cls.manifest["interfaces"]
        }

    def run_fresh_python(self, source: str) -> subprocess.CompletedProcess[str]:
        python_path = str(PROJECT_ROOT / "src")
        if os.environ.get("PYTHONPATH"):
            python_path = f"{python_path}{os.pathsep}{os.environ['PYTHONPATH']}"
        return subprocess.run(
            [sys.executable, "-P", "-c", source],
            cwd=PROJECT_ROOT,
            capture_output=True,
            env={**os.environ, "PYTHONPATH": python_path},
            text=True,
            check=False,
        )

    def test_interfaces_resolve_exact_exports(self) -> None:
        for module_name, contract in self.interfaces.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                exports = tuple(contract["exports"])
                self.assertEqual(tuple(module.__all__), exports)
                for export_name in exports:
                    self.assertIsNotNone(getattr(module, export_name))
                with self.assertRaises(AttributeError):
                    attrgetter("__emperor_undeclared_interface_export__")(module)

    def test_all_interfaces_import_in_a_fresh_interpreter(self) -> None:
        contracts = {
            module_name: tuple(contract["exports"])
            for module_name, contract in self.interfaces.items()
        }
        completed = self.run_fresh_python(
            "\n".join(
                (
                    "import importlib",
                    f"contracts = {contracts!r}",
                    "for module_name, exports in contracts.items():",
                    "    module = importlib.import_module(module_name)",
                    "    assert tuple(module.__all__) == exports",
                )
            )
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_configuration_and_namespace_imports_are_lightweight(self) -> None:
        lightweight_modules = tuple(
            module_name
            for module_name, contract in self.interfaces.items()
            if contract["kind"] in {"configuration", "namespace"}
        )
        completed = self.run_fresh_python(
            f"""
import importlib
import json
import sys

for module_name in {lightweight_modules!r}:
    importlib.import_module(module_name)

forbidden = {FORBIDDEN_LIGHTWEIGHT_DEPENDENCIES!r}
loaded = {{
    dependency: any(
        name == dependency or name.startswith(f"{{dependency}}.")
        for name in sys.modules
    )
    for dependency in forbidden
}}
print(json.dumps(loaded))
"""
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            dict.fromkeys(FORBIDDEN_LIGHTWEIGHT_DEPENDENCIES, False),
        )

    def test_retired_package_root_exports_fail_immediately(self) -> None:
        retired_exports = sorted(
            (
                (entry["module"], tuple(entry["names"]))
                for entry in self.manifest["retired_root_exports"]
            ),
            key=lambda entry: (entry[0].count("."), entry[0]),
        )
        completed = self.run_fresh_python(
            f"""
import importlib
import json

results = {{}}
for module_name, names in {retired_exports!r}:
    module = importlib.import_module(module_name)
    results[module_name] = {{name: hasattr(module, name) for name in names}}
print(json.dumps(results))
"""
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        actual = json.loads(completed.stdout)
        expected = {
            module_name: dict.fromkeys(names, False)
            for module_name, names in retired_exports
        }
        self.assertEqual(actual, expected)

    def test_retired_owner_modules_cannot_be_imported(self) -> None:
        retired_modules = tuple(self.manifest["retired_modules"])
        completed = self.run_fresh_python(
            f"""
import importlib
import json

results = {{}}
for module_name in {retired_modules!r}:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        results[module_name] = False
    else:
        results[module_name] = True
print(json.dumps(results))
"""
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            dict.fromkeys(retired_modules, False),
        )

    def test_symbols_removed_from_extant_private_modules_stay_removed(self) -> None:
        for entry in self.manifest["retired_symbol_imports"]:
            with self.subTest(module=entry["module"], name=entry["name"]):
                module = importlib.import_module(entry["module"])
                self.assertFalse(hasattr(module, entry["name"]))

    def test_retired_runtime_members_stay_removed(self) -> None:
        for entry in self.manifest["retired_members"]:
            module = importlib.import_module(entry["module"])
            owner = getattr(module, entry["owner"])
            for member_name in entry["names"]:
                with self.subTest(
                    module=entry["module"],
                    owner=entry["owner"],
                    member=member_name,
                ):
                    self.assertFalse(hasattr(owner, member_name))


if __name__ == "__main__":
    unittest.main()

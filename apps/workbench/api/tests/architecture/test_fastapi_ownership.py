from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.architecture._support import (
    app_state_violations,
    defines_http_contracts,
    framework_imports_outside_api,
    http_contract_modules_outside_api,
    is_transport_framework_import,
    load_manifest,
    manifest_records,
    module_import_side_effects,
)


class WorkbenchFastApiOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()

    def test_framework_import_ledger_is_exact(self) -> None:
        actual = tuple(
            violation.as_dict() for violation in framework_imports_outside_api()
        )
        expected = manifest_records(
            self.manifest,
            "legacy_framework_imports_outside_api",
        )
        self.assertEqual(expected, actual)

    def test_framework_detector_covers_bare_and_nested_transport_imports(self) -> None:
        for module_name in (
            "fastapi",
            "fastapi.routing",
            "starlette",
            "starlette.responses",
        ):
            with self.subTest(module=module_name):
                self.assertTrue(is_transport_framework_import(module_name))
        self.assertFalse(is_transport_framework_import("pydantic"))

    def test_http_contract_ledger_is_exact(self) -> None:
        self.assertEqual(
            tuple(self.manifest["legacy_http_contract_modules_outside_api"]),
            http_contract_modules_outside_api(),
        )

    def test_http_contract_detection_is_location_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transport_contract = root / "transport_contract.py"
            transport_contract.write_text(
                "from pydantic import BaseModel\n"
                "\n"
                "class ExampleResponse(BaseModel):\n"
                "    value: str\n",
                encoding="utf-8",
            )
            semantic_record = root / "semantic_record.py"
            semantic_record.write_text(
                "from dataclasses import dataclass\n"
                "\n"
                "@dataclass(frozen=True)\n"
                "class HistoricalInspectionRequest:\n"
                "    run_id: str\n",
                encoding="utf-8",
            )

            self.assertTrue(defines_http_contracts(transport_contract))
            self.assertFalse(defines_http_contracts(semantic_record))

    def test_app_scoped_module_state_ledger_is_exact(self) -> None:
        actual = tuple(violation.as_dict() for violation in app_state_violations())
        expected = manifest_records(
            self.manifest,
            "legacy_app_state_violations",
        )
        self.assertEqual(expected, actual)

    def test_import_purity_probe_detects_resource_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module_path = root / "impure_import_target.py"
            module_path.write_text(
                "import os\n"
                "from pathlib import Path\n"
                "import subprocess\n"
                "import sys\n"
                "import threading\n"
                "\n"
                "Path(\n"
                '    os.environ["EMPEROR_IMPORT_PURITY_ROOT"],\n'
                '    "created.txt",\n'
                ').write_text("created", encoding="utf-8")\n'
                "try:\n"
                "    threading.Thread(target=lambda: None).start()\n"
                "except RuntimeError:\n"
                "    pass\n"
                "try:\n"
                '    subprocess.Popen([sys.executable, "-c", "pass"])\n'
                "except RuntimeError:\n"
                "    pass\n",
                encoding="utf-8",
            )

            effects = module_import_side_effects(
                "impure_import_target",
                python_paths=(root,),
            )

        self.assertTrue(
            any(effect.startswith("filesystem:") for effect in effects),
            effects,
        )
        self.assertTrue(
            any(effect.startswith("thread:") for effect in effects),
            effects,
        )
        self.assertIn("process:subprocess.Popen", effects)

    def test_api_package_import_is_resource_pure(self) -> None:
        self.assertEqual((), module_import_side_effects("emperor_workbench.api"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import subprocess
import sys
import unittest

PUBLIC_EXPORTS = (
    ("ConfigurationField", "model_runtime.inspection.records"),
    ("ConfigurationFieldCondition", "model_runtime.inspection.records"),
    ("ConfigurationSchema", "model_runtime.inspection.records"),
    ("GraphConfiguration", "model_runtime.inspection.records"),
    ("GraphConfigurationField", "model_runtime.inspection.records"),
    ("GraphEdge", "model_runtime.inspection.records"),
    ("GraphNode", "model_runtime.inspection.records"),
    ("GraphRole", "model_runtime.inspection.records"),
    ("InspectionError", "model_runtime.inspection.errors"),
    ("InspectionCaptureLimits", "model_runtime.inspection.capture_limits"),
    ("InspectionRequest", "model_runtime.inspection.records"),
    ("InspectionResult", "model_runtime.inspection.records"),
    ("MethodShapeTrace", "model_runtime.inspection.shape_trace"),
    ("ModelGraph", "model_runtime.inspection.records"),
    ("ModelShapeTrace", "model_runtime.inspection.shape_trace"),
    ("ModuleShapeCall", "model_runtime.inspection.shape_trace"),
    ("ModuleShapeTrace", "model_runtime.inspection.shape_trace"),
    ("ParsedOverrides", "model_runtime.inspection.records"),
    ("SearchAxis", "model_runtime.inspection.records"),
    ("SearchSpace", "model_runtime.inspection.records"),
    ("ShapeTraceDetail", "model_runtime.inspection.shape_trace"),
    ("TensorShape", "model_runtime.inspection.shape_trace"),
    ("TensorVariableTrace", "model_runtime.inspection.shape_trace"),
    ("canonicalize_overrides", "model_runtime.inspection.overrides"),
    ("configuration_schema", "model_runtime.inspection.schema"),
    ("config_field_description", "model_runtime.inspection.field_descriptions"),
    ("inspect_model_shapes", "model_runtime.inspection.shape_trace"),
    ("inspect_model", "model_runtime.inspection.service"),
    ("parse_overrides", "model_runtime.inspection.overrides"),
    ("preset_locks", "model_runtime.inspection.schema"),
    ("reject_locked_overrides", "model_runtime.inspection.overrides"),
    ("resolve_override_key", "model_runtime.inspection.overrides"),
    ("search_space_schema", "model_runtime.inspection.schema"),
    ("serialize_overrides", "model_runtime.inspection.overrides"),
    ("supported_config_keys", "model_runtime.inspection.overrides"),
    ("validate_configuration", "model_runtime.inspection.service"),
)


class InspectionPublicInterfaceTests(unittest.TestCase):
    def test_lazy_public_exports_have_exact_order_ownership_and_identity(self) -> None:
        inspection = importlib.import_module("model_runtime.inspection")

        self.assertEqual(
            inspection.__all__,
            [name for name, _module_name in PUBLIC_EXPORTS],
        )
        for name, module_name in PUBLIC_EXPORTS:
            with self.subTest(name=name):
                owner = importlib.import_module(module_name)
                exported = getattr(inspection, name)
                self.assertIs(exported, getattr(owner, name))
                self.assertIs(getattr(inspection, name), exported)
                self.assertNotIn(name, vars(inspection))

    def test_importing_public_facade_does_not_load_export_owners(self) -> None:
        owner_modules = sorted({module_name for _name, module_name in PUBLIC_EXPORTS})
        script = (
            "import sys\n"
            "import model_runtime.inspection\n"
            f"owners = {owner_modules!r}\n"
            "loaded = [name for name in owners if name in sys.modules]\n"
            "raise SystemExit(f'loaded eagerly: {loaded}' if loaded else 0)\n"
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_unknown_public_attribute_retains_exact_error(self) -> None:
        inspection = importlib.import_module("model_runtime.inspection")
        missing_name = "missing_export"

        with self.assertRaises(AttributeError) as raised:
            getattr(inspection, missing_name)

        self.assertEqual(
            str(raised.exception),
            "module 'model_runtime.inspection' has no attribute 'missing_export'",
        )


if __name__ == "__main__":
    unittest.main()

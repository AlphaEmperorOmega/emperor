import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from emperor.memory import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
    MemoryInterface,
    MemoryPositionOptions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPORTS = (
    "AttentionDynamicMemoryConfig",
    "DynamicMemoryConfig",
    "ElementWiseWeightedDynamicMemoryConfig",
    "GatedResidualDynamicMemoryConfig",
    "MemoryInterface",
    "MemoryPositionOptions",
    "WeightedDynamicMemoryConfig",
)

EXPECTED_OWNERS = {
    "AttentionDynamicMemoryConfig": "emperor.memory._config",
    "DynamicMemoryConfig": "emperor.memory._config",
    "ElementWiseWeightedDynamicMemoryConfig": "emperor.memory._config",
    "GatedResidualDynamicMemoryConfig": "emperor.memory._config",
    "MemoryInterface": "emperor.memory._interface",
    "MemoryPositionOptions": "emperor.memory._config",
    "WeightedDynamicMemoryConfig": "emperor.memory._config",
}

BASE_CONFIG_FIELDS = (
    "input_dim",
    "output_dim",
    "memory_position_option",
    "test_time_training_learning_rate",
    "test_time_training_num_inner_steps",
    "model_config",
)


class TestMemoryPublicInterface(unittest.TestCase):
    def test_unknown_attribute_raises_exact_module_error(self):
        import emperor.memory as memory

        with self.assertRaisesRegex(
            AttributeError,
            "^module 'emperor.memory' has no attribute 'missing_memory_export'$",
        ):
            self.assertIsNone(memory.missing_memory_export)

    def test_root_interface_eagerly_exports_only_lightweight_configuration(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json
import sys

import emperor.memory as memory

expected_eager_modules = (
    "emperor.memory._config",
    "emperor.memory._interface",
)
heavy_modules = (
    "emperor.memory._base",
    "emperor.memory._validation",
    "emperor.memory._variants.attention",
    "emperor.memory._variants.element_wise_weighted",
    "emperor.memory._variants.gated_residual",
    "emperor.memory._variants.weighted",
    "emperor.memory._monitoring",
)
runtime_loaded = {
    "emperor.layers": "emperor.layers" in sys.modules,
    "lightning": "lightning" in sys.modules,
    "torch": "torch" in sys.modules,
}
owners = {name: getattr(memory, name).__module__ for name in memory.__all__}

print(json.dumps({
    "all": memory.__all__,
    "expected_eager_modules": {
        name: name in sys.modules for name in expected_eager_modules
    },
    "heavy_modules": {name: name in sys.modules for name in heavy_modules},
    "owners": owners,
    "private_exports": {
        name: hasattr(memory, name)
        for name in (
            "AdaptiveGeneratorValidatorBase",
            "AttentionDynamicMemory",
            "DynamicMemoryAbstract",
            "DynamicMemoryValidator",
            "ElementWiseWeightedDynamicMemory",
            "GatedResidualDynamicMemory",
            "MemoryMonitorCallback",
            "WeightedDynamicMemory",
        )
    },
    "runtime_loaded": runtime_loaded,
    "shortcut_attributes": {
        "__getattr__": hasattr(memory, "__getattr__"),
        "_LAZY_EXPORTS": hasattr(memory, "_LAZY_EXPORTS"),
    },
}))
""",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(
                    Path(tempfile.gettempdir()) / "matplotlib-memory-interface"
                ),
            },
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)

        self.assertEqual(tuple(result["all"]), EXPECTED_EXPORTS)
        self.assertEqual(result["owners"], EXPECTED_OWNERS)
        self.assertEqual(
            result["expected_eager_modules"],
            dict.fromkeys(result["expected_eager_modules"], True),
        )
        self.assertEqual(
            result["heavy_modules"],
            dict.fromkeys(result["heavy_modules"], False),
        )
        self.assertEqual(
            result["private_exports"],
            dict.fromkeys(result["private_exports"], False),
        )
        self.assertEqual(
            result["runtime_loaded"],
            {"emperor.layers": False, "lightning": False, "torch": False},
        )
        self.assertEqual(
            result["shortcut_attributes"],
            {"__getattr__": False, "_LAZY_EXPORTS": False},
        )

    def test_monitoring_has_its_own_explicit_interface(self):
        import emperor.memory as memory
        import emperor.memory.monitoring as monitoring

        self.assertEqual(monitoring.__all__, ("MemoryMonitorCallback",))
        self.assertEqual(
            monitoring.MemoryMonitorCallback.__module__,
            "emperor.memory._monitoring",
        )
        self.assertFalse(hasattr(memory, "MemoryMonitorCallback"))

    def test_removed_implementations_cannot_be_imported_from_the_root_interface(self):
        removed_names = (
            "AttentionDynamicMemory",
            "DynamicMemoryAbstract",
            "ElementWiseWeightedDynamicMemory",
            "GatedResidualDynamicMemory",
            "MemoryMonitorCallback",
            "WeightedDynamicMemory",
        )

        for name in removed_names:
            with self.subTest(name=name):
                with self.assertRaises(ImportError):
                    exec(f"from emperor.memory import {name}", {})

    def test_owner_protocol_is_non_constructible(self):
        with self.assertRaises(TypeError):
            MemoryInterface()

    def test_config_schema_and_enum_values_are_preserved(self):
        self.assertEqual(
            tuple(field.name for field in fields(DynamicMemoryConfig)),
            BASE_CONFIG_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(AttentionDynamicMemoryConfig)),
            (*BASE_CONFIG_FIELDS, "num_memory_slots"),
        )
        self.assertEqual(
            tuple((option.name, option.value) for option in MemoryPositionOptions),
            (("BEFORE_AFFINE", 1), ("AFTER_AFFINE", 2)),
        )


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import torch

from emperor.memory import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryInterface,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,
)
from unit.test_memory import MEMORY_CASES, make_memory_config

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPORTS = (
    "AttentionDynamicMemoryConfig",
    "DynamicMemoryConfig",
    "ElementWiseWeightedDynamicMemoryConfig",
    "GatedResidualDynamicMemoryConfig",
    "MemoryInterface",
    "MemoryMonitorCallback",
    "MemoryPositionOptions",
    "WeightedDynamicMemoryConfig",
)

EXPECTED_OWNERS = {
    "AttentionDynamicMemoryConfig": "emperor.memory._config",
    "DynamicMemoryConfig": "emperor.memory._config",
    "ElementWiseWeightedDynamicMemoryConfig": "emperor.memory._config",
    "GatedResidualDynamicMemoryConfig": "emperor.memory._config",
    "MemoryInterface": "emperor.memory._interface",
    "MemoryMonitorCallback": "emperor.memory._monitoring",
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

VARIANT_CHILDREN = {
    GatedResidualDynamicMemoryConfig: (
        "memory_model",
        "memory_gate_model",
    ),
    WeightedDynamicMemoryConfig: (
        "memory_model",
        "memory_weight_model",
    ),
    ElementWiseWeightedDynamicMemoryConfig: (
        "memory_model",
        "memory_weight_model",
    ),
    AttentionDynamicMemoryConfig: (
        "memory_model",
        "query_model",
        "key_model",
        "value_model",
        "output_model",
        "memory_gate_model",
    ),
}

RNG_DIGESTS = {
    GatedResidualDynamicMemoryConfig: (
        "d8d218f5608821f0b04863478a664be7b676f6e6e0f591adc0cd4daa7d13d8a7",
        "530c32ff8e751b9311c3d232440ebe32ceb1923c29f7e7b190d1a81d37126fb7",
    ),
    WeightedDynamicMemoryConfig: (
        "bff69a9d5f5bfd2808eae9547c35f2a0c32f6bc2f6fe7bbcd3ad5ee7bf5c9ce8",
        "d8383d41e0b55a18cf5a804218b9988e4e4921a0a0036317ed68e8513dd5b0bf",
    ),
    ElementWiseWeightedDynamicMemoryConfig: (
        "d8d218f5608821f0b04863478a664be7b676f6e6e0f591adc0cd4daa7d13d8a7",
        "530c32ff8e751b9311c3d232440ebe32ceb1923c29f7e7b190d1a81d37126fb7",
    ),
    AttentionDynamicMemoryConfig: (
        "7eaaf2dfcc90ab08cc59ba4410b852173c81ca0ed6503967c89eccbafa45b34a",
        "f5e99e9ecf2b0171edac78e3b5d393b2926463a85600dbc2e1a0c9c29a3269f6",
    ),
}


class TestMemoryPublicInterface(unittest.TestCase):
    def test_unknown_attribute_raises_exact_module_error(self):
        import emperor.memory as memory

        with self.assertRaisesRegex(
            AttributeError,
            "^module 'emperor.memory' has no attribute 'missing_memory_export'$",
        ):
            self.assertIsNone(memory.missing_memory_export)

    def test_root_interface_eagerly_exports_configuration_protocol_and_callback(self):
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
        expected_heavy_modules = dict.fromkeys(result["heavy_modules"], False)
        expected_heavy_modules["emperor.memory._monitoring"] = True
        self.assertEqual(result["heavy_modules"], expected_heavy_modules)
        expected_private_exports = dict.fromkeys(result["private_exports"], False)
        expected_private_exports["MemoryMonitorCallback"] = True
        self.assertEqual(result["private_exports"], expected_private_exports)
        self.assertEqual(
            result["runtime_loaded"],
            {"emperor.layers": False, "lightning": True, "torch": True},
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
        self.assertIs(memory.MemoryMonitorCallback, monitoring.MemoryMonitorCallback)

    def test_removed_implementations_cannot_be_imported_from_the_root_interface(self):
        removed_names = (
            "AttentionDynamicMemory",
            "DynamicMemoryAbstract",
            "ElementWiseWeightedDynamicMemory",
            "GatedResidualDynamicMemory",
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

    def test_each_concrete_config_builds_its_private_implementation(self):
        for config_cls, implementation_cls in MEMORY_CASES:
            with self.subTest(config=config_cls.__name__):
                config = make_memory_config(
                    config_cls=config_cls,
                    input_dim=4,
                    output_dim=4,
                )

                self.assertIs(config.registry_owner(), implementation_cls)
                self.assertIs(type(config.build()), implementation_cls)

    def test_exact_children_state_keys_and_strict_load_are_preserved(self):
        parameter_suffixes = (
            "layers.0.model.weight_params",
            "layers.0.model.bias_params",
        )

        for config_cls, _ in MEMORY_CASES:
            for test_time_training in (False, True):
                with self.subTest(
                    config=config_cls.__name__,
                    test_time_training=test_time_training,
                ):
                    ttt_kwargs = (
                        {
                            "test_time_training_learning_rate": 0.1,
                            "test_time_training_num_inner_steps": 1,
                        }
                        if test_time_training
                        else {}
                    )
                    config = make_memory_config(
                        config_cls=config_cls,
                        input_dim=4,
                        output_dim=4,
                        **ttt_kwargs,
                    )
                    model = config.build()
                    children = VARIANT_CHILDREN[config_cls]
                    if test_time_training:
                        children = (children[0], "memory_decoder", *children[1:])

                    expected_keys = tuple(
                        f"{child}.{suffix}"
                        for child in children
                        for suffix in parameter_suffixes
                    )
                    self.assertEqual(tuple(model._modules), children)
                    self.assertEqual(tuple(model.state_dict()), expected_keys)

                    restored = make_memory_config(
                        config_cls=config_cls,
                        input_dim=4,
                        output_dim=4,
                        **ttt_kwargs,
                    ).build()
                    load_result = restored.load_state_dict(
                        model.state_dict(), strict=True
                    )
                    self.assertEqual(load_result.missing_keys, [])
                    self.assertEqual(load_result.unexpected_keys, [])
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[name], value)

    def test_seeded_construction_rng_contract_is_preserved(self):
        for config_cls, _ in MEMORY_CASES:
            for test_time_training in (False, True):
                with self.subTest(
                    config=config_cls.__name__,
                    test_time_training=test_time_training,
                ):
                    ttt_kwargs = (
                        {
                            "test_time_training_learning_rate": 0.1,
                            "test_time_training_num_inner_steps": 1,
                        }
                        if test_time_training
                        else {}
                    )
                    with torch.random.fork_rng():
                        torch.manual_seed(7)
                        make_memory_config(
                            config_cls=config_cls,
                            input_dim=4,
                            output_dim=4,
                            **ttt_kwargs,
                        ).build()
                        digest = hashlib.sha256(
                            torch.random.get_rng_state().numpy().tobytes()
                        ).hexdigest()

                    self.assertEqual(
                        digest,
                        RNG_DIGESTS[config_cls][int(test_time_training)],
                    )

    def test_invalid_config_is_rejected_before_rng_consumption(self):
        for config_cls, _ in MEMORY_CASES:
            with self.subTest(config=config_cls.__name__):
                config = make_memory_config(
                    config_cls=config_cls,
                    input_dim=4,
                    output_dim=4,
                )
                config.input_dim = 0

                with torch.random.fork_rng():
                    torch.manual_seed(17)
                    expected_next_values = torch.randn(8)

                    torch.manual_seed(17)
                    with self.assertRaises(ValueError):
                        config.build()
                    actual_next_values = torch.randn(8)

                torch.testing.assert_close(actual_next_values, expected_next_values)


if __name__ == "__main__":
    unittest.main()

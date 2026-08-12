from __future__ import annotations

import re
import unittest
from dataclasses import replace
from importlib import import_module
from types import ModuleType

_VIT_VARIANTS = (
    ("linear", "linear_builder_kwargs_from_flat"),
    ("linear_adaptive", "linear_adaptive_builder_kwargs_from_flat"),
    ("expert_linear", "expert_linear_builder_kwargs_from_flat"),
    (
        "expert_linear_adaptive",
        "expert_linear_adaptive_builder_kwargs_from_flat",
    ),
)


class _RecordingConfig(ModuleType):
    def __init__(self, source: ModuleType) -> None:
        super().__init__(f"recording_{source.__name__}")
        self._source = source
        self.reads: set[str] = set()

    def __getattr__(self, name: str) -> object:
        self.reads.add(name)
        return getattr(self._source, name)


class VitBuilderAdapterContracts(unittest.TestCase):
    def test_flat_values_override_supplied_groups_without_mutating_the_input(
        self,
    ) -> None:
        sentinel = object()
        for variant, adapter_name in _VIT_VARIANTS:
            with self.subTest(variant=variant):
                adapter_module = import_module(f"models.vit.{variant}._builder_adapter")
                config_module = import_module(f"models.vit.{variant}.config")
                defaults_module = import_module(
                    f"models.vit.{variant}._config_defaults"
                )
                supplied_patch = replace(
                    defaults_module.vit_patch_options(config_module),
                    patch_size=3,
                )
                flat_values = {
                    "patch_options": supplied_patch,
                    "image_patch_size": 5,
                    "unrelated": sentinel,
                }

                resolved = getattr(adapter_module, adapter_name)(
                    flat_values,
                    config_module,
                )

                self.assertEqual(resolved["patch_options"].patch_size, 5)
                self.assertNotIn("image_patch_size", resolved)
                self.assertIs(resolved["unrelated"], sentinel)
                self.assertEqual(
                    flat_values,
                    {
                        "patch_options": supplied_patch,
                        "image_patch_size": 5,
                        "unrelated": sentinel,
                    },
                )

    def test_each_staged_control_path_consumes_flat_keys_and_preserves_leftovers(
        self,
    ) -> None:
        flat_keys = {
            "image_patch_size",
            "stack_num_layers",
            "attn_gate_stack_hidden_dim",
            "ff_recurrent_max_steps",
        }
        for variant, _ in _VIT_VARIANTS:
            with self.subTest(variant=variant):
                adapter_module = import_module(f"models.vit.{variant}._builder_adapter")
                config_module = import_module(f"models.vit.{variant}.config")
                values = {
                    "image_patch_size": 5,
                    "stack_num_layers": 2,
                    "attn_gate_stack_hidden_dim": 21,
                    "ff_recurrent_max_steps": 4,
                    "unrelated": "preserved",
                }
                consumed: set[str] = set()

                resolved = adapter_module._vit_builder_kwargs(
                    values,
                    config_module,
                    consumed,
                )

                self.assertTrue(flat_keys.issubset(consumed))
                self.assertNotIn("unrelated", consumed)
                self.assertEqual(resolved["patch_options"].patch_size, 5)
                self.assertEqual(resolved["stack_options"].num_layers, 2)
                self.assertEqual(
                    resolved[
                        "attention_projection_layer_controller_options"
                    ].gate_stack_source.hidden_dim,
                    21,
                )
                self.assertEqual(
                    resolved[
                        "feed_forward_recurrent_controller_options"
                    ].recurrent_max_steps,
                    4,
                )

    def test_supplied_structured_options_do_not_read_package_defaults(self) -> None:
        for variant, _ in _VIT_VARIANTS:
            with self.subTest(variant=variant):
                adapter_module = import_module(f"models.vit.{variant}._builder_adapter")
                config_module = import_module(f"models.vit.{variant}.config")
                defaults_module = import_module(
                    f"models.vit.{variant}._config_defaults"
                )
                supplied_patch = defaults_module.vit_patch_options(config_module)
                consumed: set[str] = set()

                resolved = adapter_module._vit_builder_kwargs(
                    {"patch_options": supplied_patch},
                    ModuleType("partial_config"),
                    consumed,
                )

                self.assertEqual(resolved, {"patch_options": supplied_patch})
                self.assertEqual(consumed, {"patch_options"})

    def test_unrelated_input_does_not_read_package_defaults(self) -> None:
        for variant, _ in _VIT_VARIANTS:
            with self.subTest(variant=variant):
                adapter_module = import_module(f"models.vit.{variant}._builder_adapter")
                consumed: set[str] = set()

                resolved = adapter_module._vit_builder_kwargs(
                    {"unrelated": object()},
                    ModuleType("partial_config"),
                    consumed,
                )

                self.assertEqual(resolved, {})
                self.assertEqual(consumed, set())

    def test_single_attention_override_reads_only_attention_role_defaults(
        self,
    ) -> None:
        for variant, _ in _VIT_VARIANTS:
            with self.subTest(variant=variant):
                adapter_module = import_module(f"models.vit.{variant}._builder_adapter")
                source_config = import_module(f"models.vit.{variant}.config")
                recording_config = _RecordingConfig(source_config)
                consumed: set[str] = set()

                adapter_module._vit_builder_kwargs(
                    {"attn_gate_stack_hidden_dim": 21},
                    recording_config,
                    consumed,
                )

                self.assertIn("attn_gate_stack_hidden_dim", consumed)
                self.assertTrue(recording_config.reads)
                self.assertTrue(
                    all(
                        attribute.startswith("ATTN_")
                        for attribute in recording_config.reads
                    ),
                    recording_config.reads,
                )

    def test_runtime_default_boundaries_keep_exact_unknown_key_errors(self) -> None:
        for variant, _ in _VIT_VARIANTS:
            runtime_defaults = import_module(f"models.vit.{variant}.runtime_defaults")
            message = (
                f"models.vit.{variant}: unknown Runtime Defaults field(s): "
                "'unknown_group_options'"
            )
            with (
                self.subTest(variant=variant),
                self.assertRaisesRegex(ValueError, f"^{re.escape(message)}$"),
            ):
                runtime_defaults.runtime_from_flat({"unknown_group_options": object()})


if __name__ == "__main__":
    unittest.main()

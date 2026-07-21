import dataclasses
import importlib
import unittest
from pathlib import Path

import emperor.layers as layers
import emperor.transformer as transformer

TRANSFORMER_EXPORTS = (
    "Transformer",
    "TransformerConfig",
    "TransformerEncoderLayer",
    "TransformerEncoderBlockLayer",
    "TransformerDecoderBlockLayer",
    "TransformerDecoderLayerState",
    "TransformerDecoderLayer",
    "TransformerEncoderLayerConfig",
    "TransformerEncoderBlockLayerConfig",
    "TransformerDecoderBlockLayerConfig",
    "TransformerDecoderLayerConfig",
    "FeedForward",
    "FeedForwardConfig",
)

REMOVED_EXPORTS = (
    "TransformerEncoderStack",
    "TransformerDecoderStack",
    "TransformerEncoderStackConfig",
    "TransformerDecoderStackConfig",
    "ControllerStackOptions",
    "DynamicMemoryOptions",
    "LayerControllerOptions",
    "RecurrentControllerOptions",
    "SubmoduleStackOptions",
    "SubmoduleStackSource",
    "TransformerAttentionOptions",
    "TransformerFeedForwardOptions",
    "TransformerPathOptions",
    "TransformerStackOptions",
    "attention_options_from_config",
    "configure_transformer_submodule",
    "expand_transformer_path_locks",
    "feed_forward_options_from_config",
    "resolve_controller_stack_options",
    "resolve_transformer_path_options",
)


class TestTransformerInterface(unittest.TestCase):
    def test_exact_component_exports(self):
        self.assertEqual(transformer.__all__, TRANSFORMER_EXPORTS)
        for name in TRANSFORMER_EXPORTS:
            self.assertIsNotNone(getattr(transformer, name))

    def test_removed_construction_exports_are_unavailable(self):
        for name in REMOVED_EXPORTS:
            with self.subTest(name=name):
                with self.assertRaises(AttributeError):
                    getattr(transformer, name)

    def test_lazy_exports_resolve_to_component_modules(self):
        expected_modules = {
            "Transformer": "emperor.transformer._model",
            "TransformerConfig": "emperor.transformer._config",
            "TransformerEncoderLayer": "emperor.transformer._layers",
            "TransformerEncoderBlockLayer": "emperor.transformer._layers",
            "TransformerDecoderBlockLayer": "emperor.transformer._layers",
            "TransformerDecoderLayerState": "emperor.transformer._state",
            "TransformerDecoderLayer": "emperor.transformer._layers",
            "TransformerEncoderLayerConfig": "emperor.transformer._config",
            "TransformerEncoderBlockLayerConfig": "emperor.transformer._config",
            "TransformerDecoderBlockLayerConfig": "emperor.transformer._config",
            "TransformerDecoderLayerConfig": "emperor.transformer._config",
            "FeedForward": "emperor.transformer._feed_forward",
            "FeedForwardConfig": "emperor.transformer._feed_forward",
        }
        for name, module_name in expected_modules.items():
            with self.subTest(name=name):
                exported = getattr(transformer, name)
                self.assertIs(
                    exported, getattr(importlib.import_module(module_name), name)
                )

    def test_layer_configs_do_not_own_causality(self):
        for config_type in (
            transformer.TransformerEncoderLayerConfig,
            transformer.TransformerDecoderLayerConfig,
        ):
            with self.subTest(config_type=config_type.__name__):
                field_names = {field.name for field in dataclasses.fields(config_type)}
                self.assertNotIn("causal_attention_mask_flag", field_names)

    def test_mirrored_stack_is_public_layer_component(self):
        self.assertIn("MirroredLayerStack", layers.__all__)
        self.assertIn("MirroredLayerStackConfig", layers.__all__)
        self.assertEqual(
            layers.MirroredLayerStack.__module__,
            "emperor.layers._mirrored",
        )
        self.assertEqual(
            layers.MirroredLayerStackConfig.__module__,
            "emperor.layers._config",
        )

    def test_removed_modules_do_not_exist(self):
        package_root = Path(transformer.__file__).resolve().parent
        for relative_path in (
            "_stacks.py",
            "_submodule_configuration.py",
            "_options/__init__.py",
            "_options/config_adapter.py",
            "_options/overrides.py",
            "_options/records.py",
        ):
            with self.subTest(relative_path=relative_path):
                self.assertFalse((package_root / relative_path).exists())


if __name__ == "__main__":
    unittest.main()

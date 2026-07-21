import dataclasses
import importlib
import unittest
from pathlib import Path

import torch

from emperor.attention import AttentionLayerState
from emperor.layers import Layer, LayerStack, RecurrentLayer
from emperor.transformer import (
    TransformerDecoderLayer,
    TransformerDecoderLayerState,
    TransformerEncoderLayer,
)
from unit.test_transformer import decoder_stack, encoder_stack, recurrent

MODEL_PACKAGES = (
    "linear",
    "linear_adaptive",
    "expert_linear",
    "expert_linear_adaptive",
)


class TestTransformerLayerComposition(unittest.TestCase):
    def test_encoder_forward_is_composed_from_layer_wrappers(self):
        config = encoder_stack(num_layers=1).layer_config.layer_model_config
        model = TransformerEncoderLayer(config).eval()
        source = torch.randn(2, 4, config.embedding_dim)
        padding_mask = torch.tensor(
            [[False, False, True, True], [False, False, False, True]]
        )
        attention_mask = torch.zeros(4, 4, dtype=torch.bool)

        manual_state = AttentionLayerState(
            hidden=source.clone(),
            key_padding_mask=padding_mask,
            attention_mask=attention_mask,
        )
        with torch.no_grad():
            manual_state = model.self_attention_layer(manual_state)
            manual_state = model.feed_forward_layer(manual_state)
            manual_loss = (
                manual_state.loss
                if manual_state.loss is not None
                else source.new_zeros(())
            )
            actual, actual_loss = model(
                source,
                source_key_padding_mask=padding_mask,
                attention_mask=attention_mask,
            )

        self.assertIsInstance(model.self_attention_layer, Layer)
        self.assertIsInstance(model.feed_forward_layer, Layer)
        torch.testing.assert_close(actual, manual_state.hidden)
        torch.testing.assert_close(actual_loss, manual_loss)

    def test_decoder_forward_is_composed_from_layer_wrappers(self):
        config = decoder_stack(num_layers=1).layer_config.layer_model_config
        model = TransformerDecoderLayer(config).eval()
        target = torch.randn(2, 3, config.embedding_dim)
        encoder_output = torch.randn(2, 5, config.embedding_dim)
        target_padding_mask = torch.tensor(
            [[False, False, True], [False, False, False]]
        )
        encoder_padding_mask = torch.tensor(
            [[False, False, False, True, True], [False, False, False, False, True]]
        )
        target_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        cross_mask = torch.zeros(3, 5, dtype=torch.bool)

        manual_state = TransformerDecoderLayerState(
            hidden=target.clone(),
            target_key_padding_mask=target_padding_mask,
            target_attention_mask=target_mask,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            cross_attention_mask=cross_mask,
        )
        with torch.no_grad():
            manual_state = model.self_attention_layer(manual_state)
            manual_state = model.cross_attention_layer(manual_state)
            manual_state = model.feed_forward_layer(manual_state)
            manual_loss = (
                manual_state.loss
                if manual_state.loss is not None
                else target.new_zeros(())
            )
            actual, actual_loss = model(
                target,
                encoder_output=encoder_output,
                key_padding_mask=target_padding_mask,
                encoder_padding_mask=encoder_padding_mask,
                attention_mask=target_mask,
                encoder_attention_mask=cross_mask,
            )

        self.assertIsInstance(model.self_attention_layer, Layer)
        self.assertIsInstance(model.cross_attention_layer, Layer)
        self.assertIsInstance(model.feed_forward_layer, Layer)
        torch.testing.assert_close(actual, manual_state.hidden)
        torch.testing.assert_close(actual_loss, manual_loss)

    def test_generic_stack_and_recurrent_components_own_transformer_blocks(self):
        stack = encoder_stack().build()
        recurrent_model = recurrent(encoder_stack(num_layers=1)).build()

        self.assertIsInstance(stack, LayerStack)
        self.assertIsInstance(recurrent_model, RecurrentLayer)
        self.assertTrue(all(isinstance(layer, Layer) for layer in stack))
        self.assertIsInstance(recurrent_model.block_model, LayerStack)
        self.assertTrue(
            all(isinstance(layer, Layer) for layer in recurrent_model.block_model)
        )


class TestTransformerModelPackageOwnership(unittest.TestCase):
    def package_modules(self, package_name: str):
        root = f"models.transformer.{package_name}"
        return (
            importlib.import_module(f"{root}.runtime_options"),
            importlib.import_module(f"{root}.runtime_defaults"),
            importlib.import_module(f"{root}.presets"),
        )

    def test_runtime_records_are_package_local_frozen_dataclasses(self):
        expected_attention_fields = {
            "num_heads",
            "add_key_value_bias_flag",
            "zero_attention_flag",
            "stack_options",
            "layer_controller_options",
            "dynamic_memory_options",
            "recurrent_controller_options",
        }
        expected_feed_forward_fields = {
            "stack_options",
            "layer_controller_options",
            "dynamic_memory_options",
            "recurrent_controller_options",
        }
        for package_name in MODEL_PACKAGES:
            options, _defaults, _presets = self.package_modules(package_name)
            with self.subTest(package_name=package_name):
                attention_type = options.TransformerAttentionOptions
                feed_forward_type = options.TransformerFeedForwardOptions
                self.assertTrue(attention_type.__dataclass_params__.frozen)
                self.assertTrue(feed_forward_type.__dataclass_params__.frozen)
                self.assertEqual(
                    {field.name for field in dataclasses.fields(attention_type)},
                    expected_attention_fields,
                )
                self.assertEqual(
                    {field.name for field in dataclasses.fields(feed_forward_type)},
                    expected_feed_forward_fields,
                )
                self.assertEqual(
                    attention_type.__module__,
                    f"models.transformer.{package_name}.runtime_options",
                )
                self.assertEqual(
                    feed_forward_type.__module__,
                    f"models.transformer.{package_name}.runtime_options",
                )

    def test_stack_width_depth_and_bias_have_one_representation(self):
        for package_name in MODEL_PACKAGES:
            options, _defaults, _presets = self.package_modules(package_name)
            attention = options.TransformerAttentionOptions()
            feed_forward = options.TransformerFeedForwardOptions()
            with self.subTest(package_name=package_name):
                for path in (attention, feed_forward):
                    self.assertFalse(hasattr(path, "hidden_dim"))
                    self.assertFalse(hasattr(path, "num_layers"))
                    self.assertFalse(hasattr(path, "bias_flag"))
                self.assertFalse(hasattr(attention, "projection_bias_flag"))
                self.assertEqual(feed_forward.stack_options.hidden_dim, 512)
                self.assertEqual(feed_forward.stack_options.num_layers, 2)

    def test_flat_overrides_resolve_only_canonical_names(self):
        for package_name in MODEL_PACKAGES:
            _options, defaults, _presets = self.package_modules(package_name)
            runtime = defaults.runtime_from_flat(
                {
                    "ff_stack_hidden_dim": 96,
                    "ff_num_layers": 3,
                    "attn_bias_flag": False,
                }
            )
            with self.subTest(package_name=package_name):
                self.assertEqual(
                    runtime.encoder_feed_forward_options.stack_options.hidden_dim,
                    96,
                )
                self.assertEqual(
                    runtime.decoder_feed_forward_options.stack_options.num_layers,
                    3,
                )
                self.assertFalse(
                    runtime.encoder_attention_options.stack_options.bias_flag
                )
                for legacy_name in (
                    "feed_forward_hidden_dim",
                    "feed_forward_num_layers",
                    "attn_projection_bias_flag",
                ):
                    with self.assertRaises(TypeError):
                        defaults.runtime_from_flat({legacy_name: 1})

    def test_runtime_translation_and_lock_expansion_live_in_each_package(self):
        for package_name in MODEL_PACKAGES:
            _options, defaults, presets = self.package_modules(package_name)
            runtime = defaults.runtime_from_config()
            expanded = presets.expand_transformer_path_locks(
                {"attn_bias_flag": True, "ff_num_layers": 3}
            )
            with self.subTest(package_name=package_name):
                self.assertEqual(runtime.encoder_options.num_layers, 3)
                self.assertTrue(expanded["encoder_attn_bias_flag"])
                self.assertTrue(expanded["decoder_self_attn_bias_flag"])
                self.assertTrue(expanded["decoder_cross_attn_bias_flag"])
                self.assertEqual(expanded["encoder_ff_num_layers"], 3)
                self.assertEqual(expanded["decoder_ff_num_layers"], 3)

    def test_package_construction_does_not_import_emperor_runtime_machinery(self):
        project_root = Path(__file__).resolve().parents[2]
        for package_name in MODEL_PACKAGES:
            package_root = (
                project_root / "src" / "models" / "transformer" / package_name
            )
            with self.subTest(package_name=package_name):
                runtime_source = (package_root / "runtime_defaults.py").read_text()
                building_source = (package_root / "_building.py").read_text()
                local_source = (package_root / "_transformer_submodule.py").read_text()
                self.assertNotIn("emperor.transformer._options", runtime_source)
                self.assertNotIn("configure_transformer_submodule", runtime_source)
                self.assertIn("._transformer_submodule", building_source)
                self.assertIn("def configure_transformer_submodule", local_source)


if __name__ == "__main__":
    unittest.main()

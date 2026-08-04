import dataclasses
import importlib
import re
import unittest
from pathlib import Path

import torch

from emperor.attention import AttentionLayerState
from emperor.config import ConfigBase
from emperor.layers import Layer, LayerStack, LayerState, RecurrentLayer
from emperor.nn import Module
from emperor.transformer import (
    TransformerDecoderLayer,
    TransformerDecoderLayerState,
    TransformerEncoderBlockLayer,
    TransformerEncoderLayer,
)
from unit.test_transformer import decoder_stack, encoder_stack, recurrent

MODEL_PACKAGES = (
    "linear",
    "linear_adaptive",
    "expert_linear",
    "expert_linear_adaptive",
)


@dataclasses.dataclass
class _RecordingEncoderConfig(ConfigBase):
    def _registry_owner(self) -> type:
        return _RecordingEncoder


class _RecordingEncoder(Module):
    def __init__(
        self,
        cfg: _RecordingEncoderConfig,
        overrides: _RecordingEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.calls: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

    def forward(
        self,
        source_token_embeddings: torch.Tensor,
        source_key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((source_key_padding_mask, attention_mask))
        return source_token_embeddings + 1.0, source_token_embeddings.new_tensor(0.25)


class TestTransformerLayerComposition(unittest.TestCase):
    def test_attention_sublayers_reject_context_free_states(self):
        encoder_config = encoder_stack(num_layers=1).layer_config.layer_model_config
        decoder_config = decoder_stack(num_layers=1).layer_config.layer_model_config
        encoder = TransformerEncoderLayer(encoder_config)
        decoder = TransformerDecoderLayer(decoder_config)
        context_free_state = LayerState(hidden=torch.randn(2, 3, 8))

        cases = (
            (
                encoder.self_attention_layer,
                "Encoder self-attention requires an AttentionLayerState.",
            ),
            (
                decoder.self_attention_layer,
                "Decoder self-attention requires a TransformerDecoderLayerState.",
            ),
            (
                decoder.cross_attention_layer,
                "Decoder cross-attention requires a TransformerDecoderLayerState.",
            ),
        )
        for layer, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, rf"^{re.escape(message)}$"):
                    layer(context_free_state)

    def test_encoder_block_forwards_only_attention_state_masks(self):
        block_config = encoder_stack(embedding_dim=4, num_layers=1).layer_config
        block_config.layer_model_config = _RecordingEncoderConfig()
        block = block_config.build()
        self.assertIsInstance(block, TransformerEncoderBlockLayer)

        plain_state = LayerState(hidden=torch.zeros(2, 3, 4))
        plain_output = block(plain_state)

        self.assertIs(plain_output, plain_state)
        torch.testing.assert_close(plain_output.hidden, torch.ones(2, 3, 4))
        torch.testing.assert_close(plain_output.loss, torch.tensor(0.25))
        self.assertEqual(block.model.calls, [(None, None)])

        block.model.calls.clear()
        padding_mask = torch.tensor(
            [[False, False, True], [False, True, True]],
        )
        attention_mask = torch.zeros(3, 3, dtype=torch.bool)
        attention_state = AttentionLayerState(
            hidden=torch.zeros(2, 3, 4),
            loss=torch.tensor(1.0),
            key_padding_mask=padding_mask,
            attention_mask=attention_mask,
        )
        attention_output = block(attention_state)

        self.assertIs(attention_output, attention_state)
        self.assertIs(block.model.calls[0][0], padding_mask)
        self.assertIs(block.model.calls[0][1], attention_mask)
        torch.testing.assert_close(attention_output.loss, torch.tensor(1.25))

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
                    with self.assertRaisesRegex(ValueError, "unknown Runtime Defaults"):
                        defaults.runtime_from_flat({legacy_name: 1})

    def test_runtime_translation_and_exact_preset_locks_live_in_each_package(self):
        for package_name in MODEL_PACKAGES:
            _options, defaults, presets = self.package_modules(package_name)
            runtime = defaults.runtime_from_config()
            package_presets = presets.ExperimentPresets()
            attention_bias_locks = package_presets.locks_for_preset(
                presets.ExperimentPreset.ATTENTION_BIAS
            )
            pre_norm_locks = package_presets.locks_for_preset(
                presets.ExperimentPreset.PRE_NORM
            )
            with self.subTest(package_name=package_name):
                self.assertEqual(runtime.encoder_options.num_layers, 3)
                self.assertEqual(
                    set(attention_bias_locks),
                    {"attn_bias_flag", "attn_add_key_value_bias_flag"},
                )
                self.assertEqual(
                    set(pre_norm_locks),
                    {
                        "encoder_layer_norm_position",
                        "decoder_layer_norm_position",
                    },
                )
                self.assertFalse(hasattr(presets, "expand_transformer_path_locks"))

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

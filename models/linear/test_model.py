import unittest

import torch

import models.linear.config as config

from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.experiments.base import RandomSearch
from models.linear.model import Model
from models.linear.config_builder import LinearConfigBuilder
from models.linear.presets import ExperimentOptions, ExperimentPresets


class TestLinearModel(unittest.TestCase):
    def test_all_options_forward_one_batch_per_dataset(self):
        batch_size = 4
        presets = ExperimentPresets()

        for dataset in config.DATASET_OPTIONS:
            for option in ExperimentOptions:
                message = f"dataset={dataset.__name__}, option={option.name}"
                with self.subTest(msg=message):
                    cfg = presets.get_config(option, dataset)[0]
                    model = Model(cfg)
                    X = self._fake_batch(dataset, batch_size)

                    output = model(X)
                    logits = output[0] if isinstance(output, tuple) else output

                    self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )

    def test_preset_accepts_search_flags(self):
        configs = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
            config.DATASET_OPTIONS[0],
            RandomSearch(num_samples=2),
        )

        self.assertEqual(len(configs), 2)

    def test_non_baseline_options_accept_search_flags(self):
        presets = ExperimentPresets()
        searchable_options = [
            ExperimentOptions.GATING,
            ExperimentOptions.HALTING,
            ExperimentOptions.GATING_HALTING,
        ]

        for option in searchable_options:
            with self.subTest(option=option.name):
                configs = presets.get_config(
                    option,
                    config.DATASET_OPTIONS[0],
                    RandomSearch(num_samples=2),
                )

                self.assertEqual(len(configs), 2)

    def test_gate_config_uses_builder_overrides(self):
        cfg = LinearConfigBuilder(
            stack_gate_flag=True,
            gate_hidden_dim=32,
            gate_layer_norm_position=LayerNormPositionOptions.AFTER,
            gate_stack_num_layers=3,
            gate_stack_activation=ActivationOptions.SILU,
            gate_stack_residual_flag=False,
            gate_stack_dropout_probability=0.1,
            gate_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            gate_stack_apply_output_pipeline_flag=True,
            gate_bias_flag=False,
        ).build()

        gate_cfg = cfg.experiment_config.model_config.layer_config.gate_config

        self.assertEqual(gate_cfg.hidden_dim, 32)
        self.assertEqual(gate_cfg.num_layers, 3)
        self.assertEqual(gate_cfg.last_layer_bias_option, LastLayerBiasOptions.DISABLED)
        self.assertTrue(gate_cfg.apply_output_pipeline_flag)
        self.assertEqual(
            gate_cfg.layer_config.layer_norm_position, LayerNormPositionOptions.AFTER
        )
        self.assertEqual(gate_cfg.layer_config.activation, ActivationOptions.SILU)
        self.assertFalse(gate_cfg.layer_config.residual_flag)
        self.assertEqual(gate_cfg.layer_config.dropout_probability, 0.1)
        self.assertFalse(gate_cfg.layer_config.layer_model_config.bias_flag)

    def test_halting_config_uses_builder_overrides(self):
        cfg = LinearConfigBuilder(
            output_dim=9,
            stack_halting_flag=True,
            halting_threshold=0.5,
            halting_dropout=0.2,
            halting_hidden_dim=48,
            halting_output_dim=4,
            halting_layer_norm_position=LayerNormPositionOptions.BEFORE,
            halting_stack_num_layers=5,
            halting_stack_activation=ActivationOptions.MISH,
            halting_stack_residual_flag=False,
            halting_stack_dropout_probability=0.3,
            halting_stack_last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            halting_stack_apply_output_pipeline_flag=True,
            halting_bias_flag=False,
        ).build()

        halting_cfg = cfg.experiment_config.model_config.layer_config.halting_config
        halting_stack_cfg = halting_cfg.halting_gate_config

        self.assertEqual(halting_cfg.threshold, 0.5)
        self.assertEqual(halting_cfg.halting_dropout, 0.2)
        self.assertEqual(halting_stack_cfg.hidden_dim, 48)
        self.assertEqual(halting_stack_cfg.output_dim, 4)
        self.assertEqual(halting_stack_cfg.num_layers, 5)
        self.assertEqual(
            halting_stack_cfg.last_layer_bias_option, LastLayerBiasOptions.DEFAULT
        )
        self.assertTrue(halting_stack_cfg.apply_output_pipeline_flag)
        self.assertEqual(
            halting_stack_cfg.layer_config.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertEqual(
            halting_stack_cfg.layer_config.activation, ActivationOptions.MISH
        )
        self.assertFalse(halting_stack_cfg.layer_config.residual_flag)
        self.assertEqual(halting_stack_cfg.layer_config.dropout_probability, 0.3)
        self.assertFalse(halting_stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_search_keys_restrict_sweep_to_subset_of_axes(self):
        configs = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
            config.DATASET_OPTIONS[0],
            RandomSearch(num_samples=20),
            search_keys=["hidden_dim"],
        )

        learning_rates = {cfg.learning_rate for cfg in configs}
        hidden_dims = {cfg.hidden_dim for cfg in configs}

        self.assertEqual(len(learning_rates), 1)
        self.assertEqual(hidden_dims, set(config.SEARCH_SPACE_HIDDEN_DIM))

    def test_search_keys_unknown_axis_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ExperimentPresets().get_config(
                ExperimentOptions.BASELINE,
                config.DATASET_OPTIONS[0],
                RandomSearch(num_samples=2),
                search_keys=["bogus_axis"],
            )

        self.assertIn("Unknown", str(ctx.exception))

    def test_halting_hidden_dim_falls_back_to_output_dim(self):
        cfg = LinearConfigBuilder(
            output_dim=11,
            stack_halting_flag=True,
            halting_hidden_dim=0,
        ).build()

        halting_cfg = cfg.experiment_config.model_config.layer_config.halting_config

        self.assertEqual(halting_cfg.halting_gate_config.hidden_dim, 11)


if __name__ == "__main__":
    unittest.main()

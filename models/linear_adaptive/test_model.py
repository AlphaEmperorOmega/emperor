import unittest

import torch

import models.linear_adaptive.config as config

from emperor.augmentations.adaptive_parameters.core.bias import (
    GeneratorDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    CombinedDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    MaskDimensionOptions,
)
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.experiments.base import RandomSearch
from emperor.linears.core.config import AdaptiveLinearLayerConfig
from models.linear_adaptive.config_builder import LinearAdaptiveConfigBuilder
from models.linear_adaptive.model import Model
from models.linear_adaptive.presets import ExperimentOptions, ExperimentPresets


class TestAdaptiveLinearModel(unittest.TestCase):
    def test_all_options_forward_one_batch_per_dataset(self):
        batch_size = 2
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

    def test_preset_builds_adaptive_linear_layer_config(self):
        cfg = ExperimentPresets()._preset(input_dim=8, hidden_dim=16, output_dim=4)
        layer_model_config = (
            cfg.experiment_config.model_config.layer_config.layer_model_config
        )

        self.assertIsInstance(layer_model_config, AdaptiveLinearLayerConfig)
        self.assertEqual(cfg.input_dim, 8)
        self.assertEqual(cfg.hidden_dim, 16)
        self.assertEqual(cfg.output_dim, 4)

    def test_config_builder_builds_model_config(self):
        cfg = LinearAdaptiveConfigBuilder(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
        ).build()

        self.assertEqual(cfg.input_dim, 8)
        self.assertEqual(cfg.hidden_dim, 16)
        self.assertEqual(cfg.output_dim, 4)
        self.assertIsInstance(
            cfg.experiment_config.model_config.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )

    def test_config_search_space_builds_configs(self):
        configs = ExperimentPresets().get_config(
            ExperimentOptions.CONFIG,
            config.DATASET_OPTIONS[0],
            RandomSearch(num_samples=2),
        )

        self.assertTrue(len(configs) > 1)
        self.assertTrue(
            all(
                isinstance(
                    cfg.experiment_config.model_config.layer_config.layer_model_config,
                    AdaptiveLinearLayerConfig,
                )
                for cfg in configs
            )
        )

    def test_adaptive_sub_configs_match_options(self):
        cfg = ExperimentPresets()._preset(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            weight_flag=True,
            diagonal_option=CombinedDynamicDiagonalConfig,
            bias_option=GeneratorDynamicBiasConfig,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        augmentation_config = (
            cfg.experiment_config.model_config.layer_config.layer_model_config
            .adaptive_augmentation_config
        )

        self.assertIsInstance(
            augmentation_config.weight_config, DualModelDynamicWeightConfig
        )
        self.assertIsInstance(
            augmentation_config.diagonal_config, CombinedDynamicDiagonalConfig
        )
        self.assertIsInstance(
            augmentation_config.bias_config, GeneratorDynamicBiasConfig
        )
        self.assertIsInstance(
            augmentation_config.mask_config, WeightInformedScoreAxisMaskConfig
        )
        self.assertEqual(
            augmentation_config.mask_config.mask_dimension_option,
            MaskDimensionOptions.ROW,
        )

    def test_gate_config_uses_builder_overrides(self):
        cfg = LinearAdaptiveConfigBuilder(
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
        cfg = LinearAdaptiveConfigBuilder(
            stack_halting_flag=True,
            halting_threshold=0.5,
            halting_dropout=0.2,
            halting_gate_hidden_dim=48,
            halting_gate_output_dim=2,
            halting_gate_layer_norm_position=LayerNormPositionOptions.BEFORE,
            halting_gate_stack_num_layers=5,
            halting_gate_stack_activation=ActivationOptions.MISH,
            halting_gate_stack_residual_flag=False,
            halting_gate_stack_dropout_probability=0.3,
            halting_gate_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            halting_gate_stack_apply_output_pipeline_flag=True,
            halting_gate_bias_flag=False,
        ).build()

        halting_cfg = cfg.experiment_config.model_config.layer_config.halting_config
        halting_gate_cfg = halting_cfg.halting_gate_config

        self.assertEqual(halting_cfg.threshold, 0.5)
        self.assertEqual(halting_cfg.halting_dropout, 0.2)
        self.assertEqual(halting_gate_cfg.hidden_dim, 48)
        self.assertEqual(halting_gate_cfg.output_dim, 2)
        self.assertEqual(halting_gate_cfg.num_layers, 5)
        self.assertEqual(
            halting_gate_cfg.last_layer_bias_option, LastLayerBiasOptions.DISABLED
        )
        self.assertTrue(halting_gate_cfg.apply_output_pipeline_flag)
        self.assertEqual(
            halting_gate_cfg.layer_config.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertEqual(
            halting_gate_cfg.layer_config.activation, ActivationOptions.MISH
        )
        self.assertFalse(halting_gate_cfg.layer_config.residual_flag)
        self.assertEqual(halting_gate_cfg.layer_config.dropout_probability, 0.3)
        self.assertFalse(halting_gate_cfg.layer_config.layer_model_config.bias_flag)

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )


if __name__ == "__main__":
    unittest.main()

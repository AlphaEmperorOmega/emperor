import unittest

import torch

import models.linears.linear_adaptive.config as config

from emperor.augmentations.adaptive_parameters.core.bias import (
    AdditiveDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    CombinedDynamicDiagonalConfig,
    StandardDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    SingleModelDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    MaskDimensionOptions,
)
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import RecurrentLayerConfig
from emperor.experiments.base import GridSearch
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from models.linears.linear_adaptive.config_builder import LinearAdaptiveConfigBuilder
from models.linears.linear_adaptive.model import Model
from models.linears.linear_adaptive.presets import ExperimentOptions, ExperimentPresets
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestAdaptiveLinearModel(unittest.TestCase):
    def test_all_options_forward_one_mnist_batch(self):
        batch_size = 2
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for option in ExperimentOptions:
            with self.subTest(option=option.name):
                cfg = presets.get_config(option, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                output = model(X)
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_baseline_forwards_all_datasets(self):
        batch_size = 2
        presets = ExperimentPresets()

        for dataset in config.DATASET_OPTIONS:
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(ExperimentOptions.BASELINE, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                logits = model(X)

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_all_presets_train_one_epoch(self):
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for option in ExperimentOptions:
            with self.subTest(option=option.name):
                cfg = presets.get_config(option, dataset)[0]
                model = Model(cfg)
                datamodule = RandomImageClassificationDataModule(dataset)

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

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
            cfg.experiment_config.input_model_config.layer_model_config,
            LinearLayerConfig,
        )
        self.assertIsInstance(
            cfg.experiment_config.model_config.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )
        self.assertIsInstance(
            cfg.experiment_config.output_model_config.layer_model_config,
            LinearLayerConfig,
        )

    def test_boundary_layer_model_options_can_differ(self):
        cases = [
            (
                AdaptiveLinearLayerConfig,
                None,
                "input_layer_weight_option",
            ),
            (
                None,
                AdaptiveLinearLayerConfig,
                "output_layer_weight_option",
            ),
        ]

        for input_option, output_option, boundary_weight_key in cases:
            with self.subTest(
                input_option=self._option_name(input_option),
                output_option=self._option_name(output_option),
            ):
                boundary_kwargs = {
                    boundary_weight_key: DualModelDynamicWeightConfig,
                }
                cfg = LinearAdaptiveConfigBuilder(
                    input_dim=8,
                    hidden_dim=16,
                    output_dim=4,
                    weight_option=LowRankDynamicWeightConfig,
                    input_layer_model_option=input_option,
                    output_layer_model_option=output_option,
                    **boundary_kwargs,
                ).build()
                input_layer_model_config = (
                    cfg.experiment_config.input_model_config.layer_model_config
                )
                output_layer_model_config = (
                    cfg.experiment_config.output_model_config.layer_model_config
                )
                hidden_augmentation_config = (
                    cfg.experiment_config.model_config.layer_config.layer_model_config
                    .adaptive_augmentation_config
                )

                expected_input_type = input_option or LinearLayerConfig
                expected_output_type = output_option or LinearLayerConfig
                self.assertIs(type(input_layer_model_config), expected_input_type)
                self.assertIs(type(output_layer_model_config), expected_output_type)
                self.assertIsInstance(
                    hidden_augmentation_config.weight_config,
                    LowRankDynamicWeightConfig,
                )
                if input_option is AdaptiveLinearLayerConfig:
                    self.assertIsInstance(
                        input_layer_model_config.adaptive_augmentation_config.weight_config,
                        DualModelDynamicWeightConfig,
                    )
                if output_option is AdaptiveLinearLayerConfig:
                    self.assertIsInstance(
                        output_layer_model_config.adaptive_augmentation_config.weight_config,
                        DualModelDynamicWeightConfig,
                    )

                output = Model(cfg)(torch.randn(2, 1, 2, 4))
                logits = output[0] if isinstance(output, tuple) else output
                self.assertEqual(logits.shape, (2, 4))

    def test_adaptive_boundary_projectors_can_disable_augmentation(self):
        cfg = LinearAdaptiveConfigBuilder(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            input_layer_model_option=AdaptiveLinearLayerConfig,
            output_layer_model_option=AdaptiveLinearLayerConfig,
        ).build()

        input_augmentation_config = (
            cfg.experiment_config.input_model_config.layer_model_config
            .adaptive_augmentation_config
        )
        output_augmentation_config = (
            cfg.experiment_config.output_model_config.layer_model_config
            .adaptive_augmentation_config
        )

        for augmentation_config in (
            input_augmentation_config,
            output_augmentation_config,
        ):
            self.assertIsNone(augmentation_config.weight_config)
            self.assertIsNone(augmentation_config.bias_config)
            self.assertIsNone(augmentation_config.diagonal_config)
            self.assertIsNone(augmentation_config.mask_config)

        output = Model(cfg)(torch.randn(2, 1, 2, 4))
        logits = output[0] if isinstance(output, tuple) else output
        self.assertEqual(logits.shape, (2, 4))

    def test_boundary_adaptive_options_require_adaptive_boundary_projector(self):
        cases = [
            (
                "input_layer_weight_option",
                {"input_layer_weight_option": DualModelDynamicWeightConfig},
            ),
            (
                "output_layer_bias_option",
                {"output_layer_bias_option": AdditiveDynamicBiasConfig},
            ),
            (
                "input_layer_row_mask_option",
                {"input_layer_row_mask_option": WeightInformedScoreAxisMaskConfig},
            ),
        ]

        for expected_field, kwargs in cases:
            with self.subTest(expected_field=expected_field):
                with self.assertRaisesRegex(ValueError, expected_field):
                    LinearAdaptiveConfigBuilder(
                        input_dim=8,
                        hidden_dim=16,
                        output_dim=4,
                        **kwargs,
                    ).build()

    def test_boundary_layer_model_options_are_searchable(self):
        configs = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
            config.DATASET_OPTIONS[0],
            GridSearch(),
            search_keys=[
                "input_layer_model_option",
                "output_layer_model_option",
            ],
        )
        boundary_pairs = {
            (
                type(cfg.experiment_config.input_model_config.layer_model_config),
                type(cfg.experiment_config.output_model_config.layer_model_config),
            )
            for cfg in configs
        }

        self.assertEqual(len(configs), 4)
        self.assertEqual(
            boundary_pairs,
            {
                (LinearLayerConfig, LinearLayerConfig),
                (LinearLayerConfig, AdaptiveLinearLayerConfig),
                (AdaptiveLinearLayerConfig, LinearLayerConfig),
                (AdaptiveLinearLayerConfig, AdaptiveLinearLayerConfig),
            },
        )
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
            weight_option=DualModelDynamicWeightConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
            bias_option=AdditiveDynamicBiasConfig,
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
            augmentation_config.bias_config, AdditiveDynamicBiasConfig
        )
        self.assertIsInstance(
            augmentation_config.mask_config, WeightInformedScoreAxisMaskConfig
        )
        self.assertEqual(
            augmentation_config.mask_config.mask_dimension_option,
            MaskDimensionOptions.ROW,
        )

    def test_weight_bias_diagonal_presets_do_not_include_masks(self):
        expected_configs = {
            ExperimentOptions.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
                SingleModelDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                CombinedDynamicDiagonalConfig,
            ),
            ExperimentOptions.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
                DualModelDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                CombinedDynamicDiagonalConfig,
            ),
            ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
                LayeredWeightedBankDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                CombinedDynamicDiagonalConfig,
            ),
            ExperimentOptions.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
                LowRankDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                CombinedDynamicDiagonalConfig,
            ),
            ExperimentOptions.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
                SingleModelDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                StandardDynamicDiagonalConfig,
            ),
            ExperimentOptions.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
                DualModelDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                StandardDynamicDiagonalConfig,
            ),
            ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
                LayeredWeightedBankDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                StandardDynamicDiagonalConfig,
            ),
            ExperimentOptions.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
                LowRankDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                StandardDynamicDiagonalConfig,
            ),
        }

        for option, expected_types in expected_configs.items():
            with self.subTest(option=option.name):
                cfg = ExperimentPresets().get_config(option)[0]
                augmentation_config = (
                    cfg.experiment_config.model_config.layer_config.layer_model_config
                    .adaptive_augmentation_config
                )

                self.assertIsInstance(
                    augmentation_config.weight_config, expected_types[0]
                )
                self.assertIsInstance(
                    augmentation_config.bias_config, expected_types[1]
                )
                self.assertIsInstance(
                    augmentation_config.diagonal_config, expected_types[2]
                )
                self.assertIsNone(augmentation_config.mask_config)

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
            halting_hidden_dim=48,
            halting_output_dim=2,
            halting_layer_norm_position=LayerNormPositionOptions.BEFORE,
            halting_stack_num_layers=5,
            halting_stack_activation=ActivationOptions.MISH,
            halting_stack_residual_flag=False,
            halting_stack_dropout_probability=0.3,
            halting_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            halting_stack_apply_output_pipeline_flag=True,
            halting_bias_flag=False,
        ).build()

        halting_cfg = cfg.experiment_config.model_config.layer_config.halting_config
        halting_stack_cfg = halting_cfg.halting_gate_config

        self.assertEqual(halting_cfg.threshold, 0.5)
        self.assertEqual(halting_cfg.halting_dropout, 0.2)
        self.assertEqual(halting_stack_cfg.hidden_dim, 48)
        self.assertEqual(halting_stack_cfg.output_dim, 2)
        self.assertEqual(halting_stack_cfg.num_layers, 5)
        self.assertEqual(
            halting_stack_cfg.last_layer_bias_option, LastLayerBiasOptions.DISABLED
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

    def test_recurrent_presets_wire_optional_controllers(self):
        expected_controllers = {
            ExperimentOptions.RECURRENT: (False, False),
            ExperimentOptions.RECURRENT_GATING: (True, False),
            ExperimentOptions.RECURRENT_HALTING: (False, True),
            ExperimentOptions.RECURRENT_GATING_HALTING: (True, True),
            ExperimentOptions.FULL_STACK_RECURRENT: (False, False),
        }

        for option, (expected_gate, expected_halting) in expected_controllers.items():
            with self.subTest(option=option.name):
                cfg = ExperimentPresets().get_config(option)[0]
                recurrent_cfg = cfg.experiment_config.model_config

                self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
                self.assertEqual(recurrent_cfg.gate_config is not None, expected_gate)
                self.assertEqual(
                    recurrent_cfg.halting_config is not None,
                    expected_halting,
                )

    def test_new_adaptive_combination_presets_wire_config(self):
        presets = ExperimentPresets()

        cfg = presets.get_config(ExperimentOptions.DUAL_WEIGHT_GATING)[0]
        layer_cfg = cfg.experiment_config.model_config.layer_config
        augmentation_config = self._augmentation_config(cfg)
        self.assertIsInstance(
            augmentation_config.weight_config, DualModelDynamicWeightConfig
        )
        self.assertIsNotNone(layer_cfg.gate_config)

        cfg = presets.get_config(ExperimentOptions.DUAL_WEIGHT_HALTING)[0]
        layer_cfg = cfg.experiment_config.model_config.layer_config
        augmentation_config = self._augmentation_config(cfg)
        self.assertIsInstance(
            augmentation_config.weight_config, DualModelDynamicWeightConfig
        )
        self.assertIsNotNone(layer_cfg.halting_config)

        cfg = presets.get_config(ExperimentOptions.FULL_STACK_GATING)[0]
        layer_cfg = cfg.experiment_config.model_config.layer_config
        self._assert_full_stack_augmentation(self._augmentation_config(cfg))
        self.assertIsNotNone(layer_cfg.gate_config)

        cfg = presets.get_config(ExperimentOptions.FULL_STACK_RECURRENT)[0]
        recurrent_cfg = cfg.experiment_config.model_config
        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self._assert_full_stack_augmentation(self._augmentation_config(cfg))

        cfg = presets.get_config(ExperimentOptions.BANK_WEIGHT_MASK)[0]
        augmentation_config = self._augmentation_config(cfg)
        self.assertIsInstance(
            augmentation_config.weight_config,
            LayeredWeightedBankDynamicWeightConfig,
        )
        self.assertIsInstance(
            augmentation_config.mask_config,
            WeightInformedScoreAxisMaskConfig,
        )
        self.assertIsNone(augmentation_config.bias_config)
        self.assertIsNone(augmentation_config.diagonal_config)

        cfg = presets.get_config(ExperimentOptions.LOW_RANK_POST_NORM)[0]
        layer_cfg = cfg.experiment_config.model_config.layer_config
        augmentation_config = self._augmentation_config(cfg)
        self.assertIsInstance(
            augmentation_config.weight_config,
            LowRankDynamicWeightConfig,
        )
        self.assertEqual(
            layer_cfg.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )

    def _augmentation_config(self, cfg):
        model_config = cfg.experiment_config.model_config
        if isinstance(model_config, RecurrentLayerConfig):
            model_config = model_config.block_config
        return (
            model_config.layer_config.layer_model_config.adaptive_augmentation_config
        )

    def _assert_full_stack_augmentation(self, augmentation_config) -> None:
        self.assertIsInstance(
            augmentation_config.weight_config,
            DualModelDynamicWeightConfig,
        )
        self.assertIsInstance(
            augmentation_config.bias_config,
            AdditiveDynamicBiasConfig,
        )
        self.assertIsInstance(
            augmentation_config.diagonal_config,
            CombinedDynamicDiagonalConfig,
        )
        self.assertIsInstance(
            augmentation_config.mask_config,
            WeightInformedScoreAxisMaskConfig,
        )

    def _option_name(self, option) -> str:
        if option is None:
            return "None"
        return option.__name__

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )


if __name__ == "__main__":
    unittest.main()

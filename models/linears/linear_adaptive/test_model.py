from emperor.base.layer.residual import ResidualConnectionOptions
import unittest

import torch

import models.linears.linear_adaptive.config as config

from emperor.augmentations.adaptive_parameters.core.bias import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    GeneratorDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
    StandardDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    DiagonalAxisMaskConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
)
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.experiments.base import GridSearch, PresetLock
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from emperor.memory.config import (
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from emperor.memory.options import MemoryPositionOptions
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

    def shared_gate_config(self, dim: int = 16) -> GateConfig:
        return GateConfig(
            model_config=LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=dim,
                output_dim=dim,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    bias_flag=True,
                ),
            ),
            ),
            option=LayerGateOptions.MULTIPLIER,
            activation=ActivationOptions.SIGMOID,
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
                hidden_augmentation_config = cfg.experiment_config.model_config.layer_config.layer_model_config.adaptive_augmentation_config

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

        input_augmentation_config = cfg.experiment_config.input_model_config.layer_model_config.adaptive_augmentation_config
        output_augmentation_config = cfg.experiment_config.output_model_config.layer_model_config.adaptive_augmentation_config

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
                "input_layer_bias_option",
                {"input_layer_bias_option": AdditiveDynamicBiasConfig},
            ),
            (
                "input_layer_diagonal_option",
                {"input_layer_diagonal_option": StandardDynamicDiagonalConfig},
            ),
            (
                "input_layer_row_mask_option",
                {"input_layer_row_mask_option": WeightInformedScoreAxisMaskConfig},
            ),
            (
                "output_layer_weight_option",
                {"output_layer_weight_option": DualModelDynamicWeightConfig},
            ),
            (
                "output_layer_bias_option",
                {"output_layer_bias_option": AdditiveDynamicBiasConfig},
            ),
            (
                "output_layer_diagonal_option",
                {"output_layer_diagonal_option": StandardDynamicDiagonalConfig},
            ),
            (
                "output_layer_row_mask_option",
                {"output_layer_row_mask_option": WeightInformedScoreAxisMaskConfig},
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

    def test_stack_layer_norm_position_is_searchable(self):
        configs = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
            config.DATASET_OPTIONS[0],
            GridSearch(),
            search_keys=["stack_layer_norm_position"],
        )
        layer_norm_positions = {
            cfg.experiment_config.model_config.layer_config.layer_norm_position
            for cfg in configs
        }

        self.assertEqual(len(configs), len(config.SEARCH_SPACE_STACK_LAYER_NORM_POSITION))
        self.assertEqual(
            layer_norm_positions,
            set(config.SEARCH_SPACE_STACK_LAYER_NORM_POSITION),
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
        augmentation_config = cfg.experiment_config.model_config.layer_config.layer_model_config.adaptive_augmentation_config

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

    def test_individual_adaptive_presets_wire_expected_config_type(self):
        cases = [
            (
                ExperimentOptions.SINGLE_MODEL_WEIGHT,
                "weight_config",
                SingleModelDynamicWeightConfig,
            ),
            (
                ExperimentOptions.DUAL_MODEL_WEIGHT,
                "weight_config",
                DualModelDynamicWeightConfig,
            ),
            (
                ExperimentOptions.LOW_RANK_WEIGHT,
                "weight_config",
                LowRankDynamicWeightConfig,
            ),
            (
                ExperimentOptions.HYPERNETWORK_WEIGHT,
                "weight_config",
                HypernetworkDynamicWeightConfig,
            ),
            (
                ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT,
                "weight_config",
                LayeredWeightedBankDynamicWeightConfig,
            ),
            (
                ExperimentOptions.SOFT_WEIGHTED_BANK_WEIGHT,
                "weight_config",
                SoftWeightedBankDynamicWeightConfig,
            ),
            (
                ExperimentOptions.AFFINE_TRANSFORM_BIAS,
                "bias_config",
                AffineTransformDynamicBiasConfig,
            ),
            (
                ExperimentOptions.ADDITIVE_BIAS,
                "bias_config",
                AdditiveDynamicBiasConfig,
            ),
            (
                ExperimentOptions.GENERATOR_BIAS,
                "bias_config",
                GeneratorDynamicBiasConfig,
            ),
            (
                ExperimentOptions.MULTIPLICATIVE_BIAS,
                "bias_config",
                MultiplicativeDynamicBiasConfig,
            ),
            (
                ExperimentOptions.SIGMOID_GATED_BIAS,
                "bias_config",
                SigmoidGatedDynamicBiasConfig,
            ),
            (
                ExperimentOptions.TANH_GATED_BIAS,
                "bias_config",
                TanhGatedDynamicBiasConfig,
            ),
            (
                ExperimentOptions.WEIGHTED_BANK_BIAS,
                "bias_config",
                WeightedBankDynamicBiasConfig,
            ),
            (
                ExperimentOptions.STANDARD_DIAGONAL,
                "diagonal_config",
                StandardDynamicDiagonalConfig,
            ),
            (
                ExperimentOptions.ANTI_DIAGONAL,
                "diagonal_config",
                AntiDynamicDiagonalConfig,
            ),
            (
                ExperimentOptions.COMBINED_DIAGONAL,
                "diagonal_config",
                CombinedDynamicDiagonalConfig,
            ),
            (
                ExperimentOptions.DIAGONAL_AXIS_MASK,
                "mask_config",
                DiagonalAxisMaskConfig,
            ),
            (
                ExperimentOptions.OUTER_PRODUCT_MASK,
                "mask_config",
                OuterProductMaskConfig,
            ),
            (
                ExperimentOptions.PER_AXIS_SCORE_MASK,
                "mask_config",
                PerAxisScoreMaskConfig,
            ),
            (
                ExperimentOptions.TOP_SLICE_AXIS_MASK,
                "mask_config",
                TopSliceAxisMaskConfig,
            ),
            (
                ExperimentOptions.WEIGHT_INFORMED_SCORE_MASK,
                "mask_config",
                WeightInformedScoreAxisMaskConfig,
            ),
        ]

        for option, expected_field, expected_type in cases:
            with self.subTest(option=option.name):
                cfg = ExperimentPresets().get_config(option)[0]
                augmentation_config = self._augmentation_config(cfg)

                for field in (
                    "weight_config",
                    "bias_config",
                    "diagonal_config",
                    "mask_config",
                ):
                    value = getattr(augmentation_config, field)
                    if field == expected_field:
                        self.assertIsInstance(value, expected_type)
                    else:
                        self.assertIsNone(value)

    def test_specialized_weight_presets_wire_expected_knobs(self):
        cases = [
            (
                ExperimentOptions.DECAY_EXPONENTIAL_WEIGHT,
                {
                    "decay_schedule": WeightDecayScheduleOptions.EXPONENTIAL,
                    "decay_rate": 1e-3,
                    "decay_warmup_batches": 500,
                },
            ),
            (
                ExperimentOptions.NORM_L2_WEIGHT,
                {
                    "normalization_option": WeightNormalizationOptions.L2_SCALE,
                },
            ),
            (
                ExperimentOptions.DEEP_GENERATOR,
                {
                    "generator_depth": DynamicDepthOptions.DEPTH_OF_EIGHT,
                },
            ),
        ]

        for option, expected_values in cases:
            with self.subTest(option=option.name):
                cfg = ExperimentPresets().get_config(option)[0]
                weight_config = self._augmentation_config(cfg).weight_config

                self.assertIsInstance(weight_config, DualModelDynamicWeightConfig)
                for field, expected in expected_values.items():
                    self.assertEqual(getattr(weight_config, field), expected)

    def test_preset_locks_are_exposed_with_reasons(self):
        presets = ExperimentPresets()

        for option, expected_locks in presets.PRESET_LOCKS.items():
            with self.subTest(option=option.name):
                locks = presets.locked_fields(option)

                self.assertEqual(set(locks), set(expected_locks))
                for field, lock in locks.items():
                    expected = expected_locks[field]
                    expected_value = (
                        expected.value if isinstance(expected, PresetLock) else expected
                    )
                    self.assertEqual(lock.value, expected_value)
                    self.assertIn(option.name, lock.reason)

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
                augmentation_config = cfg.experiment_config.model_config.layer_config.layer_model_config.adaptive_augmentation_config

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
            gate_option=LayerGateOptions.MULTIPLIER,
            gate_hidden_dim=32,
            gate_layer_norm_position=LayerNormPositionOptions.AFTER,
            gate_stack_num_layers=3,
            gate_stack_activation=ActivationOptions.SILU,
            gate_stack_residual_connection_option=ResidualConnectionOptions.DISABLED,
            gate_stack_dropout_probability=0.1,
            gate_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            gate_stack_apply_output_pipeline_flag=True,
            gate_bias_flag=False,
        ).build()

        layer_cfg = cfg.experiment_config.model_config.layer_config
        gate_config = layer_cfg.gate_config
        gate_cfg = gate_config.model_config

        self.assertEqual(gate_config.option, LayerGateOptions.MULTIPLIER)
        self.assertEqual(gate_cfg.hidden_dim, 32)
        self.assertEqual(gate_cfg.num_layers, 3)
        self.assertEqual(gate_cfg.last_layer_bias_option, LastLayerBiasOptions.DISABLED)
        self.assertTrue(gate_cfg.apply_output_pipeline_flag)
        self.assertEqual(
            gate_cfg.layer_config.layer_norm_position, LayerNormPositionOptions.AFTER
        )
        self.assertEqual(gate_cfg.layer_config.activation, ActivationOptions.SILU)
        self.assertEqual(
            gate_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(gate_cfg.layer_config.dropout_probability, 0.1)
        self.assertFalse(gate_cfg.layer_config.layer_model_config.bias_flag)

    def test_shared_gate_config_is_stored_on_stack_config(self):
        shared_gate_config = self.shared_gate_config()
        cfg = LinearAdaptiveConfigBuilder(shared_gate_config=shared_gate_config).build()
        model_cfg = cfg.experiment_config.model_config

        self.assertIs(model_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(model_cfg.layer_config.gate_config)

    def test_shared_gate_config_rejects_enabled_stack_gate(self):
        with self.assertRaises(ValueError):
            LinearAdaptiveConfigBuilder(
                stack_gate_flag=True,
                shared_gate_config=self.shared_gate_config(),
            ).build()

    def test_shared_gate_config_allows_absent_stack_gate(self):
        shared_gate_config = self.shared_gate_config()
        cfg = LinearAdaptiveConfigBuilder(
            stack_gate_flag=False,
            shared_gate_config=shared_gate_config,
        ).build()
        model_cfg = cfg.experiment_config.model_config

        self.assertIs(model_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(model_cfg.layer_config.gate_config)

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
            halting_stack_residual_connection_option=ResidualConnectionOptions.DISABLED,
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
        self.assertEqual(
            halting_stack_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(halting_stack_cfg.layer_config.dropout_probability, 0.3)
        self.assertFalse(halting_stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_controller_stack_defaults_preserve_current_behavior(self):
        cfg = LinearAdaptiveConfigBuilder(
            stack_gate_flag=True,
            stack_halting_flag=True,
            memory_flag=True,
        ).build()

        model_cfg = cfg.experiment_config.model_config
        gate_cfg = model_cfg.layer_config.gate_config.model_config
        halting_stack_cfg = model_cfg.layer_config.halting_config.halting_gate_config
        memory_stack_cfg = model_cfg.shared_memory_config.model_config

        self.assertEqual(gate_cfg.hidden_dim, config.HIDDEN_DIM)
        self.assertEqual(gate_cfg.num_layers, 2)
        self.assertEqual(gate_cfg.layer_config.activation, ActivationOptions.TANH)
        self.assertEqual(
            gate_cfg.layer_config.layer_norm_position,
            config.STACK_LAYER_NORM_POSITION,
        )
        self.assertEqual(
            gate_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(gate_cfg.layer_config.dropout_probability, 0.0)
        self.assertEqual(
            gate_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DEFAULT,
        )
        self.assertTrue(gate_cfg.apply_output_pipeline_flag)
        self.assertTrue(gate_cfg.layer_config.layer_model_config.bias_flag)

        self.assertEqual(halting_stack_cfg.hidden_dim, config.HIDDEN_DIM)
        self.assertEqual(halting_stack_cfg.num_layers, 2)
        self.assertEqual(
            halting_stack_cfg.layer_config.activation,
            ActivationOptions.GELU,
        )
        self.assertEqual(
            halting_stack_cfg.layer_config.layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertEqual(
            halting_stack_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertFalse(halting_stack_cfg.apply_output_pipeline_flag)
        self.assertTrue(
            halting_stack_cfg.layer_config.layer_model_config.bias_flag
        )

        self.assertEqual(memory_stack_cfg.hidden_dim, config.HIDDEN_DIM)
        self.assertEqual(memory_stack_cfg.num_layers, 2)
        self.assertEqual(
            memory_stack_cfg.layer_config.activation,
            ActivationOptions.GELU,
        )
        self.assertEqual(
            memory_stack_cfg.layer_config.layer_norm_position,
            config.STACK_LAYER_NORM_POSITION,
        )
        self.assertEqual(
            memory_stack_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DEFAULT,
        )
        self.assertFalse(memory_stack_cfg.apply_output_pipeline_flag)
        self.assertTrue(memory_stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_controller_stacks_inherit_submodule_defaults_when_overrides_are_none(
        self,
    ):
        cfg = LinearAdaptiveConfigBuilder(
            stack_gate_flag=True,
            stack_halting_flag=True,
            memory_flag=True,
            submodule_hidden_dim=37,
            submodule_layer_norm_position=LayerNormPositionOptions.AFTER,
            submodule_stack_num_layers=4,
            submodule_stack_activation=ActivationOptions.MISH,
            submodule_stack_residual_connection_option=(
                ResidualConnectionOptions.RESIDUAL
            ),
            submodule_stack_dropout_probability=0.12,
            submodule_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            submodule_stack_apply_output_pipeline_flag=True,
            submodule_bias_flag=False,
            gate_stack_activation=None,
            gate_stack_apply_output_pipeline_flag=None,
            gate_bias_flag=None,
            halting_layer_norm_position=None,
            halting_stack_last_layer_bias_option=None,
        ).build()

        model_cfg = cfg.experiment_config.model_config
        stacks = [
            model_cfg.layer_config.gate_config.model_config,
            model_cfg.layer_config.halting_config.halting_gate_config,
            model_cfg.shared_memory_config.model_config,
        ]

        for stack_cfg in stacks:
            with self.subTest(stack=type(stack_cfg).__name__):
                self.assertEqual(stack_cfg.hidden_dim, 37)
                self.assertEqual(stack_cfg.num_layers, 4)
                self.assertEqual(
                    stack_cfg.last_layer_bias_option,
                    LastLayerBiasOptions.DISABLED,
                )
                self.assertTrue(stack_cfg.apply_output_pipeline_flag)
                self.assertEqual(
                    stack_cfg.layer_config.activation,
                    ActivationOptions.MISH,
                )
                self.assertEqual(
                    stack_cfg.layer_config.layer_norm_position,
                    LayerNormPositionOptions.AFTER,
                )
                self.assertEqual(
                    stack_cfg.layer_config.residual_connection_option,
                    ResidualConnectionOptions.RESIDUAL,
                )
                self.assertEqual(stack_cfg.layer_config.dropout_probability, 0.12)
                self.assertFalse(
                    stack_cfg.layer_config.layer_model_config.bias_flag
                )

    def test_controller_stack_overrides_win_over_submodule_defaults(self):
        cfg = LinearAdaptiveConfigBuilder(
            stack_gate_flag=True,
            stack_halting_flag=True,
            memory_flag=True,
            submodule_hidden_dim=11,
            submodule_stack_activation=ActivationOptions.RELU,
            gate_hidden_dim=22,
            gate_stack_activation=ActivationOptions.SILU,
            halting_hidden_dim=33,
            halting_stack_activation=ActivationOptions.MISH,
            memory_hidden_dim=44,
            memory_stack_activation=ActivationOptions.TANH,
        ).build()

        model_cfg = cfg.experiment_config.model_config

        self.assertEqual(model_cfg.layer_config.gate_config.model_config.hidden_dim, 22)
        self.assertEqual(
            model_cfg.layer_config.gate_config.model_config.layer_config.activation,
            ActivationOptions.SILU,
        )
        self.assertEqual(
            model_cfg.layer_config.halting_config.halting_gate_config.hidden_dim,
            33,
        )
        self.assertEqual(
            model_cfg.layer_config.halting_config.halting_gate_config.layer_config.activation,
            ActivationOptions.MISH,
        )
        self.assertEqual(model_cfg.shared_memory_config.model_config.hidden_dim, 44)
        self.assertEqual(
            model_cfg.shared_memory_config.model_config.layer_config.activation,
            ActivationOptions.TANH,
        )

    def test_memory_config_uses_builder_defaults(self):
        cfg = LinearAdaptiveConfigBuilder(
            input_dim=8,
            hidden_dim=8,
            output_dim=4,
            memory_flag=True,
        ).build()

        model_cfg = cfg.experiment_config.model_config
        memory_cfg = model_cfg.shared_memory_config

        self.assertIsInstance(memory_cfg, GatedResidualDynamicMemoryConfig)
        self.assertEqual(
            memory_cfg.memory_position_option,
            MemoryPositionOptions.AFTER_AFFINE,
        )
        self.assertIsNone(memory_cfg.test_time_training_learning_rate)
        self.assertIsNone(memory_cfg.test_time_training_num_inner_steps)
        self.assertIsNone(model_cfg.layer_config.memory_config)

    def test_memory_config_uses_builder_overrides(self):
        cfg = LinearAdaptiveConfigBuilder(
            input_dim=8,
            hidden_dim=8,
            output_dim=4,
            memory_flag=True,
            memory_option=WeightedDynamicMemoryConfig,
            memory_position_option=MemoryPositionOptions.BEFORE_AFFINE,
            memory_test_time_training_learning_rate=0.02,
            memory_test_time_training_num_inner_steps=2,
            memory_hidden_dim=12,
            memory_layer_norm_position=LayerNormPositionOptions.AFTER,
            memory_stack_num_layers=3,
            memory_stack_activation=ActivationOptions.SILU,
            memory_stack_residual_connection_option=(
                ResidualConnectionOptions.DISABLED
            ),
            memory_stack_dropout_probability=0.1,
            memory_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            memory_stack_apply_output_pipeline_flag=True,
            memory_bias_flag=False,
        ).build()

        memory_cfg = cfg.experiment_config.model_config.shared_memory_config
        generator_cfg = memory_cfg.model_config

        self.assertIsInstance(memory_cfg, WeightedDynamicMemoryConfig)
        self.assertEqual(
            memory_cfg.memory_position_option,
            MemoryPositionOptions.BEFORE_AFFINE,
        )
        self.assertEqual(memory_cfg.test_time_training_learning_rate, 0.02)
        self.assertEqual(memory_cfg.test_time_training_num_inner_steps, 2)
        self.assertEqual(generator_cfg.hidden_dim, 12)
        self.assertEqual(generator_cfg.num_layers, 3)
        self.assertEqual(
            generator_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertTrue(generator_cfg.apply_output_pipeline_flag)
        self.assertEqual(
            generator_cfg.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(generator_cfg.layer_config.activation, ActivationOptions.SILU)
        self.assertEqual(
            generator_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(generator_cfg.layer_config.dropout_probability, 0.1)
        self.assertFalse(generator_cfg.layer_config.layer_model_config.bias_flag)
        self.assertIsNone(generator_cfg.layer_config.gate_config)
        self.assertIsNone(generator_cfg.layer_config.halting_config)
        self.assertIsNone(generator_cfg.layer_config.memory_config)
        self.assertIsNone(generator_cfg.shared_memory_config)
        self.assertIsNone(generator_cfg.shared_halting_config)

    def test_memory_enabled_forwards_one_fake_batch(self):
        cfg = LinearAdaptiveConfigBuilder(
            input_dim=8,
            hidden_dim=8,
            output_dim=4,
            stack_num_layers=2,
            memory_flag=True,
        ).build()
        model = Model(cfg)

        logits = model(torch.randn(2, 1, 2, 4))

        self.assertEqual(logits.shape, (2, 4))

    def test_memory_enabled_backward_produces_memory_gradients(self):
        cfg = LinearAdaptiveConfigBuilder(
            input_dim=8,
            hidden_dim=8,
            output_dim=4,
            stack_num_layers=2,
            memory_flag=True,
        ).build()
        model = Model(cfg)
        output = model(torch.randn(2, 1, 2, 4))
        logits = output[0] if isinstance(output, tuple) else output

        logits.sum().backward()

        memory_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "memory_model" in name and parameter.requires_grad
        ]
        nonzero_memory_gradients = [
            parameter.grad
            for parameter in memory_parameters
            if parameter.grad is not None and torch.any(parameter.grad.abs() > 0)
        ]
        self.assertTrue(len(memory_parameters) > 0)
        self.assertTrue(len(nonzero_memory_gradients) > 0)

    def test_recurrent_memory_stays_on_block_config(self):
        cfg = LinearAdaptiveConfigBuilder(
            input_dim=8,
            hidden_dim=8,
            output_dim=4,
            recurrent_flag=True,
            memory_flag=True,
        ).build()

        recurrent_cfg = cfg.experiment_config.model_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertIsNotNone(recurrent_cfg.block_config.shared_memory_config)
        self.assertIsNone(recurrent_cfg.memory_config)

    def test_recurrent_layer_norm_position_defaults_disabled_and_uses_override(self):
        default_cfg = LinearAdaptiveConfigBuilder(recurrent_flag=True).build()
        default_recurrent_cfg = default_cfg.experiment_config.model_config

        self.assertIsInstance(default_recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            default_recurrent_cfg.recurrent_layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )

        cfg = LinearAdaptiveConfigBuilder(
            recurrent_flag=True,
            recurrent_layer_norm_position=LayerNormPositionOptions.BEFORE,
        ).build()
        recurrent_cfg = cfg.experiment_config.model_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            recurrent_cfg.recurrent_layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )

    def test_recurrent_controllers_use_resolved_gate_and_halting_builders(self):
        cfg = LinearAdaptiveConfigBuilder(
            recurrent_flag=True,
            recurrent_gate_flag=True,
            recurrent_gate_option=LayerGateOptions.MULTIPLIER,
            recurrent_halting_flag=True,
            submodule_hidden_dim=29,
            submodule_layer_norm_position=LayerNormPositionOptions.AFTER,
            submodule_stack_num_layers=3,
            submodule_stack_activation=ActivationOptions.MISH,
            submodule_stack_residual_connection_option=(
                ResidualConnectionOptions.RESIDUAL
            ),
            submodule_stack_dropout_probability=0.18,
            submodule_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            submodule_stack_apply_output_pipeline_flag=True,
            submodule_bias_flag=False,
            gate_stack_activation=None,
            gate_stack_apply_output_pipeline_flag=None,
            gate_bias_flag=None,
            halting_layer_norm_position=None,
            halting_stack_last_layer_bias_option=None,
        ).build()

        recurrent_cfg = cfg.experiment_config.model_config
        self.assertEqual(recurrent_cfg.gate_config.option, LayerGateOptions.MULTIPLIER)
        gate_cfg = recurrent_cfg.gate_config.model_config
        halting_stack_cfg = recurrent_cfg.halting_config.halting_gate_config

        for stack_cfg in (gate_cfg, halting_stack_cfg):
            with self.subTest(stack=type(stack_cfg).__name__):
                self.assertEqual(stack_cfg.hidden_dim, 29)
                self.assertEqual(stack_cfg.num_layers, 3)
                self.assertEqual(
                    stack_cfg.last_layer_bias_option,
                    LastLayerBiasOptions.DISABLED,
                )
                self.assertTrue(stack_cfg.apply_output_pipeline_flag)
                self.assertEqual(
                    stack_cfg.layer_config.activation,
                    ActivationOptions.MISH,
                )
                self.assertEqual(
                    stack_cfg.layer_config.layer_norm_position,
                    LayerNormPositionOptions.AFTER,
                )
                self.assertEqual(
                    stack_cfg.layer_config.residual_connection_option,
                    ResidualConnectionOptions.RESIDUAL,
                )
                self.assertEqual(stack_cfg.layer_config.dropout_probability, 0.18)
                self.assertFalse(
                    stack_cfg.layer_config.layer_model_config.bias_flag
                )

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
        return model_config.layer_config.layer_model_config.adaptive_augmentation_config

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

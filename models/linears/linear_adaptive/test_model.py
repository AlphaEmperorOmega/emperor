import unittest

import torch
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
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.base.layer import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.experiments.base import GridSearch, PresetDefinition, PresetLock
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from emperor.memory.config import (
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from emperor.memory.options import MemoryPositionOptions

import models.linears.linear_adaptive.config as config
from models.linears._builder_options import (
    LinearStackOptions,
)
from models.linears.linear_adaptive.config_builder import (
    LinearAdaptiveConfigBuilder,
)
from models.linears.linear_adaptive.model import Model
from models.linears.linear_adaptive.presets import (
    ExperimentPreset,
    ExperimentPresets,
    _PRESET_DEFINITIONS,
)
from models.adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestAdaptiveLinearModel(unittest.TestCase):
    def adaptive_preset(self, **kwargs):
        return ExperimentPresets()._preset(**kwargs)

    def test_all_presets_forward_one_mnist_batch(self):
        batch_size = 2
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset, dataset)[0]
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
                cfg = presets.get_config(ExperimentPreset.BASELINE, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                logits = model(X)

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_all_presets_train_one_epoch(self):
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset, dataset)[0]
                model = Model(cfg)
                datamodule = RandomImageClassificationDataModule(dataset)

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def test_preset_builds_adaptive_linear_layer_config(self):
        cfg = ExperimentPresets()._preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
        )
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
            output_dim=4,
            stack_options=LinearStackOptions(
                hidden_dim=16,
                bias_flag=config.STACK_BIAS_FLAG,
                layer_norm_position=config.STACK_LAYER_NORM_POSITION,
                num_layers=config.STACK_NUM_LAYERS,
                activation=config.STACK_ACTIVATION,
                residual_connection_option=(
                    config.STACK_RESIDUAL_CONNECTION_OPTION
                ),
                dropout_probability=config.STACK_DROPOUT_PROBABILITY,
                last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
                apply_output_pipeline_flag=(
                    config.STACK_APPLY_OUTPUT_PIPELINE_FLAG
                ),
            ),
        ).build()

        self.assertEqual(cfg.input_dim, 8)
        self.assertEqual(cfg.hidden_dim, 16)
        self.assertEqual(cfg.output_dim, 4)
        self.assertIsInstance(
            cfg.experiment_config.input_model_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )
        self.assertIsInstance(
            cfg.experiment_config.model_config.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )
        self.assertIsInstance(
            cfg.experiment_config.output_model_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )

    def test_flat_builder_kwargs_are_rejected(self):
        cases = (
            {"stack_hidden_dim": 13},
            {"adaptive_generator_stack_hidden_dim": 19},
            {"input_layer_weight_option": DualModelDynamicWeightConfig},
            {"memory_flag": True},
            {"recurrent_flag": True},
            {"shared_gate_config": self.shared_gate_config()},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    LinearAdaptiveConfigBuilder(**kwargs)

    def test_boundary_layer_pipeline_matches_linear_projection_options(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            stack_activation=ActivationOptions.RELU,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            stack_dropout_probability=0.35,
            input_layer_weight_option=DualModelDynamicWeightConfig,
            output_layer_weight_option=LowRankDynamicWeightConfig,
            adaptive_generator_stack_hidden_dim=21,
            adaptive_generator_stack_num_layers=3,
            adaptive_generator_stack_activation=ActivationOptions.MISH,
            adaptive_generator_stack_layer_norm_position=(
                LayerNormPositionOptions.BEFORE
            ),
            adaptive_generator_stack_dropout_probability=0.2,
            adaptive_generator_stack_last_layer_bias_option=(
                LastLayerBiasOptions.DISABLED
            ),
            adaptive_generator_stack_apply_output_pipeline_flag=True,
            adaptive_generator_stack_bias_flag=False,
        )

        input_config = cfg.experiment_config.input_model_config
        hidden_config = cfg.experiment_config.model_config.layer_config
        output_config = cfg.experiment_config.output_model_config

        self.assertEqual(input_config.activation, ActivationOptions.RELU)
        self.assertEqual(
            input_config.layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertEqual(
            input_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(input_config.dropout_probability, 0.0)
        self.assertIsNone(input_config.gate_config)
        self.assertIsNone(input_config.halting_config)
        self.assertIsNone(input_config.memory_config)

        self.assertEqual(output_config.activation, ActivationOptions.DISABLED)
        self.assertEqual(
            output_config.layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertEqual(
            output_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(output_config.dropout_probability, 0.0)
        self.assertIsNone(output_config.gate_config)
        self.assertIsNone(output_config.halting_config)
        self.assertIsNone(output_config.memory_config)

        self.assertEqual(
            hidden_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(hidden_config.dropout_probability, 0.35)

        for layer_model_config in (
            input_config.layer_model_config,
            output_config.layer_model_config,
        ):
            self.assertIsInstance(layer_model_config, AdaptiveLinearLayerConfig)
            self._assert_generator_stack(
                layer_model_config.adaptive_augmentation_config.model_config,
                hidden_dim=21,
                num_layers=3,
                activation=ActivationOptions.MISH,
                layer_norm_position=LayerNormPositionOptions.BEFORE,
                dropout_probability=0.2,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_pipeline_flag=True,
                bias_flag=False,
            )

    def test_config_builder_applies_stack_bias_flag_to_layer_stack(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            stack_bias_flag=False,
        )

        layer_model_config = (
            cfg.experiment_config.model_config.layer_config.layer_model_config
        )

        self.assertIsInstance(layer_model_config, AdaptiveLinearLayerConfig)
        self.assertFalse(layer_model_config.bias_flag)

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

    def _controller_stack_configs(self, model_cfg) -> dict[str, LayerStackConfig]:
        return {
            "gate": model_cfg.layer_config.gate_config.model_config,
            "halting": model_cfg.layer_config.halting_config.halting_gate_config,
            "memory": model_cfg.shared_memory_config.model_config,
        }

    def _assert_controller_stack_options(
        self,
        stack_cfg: LayerStackConfig,
        *,
        hidden_dim: int,
        num_layers: int,
        activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        residual_connection_option: ResidualConnectionOptions,
        dropout_probability: float,
        last_layer_bias_option: LastLayerBiasOptions,
        apply_output_pipeline_flag: bool,
        bias_flag: bool,
    ) -> None:
        self.assertEqual(stack_cfg.hidden_dim, hidden_dim)
        self.assertEqual(stack_cfg.num_layers, num_layers)
        self.assertEqual(stack_cfg.layer_config.activation, activation)
        self.assertEqual(
            stack_cfg.layer_config.layer_norm_position,
            layer_norm_position,
        )
        self.assertEqual(
            stack_cfg.layer_config.residual_connection_option,
            residual_connection_option,
        )
        self.assertEqual(
            stack_cfg.layer_config.dropout_probability,
            dropout_probability,
        )
        self.assertEqual(stack_cfg.last_layer_bias_option, last_layer_bias_option)
        self.assertEqual(
            stack_cfg.apply_output_pipeline_flag,
            apply_output_pipeline_flag,
        )
        self.assertEqual(
            stack_cfg.layer_config.layer_model_config.bias_flag,
            bias_flag,
        )

    def test_boundary_projectors_enable_adaptive_options_independently(self):
        cases = [
            (
                "input_layer_weight_option",
                "input",
            ),
            (
                "output_layer_weight_option",
                "output",
            ),
        ]

        for boundary_weight_key, enabled_boundary in cases:
            with self.subTest(enabled_boundary=enabled_boundary):
                boundary_kwargs = {
                    boundary_weight_key: DualModelDynamicWeightConfig,
                }
                cfg = self.adaptive_preset(
                    input_dim=8,
                    stack_hidden_dim=16,
                    output_dim=4,
                    weight_option_flag=True,
                    weight_option=LowRankDynamicWeightConfig,
                    **boundary_kwargs,
                )
                input_layer_model_config = (
                    cfg.experiment_config.input_model_config.layer_model_config
                )
                output_layer_model_config = (
                    cfg.experiment_config.output_model_config.layer_model_config
                )
                hidden_augmentation_config = self._augmentation_config(cfg)

                self.assertIsInstance(
                    input_layer_model_config,
                    AdaptiveLinearLayerConfig,
                )
                self.assertIsInstance(
                    output_layer_model_config,
                    AdaptiveLinearLayerConfig,
                )
                self.assertIsInstance(
                    hidden_augmentation_config.weight_config,
                    LowRankDynamicWeightConfig,
                )
                if enabled_boundary == "input":
                    self.assertIsInstance(
                        input_layer_model_config.adaptive_augmentation_config.weight_config,
                        DualModelDynamicWeightConfig,
                    )
                    self.assertIsNone(
                        output_layer_model_config.adaptive_augmentation_config.weight_config
                    )
                if enabled_boundary == "output":
                    self.assertIsNone(
                        input_layer_model_config.adaptive_augmentation_config.weight_config
                    )
                    self.assertIsInstance(
                        output_layer_model_config.adaptive_augmentation_config.weight_config,
                        DualModelDynamicWeightConfig,
                    )

                output = Model(cfg)(torch.randn(2, 1, 2, 4))
                logits = output[0] if isinstance(output, tuple) else output
                self.assertEqual(logits.shape, (2, 4))

    def test_boundary_projectors_default_to_empty_adaptive_augmentation(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
        )

        input_layer_model_config = (
            cfg.experiment_config.input_model_config.layer_model_config
        )
        output_layer_model_config = (
            cfg.experiment_config.output_model_config.layer_model_config
        )

        self.assertIsInstance(input_layer_model_config, AdaptiveLinearLayerConfig)
        self.assertIsInstance(output_layer_model_config, AdaptiveLinearLayerConfig)

        for augmentation_config in (
            input_layer_model_config.adaptive_augmentation_config,
            output_layer_model_config.adaptive_augmentation_config,
        ):
            self.assertIsNone(augmentation_config.weight_config)
            self.assertIsNone(augmentation_config.bias_config)
            self.assertIsNone(augmentation_config.diagonal_config)
            self.assertIsNone(augmentation_config.mask_config)

        output = Model(cfg)(torch.randn(2, 1, 2, 4))
        logits = output[0] if isinstance(output, tuple) else output
        self.assertEqual(logits.shape, (2, 4))

    def test_empty_adaptive_boundary_projectors_behave_as_affine_layers(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
        )
        model = Model(cfg)

        for boundary_model, X in (
            (model.input_model.model, torch.randn(2, 8)),
            (model.output_model.model, torch.randn(2, 16)),
        ):
            with self.subTest(boundary_model=boundary_model.__class__.__name__):
                self.assertFalse(boundary_model.has_adaptive_augmentation)
                self.assertIsNone(boundary_model.adaptive_behaviour)

                output = boundary_model(X)
                expected = X @ boundary_model.weight_params
                if boundary_model.bias_params is not None:
                    expected = expected + boundary_model.bias_params

                torch.testing.assert_close(output, expected)

    def test_boundary_adaptive_projectors_inherit_shared_generator_stack(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            stack_bias_flag=True,
            input_layer_weight_option=DualModelDynamicWeightConfig,
            output_layer_weight_option=LowRankDynamicWeightConfig,
            adaptive_generator_stack_hidden_dim=21,
            adaptive_generator_stack_num_layers=3,
            adaptive_generator_stack_activation=ActivationOptions.RELU,
            adaptive_generator_stack_layer_norm_position=(
                LayerNormPositionOptions.AFTER
            ),
            adaptive_generator_stack_dropout_probability=0.2,
            adaptive_generator_stack_last_layer_bias_option=(
                LastLayerBiasOptions.DISABLED
            ),
            adaptive_generator_stack_apply_output_pipeline_flag=True,
            adaptive_generator_stack_bias_flag=False,
        )

        input_layer_model_config = (
            cfg.experiment_config.input_model_config.layer_model_config
        )
        output_layer_model_config = (
            cfg.experiment_config.output_model_config.layer_model_config
        )

        self.assertIsInstance(input_layer_model_config, AdaptiveLinearLayerConfig)
        self.assertIsInstance(output_layer_model_config, AdaptiveLinearLayerConfig)
        self.assertTrue(input_layer_model_config.bias_flag)
        self.assertTrue(output_layer_model_config.bias_flag)
        self.assertIsInstance(
            input_layer_model_config.adaptive_augmentation_config.weight_config,
            DualModelDynamicWeightConfig,
        )
        self.assertIsInstance(
            output_layer_model_config.adaptive_augmentation_config.weight_config,
            LowRankDynamicWeightConfig,
        )
        for layer_model_config in (
            input_layer_model_config,
            output_layer_model_config,
        ):
            augmentation_config = layer_model_config.adaptive_augmentation_config
            self.assertIsNone(augmentation_config.weight_config.model_config)
            self._assert_generator_stack(
                augmentation_config.model_config,
                hidden_dim=21,
                num_layers=3,
                activation=ActivationOptions.RELU,
                layer_norm_position=LayerNormPositionOptions.AFTER,
                dropout_probability=0.2,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_pipeline_flag=True,
                bias_flag=False,
            )

        output = Model(cfg)(torch.randn(2, 1, 2, 4))
        logits = output[0] if isinstance(output, tuple) else output
        self.assertEqual(logits.shape, (2, 4))

    def test_stack_layer_norm_position_is_searchable(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            GridSearch(),
            search_keys=["stack_layer_norm_position"],
        )
        layer_norm_positions = {
            cfg.experiment_config.model_config.layer_config.layer_norm_position
            for cfg in configs
        }

        self.assertEqual(
            len(configs), len(config.SEARCH_SPACE_STACK_LAYER_NORM_POSITION)
        )
        self.assertEqual(
            layer_norm_positions,
            set(config.SEARCH_SPACE_STACK_LAYER_NORM_POSITION),
        )

    def test_weight_generator_depth_search_key_uses_builder_alias(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.DUAL_MODEL_WEIGHT,
            config.DATASET_OPTIONS[0],
            GridSearch(),
            search_keys=["weight_generator_depth"],
        )
        generator_depths = {
            self._augmentation_config(cfg).weight_config.generator_depth
            for cfg in configs
        }

        self.assertEqual(
            len(configs),
            len(config.SEARCH_SPACE_WEIGHT_GENERATOR_DEPTH),
        )
        self.assertEqual(
            generator_depths,
            set(config.SEARCH_SPACE_WEIGHT_GENERATOR_DEPTH),
        )

    def test_single_model_weight_rejects_locked_search_override(self):
        with self.assertRaisesRegex(
            ValueError,
            "SINGLE_MODEL_WEIGHT.*weight_option.*LowRankDynamicWeightConfig",
        ):
            ExperimentPresets().get_config(
                ExperimentPreset.SINGLE_MODEL_WEIGHT,
                config.DATASET_OPTIONS[0],
                GridSearch(),
                search_overrides={"weight_option": [LowRankDynamicWeightConfig]},
            )

    def test_adaptive_sub_configs_match_options(self):
        cfg = ExperimentPresets()._preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            weight_option_flag=True,
            weight_option=DualModelDynamicWeightConfig,
            diagonal_option_flag=True,
            diagonal_option=CombinedDynamicDiagonalConfig,
            bias_option_flag=True,
            bias_option=AdditiveDynamicBiasConfig,
            mask_option_flag=True,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        augmentation_config = self._augmentation_config(cfg)

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

    def test_adaptive_parameter_factory_builds_all_weight_configs(self):
        model_config = self.shared_gate_config().model_config
        cases = [
            (
                SingleModelDynamicWeightConfig,
                {
                    "normalization_option": WeightNormalizationOptions.RMS,
                    "normalization_position_option": (
                        WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
                    ),
                },
            ),
            (
                DualModelDynamicWeightConfig,
                {
                    "normalization_option": WeightNormalizationOptions.RMS,
                    "normalization_position_option": (
                        WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
                    ),
                },
            ),
            (
                LowRankDynamicWeightConfig,
                {"normalization_option": WeightNormalizationOptions.RMS},
            ),
            (
                HypernetworkDynamicWeightConfig,
                {"normalization_option": WeightNormalizationOptions.RMS},
            ),
            (
                LayeredWeightedBankDynamicWeightConfig,
                {
                    "bank_expansion_factor": (
                        BankExpansionFactorOptions.FACTOR_OF_THREE
                    ),
                },
            ),
            (
                SoftWeightedBankDynamicWeightConfig,
                {
                    "bank_expansion_factor": (
                        BankExpansionFactorOptions.FACTOR_OF_THREE
                    ),
                },
            ),
        ]

        for weight_option, expected_values in cases:
            with self.subTest(weight_option=weight_option.__name__):
                weight_config = build_weight_config(
                    weight_option,
                    generator_depth=DynamicDepthOptions.DEPTH_OF_THREE,
                    decay_schedule=WeightDecayScheduleOptions.LINEAR,
                    decay_rate=0.25,
                    decay_warmup_batches=7,
                    normalization_option=WeightNormalizationOptions.RMS,
                    normalization_position_option=(
                        WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
                    ),
                    bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_THREE,
                    model_config=model_config,
                )

                self.assertIsInstance(weight_config, weight_option)
                self.assertEqual(
                    weight_config.generator_depth,
                    DynamicDepthOptions.DEPTH_OF_THREE,
                )
                self.assertEqual(
                    weight_config.decay_schedule,
                    WeightDecayScheduleOptions.LINEAR,
                )
                self.assertEqual(weight_config.decay_rate, 0.25)
                self.assertEqual(weight_config.decay_warmup_batches, 7)
                self.assertIs(weight_config.model_config, model_config)
                for field, expected in expected_values.items():
                    self.assertEqual(getattr(weight_config, field), expected)

    def test_adaptive_parameter_factory_builds_all_bias_configs(self):
        model_config = self.shared_gate_config().model_config
        cases = [
            (AffineTransformDynamicBiasConfig, {}),
            (AdditiveDynamicBiasConfig, {}),
            (GeneratorDynamicBiasConfig, {}),
            (MultiplicativeDynamicBiasConfig, {}),
            (SigmoidGatedDynamicBiasConfig, {}),
            (TanhGatedDynamicBiasConfig, {}),
            (
                WeightedBankDynamicBiasConfig,
                {
                    "bank_expansion_factor": BankExpansionFactorOptions.FACTOR_OF_TWO,
                },
            ),
        ]

        for bias_option, expected_values in cases:
            with self.subTest(bias_option=bias_option.__name__):
                bias_config = build_bias_config(
                    bias_option,
                    decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
                    decay_rate=0.125,
                    decay_warmup_batches=11,
                    bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
                    model_config=model_config,
                )

                self.assertIsInstance(bias_config, bias_option)
                self.assertEqual(
                    bias_config.decay_schedule,
                    WeightDecayScheduleOptions.EXPONENTIAL,
                )
                self.assertEqual(bias_config.decay_rate, 0.125)
                self.assertEqual(bias_config.decay_warmup_batches, 11)
                self.assertIs(bias_config.model_config, model_config)
                for field, expected in expected_values.items():
                    self.assertEqual(getattr(bias_config, field), expected)

    def test_adaptive_parameter_factory_builds_all_diagonal_configs(self):
        model_config = self.shared_gate_config().model_config
        cases = [
            StandardDynamicDiagonalConfig,
            AntiDynamicDiagonalConfig,
            CombinedDynamicDiagonalConfig,
        ]

        for diagonal_option in cases:
            with self.subTest(diagonal_option=diagonal_option.__name__):
                diagonal_config = build_diagonal_config(
                    diagonal_option,
                    model_config=model_config,
                )

                self.assertIsInstance(diagonal_config, diagonal_option)
                self.assertIs(diagonal_config.model_config, model_config)

    def test_adaptive_parameter_factory_builds_all_mask_configs(self):
        model_config = self.shared_gate_config().model_config
        cases = [
            (
                WeightInformedScoreAxisMaskConfig,
                {"mask_dimension_option": MaskDimensionOptions.COLUMN},
            ),
            (
                PerAxisScoreMaskConfig,
                {"mask_dimension_option": MaskDimensionOptions.COLUMN},
            ),
            (
                TopSliceAxisMaskConfig,
                {
                    "mask_dimension_option": MaskDimensionOptions.COLUMN,
                    "mask_transition_width": 0.25,
                },
            ),
            (OuterProductMaskConfig, {}),
            (
                DiagonalAxisMaskConfig,
                {"mask_transition_width": 0.25},
            ),
        ]

        for row_mask_option, expected_values in cases:
            with self.subTest(row_mask_option=row_mask_option.__name__):
                mask_config = build_mask_config(
                    row_mask_option,
                    mask_dimension_option=MaskDimensionOptions.COLUMN,
                    mask_threshold=0.6,
                    mask_surrogate_scale=0.3,
                    mask_floor=0.05,
                    mask_transition_width=0.25,
                    model_config=model_config,
                )

                self.assertIsInstance(mask_config, row_mask_option)
                self.assertEqual(mask_config.mask_threshold, 0.6)
                self.assertEqual(mask_config.mask_surrogate_scale, 0.3)
                self.assertEqual(mask_config.mask_floor, 0.05)
                self.assertIs(mask_config.model_config, model_config)
                for field, expected in expected_values.items():
                    self.assertEqual(getattr(mask_config, field), expected)

    def test_boundary_and_stack_weight_configs_share_constructor_kwargs(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            weight_option_flag=True,
            weight_option=DualModelDynamicWeightConfig,
            generator_depth=DynamicDepthOptions.DEPTH_OF_THREE,
            weight_decay_schedule=WeightDecayScheduleOptions.MULTIPLICATIVE,
            weight_decay_rate=0.125,
            weight_decay_warmup_batches=11,
            weight_normalization_option=WeightNormalizationOptions.RMS,
            weight_normalization_position_option=(
                WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
            ),
            input_layer_weight_option=DualModelDynamicWeightConfig,
            input_layer_weight_generator_depth=DynamicDepthOptions.DEPTH_OF_THREE,
            input_layer_weight_decay_schedule=(
                WeightDecayScheduleOptions.MULTIPLICATIVE
            ),
            input_layer_weight_decay_rate=0.125,
            input_layer_weight_decay_warmup_batches=11,
            input_layer_weight_normalization_option=WeightNormalizationOptions.RMS,
            input_layer_weight_normalization_position_option=(
                WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
            ),
            output_layer_weight_option=DualModelDynamicWeightConfig,
            output_layer_weight_generator_depth=DynamicDepthOptions.DEPTH_OF_THREE,
            output_layer_weight_decay_schedule=(
                WeightDecayScheduleOptions.MULTIPLICATIVE
            ),
            output_layer_weight_decay_rate=0.125,
            output_layer_weight_decay_warmup_batches=11,
            output_layer_weight_normalization_option=WeightNormalizationOptions.RMS,
            output_layer_weight_normalization_position_option=(
                WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
            ),
        )
        expected_values = {
            "generator_depth": DynamicDepthOptions.DEPTH_OF_THREE,
            "decay_schedule": WeightDecayScheduleOptions.MULTIPLICATIVE,
            "decay_rate": 0.125,
            "decay_warmup_batches": 11,
            "normalization_option": WeightNormalizationOptions.RMS,
            "normalization_position_option": (
                WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
            ),
        }
        weight_configs = [
            self._augmentation_config(cfg).weight_config,
            (
                cfg.experiment_config.input_model_config.layer_model_config
                .adaptive_augmentation_config.weight_config
            ),
            (
                cfg.experiment_config.output_model_config.layer_model_config
                .adaptive_augmentation_config.weight_config
            ),
        ]

        for weight_config in weight_configs:
            with self.subTest(weight_config=weight_config):
                self.assertIsInstance(weight_config, DualModelDynamicWeightConfig)
                for field, expected in expected_values.items():
                    self.assertEqual(getattr(weight_config, field), expected)

    def test_default_adaptive_component_flags_disable_components(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
        )
        augmentation_config = self._augmentation_config(cfg)

        self.assertIsNone(augmentation_config.weight_config)
        self.assertIsNone(augmentation_config.bias_config)
        self.assertIsNone(augmentation_config.diagonal_config)
        self.assertIsNone(augmentation_config.mask_config)

    def test_adaptive_component_options_without_flags_are_ignored(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
        )
        augmentation_config = self._augmentation_config(cfg)

        self.assertIsNone(augmentation_config.weight_config)
        self.assertIsNone(augmentation_config.bias_config)
        self.assertIsNone(augmentation_config.diagonal_config)
        self.assertIsNone(augmentation_config.mask_config)

    def test_adaptive_component_flags_require_options(self):
        cases = [
            ("weight_option_flag", "weight_option", {"weight_option_flag": True}),
            ("bias_option_flag", "bias_option", {"bias_option_flag": True}),
            (
                "diagonal_option_flag",
                "diagonal_option",
                {"diagonal_option_flag": True},
            ),
            (
                "mask_option_flag",
                "row_mask_option",
                {"mask_option_flag": True},
            ),
        ]

        for flag_key, option_key, kwargs in cases:
            with self.subTest(flag_key=flag_key):
                with self.assertRaisesRegex(ValueError, f"{option_key}.*{flag_key}"):
                    self.adaptive_preset(
                        input_dim=8,
                        stack_hidden_dim=16,
                        output_dim=4,
                        **kwargs,
                    )

    def test_adaptive_component_flags_build_default_concrete_options(self):
        cases = [
            (
                "weight",
                {
                    "weight_option_flag": True,
                    "weight_option": SingleModelDynamicWeightConfig,
                },
                "weight_config",
                SingleModelDynamicWeightConfig,
            ),
            (
                "bias",
                {
                    "bias_option_flag": True,
                    "bias_option": AffineTransformDynamicBiasConfig,
                },
                "bias_config",
                AffineTransformDynamicBiasConfig,
            ),
            (
                "diagonal",
                {
                    "diagonal_option_flag": True,
                    "diagonal_option": StandardDynamicDiagonalConfig,
                },
                "diagonal_config",
                StandardDynamicDiagonalConfig,
            ),
            (
                "mask",
                {
                    "mask_option_flag": True,
                    "row_mask_option": DiagonalAxisMaskConfig,
                },
                "mask_config",
                DiagonalAxisMaskConfig,
            ),
        ]

        for component, kwargs, attribute, expected_type in cases:
            with self.subTest(component=component):
                cfg = self.adaptive_preset(
                    input_dim=8,
                    stack_hidden_dim=16,
                    output_dim=4,
                    **kwargs,
                )
                augmentation_config = self._augmentation_config(cfg)

                self.assertIsInstance(
                    getattr(augmentation_config, attribute),
                    expected_type,
                )

    def test_adaptive_component_stacks_inherit_shared_generator_by_default(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            weight_option_flag=True,
            weight_option=DualModelDynamicWeightConfig,
            bias_option_flag=True,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option_flag=True,
            diagonal_option=CombinedDynamicDiagonalConfig,
            mask_option_flag=True,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            adaptive_generator_stack_hidden_dim=21,
            adaptive_generator_stack_num_layers=3,
            adaptive_generator_stack_activation=ActivationOptions.RELU,
            adaptive_generator_stack_layer_norm_position=(
                LayerNormPositionOptions.AFTER
            ),
            adaptive_generator_stack_dropout_probability=0.2,
            adaptive_generator_stack_last_layer_bias_option=(
                LastLayerBiasOptions.DISABLED
            ),
            adaptive_generator_stack_apply_output_pipeline_flag=True,
            adaptive_generator_stack_bias_flag=False,
        )

        augmentation_config = self._augmentation_config(cfg)

        self._assert_generator_stack(
            augmentation_config.model_config,
            hidden_dim=21,
            num_layers=3,
            activation=ActivationOptions.RELU,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            dropout_probability=0.2,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=True,
            bias_flag=False,
        )
        for component_config in (
            augmentation_config.weight_config,
            augmentation_config.bias_config,
            augmentation_config.diagonal_config,
            augmentation_config.mask_config,
        ):
            self.assertIsNone(component_config.model_config)

    def test_adaptive_component_stack_overrides_require_independent_flags(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            weight_option_flag=True,
            weight_option=DualModelDynamicWeightConfig,
            bias_option_flag=True,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option_flag=True,
            diagonal_option=CombinedDynamicDiagonalConfig,
            mask_option_flag=True,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            adaptive_generator_stack_hidden_dim=21,
            weight_generator_stack_hidden_dim=31,
            bias_generator_stack_hidden_dim=41,
            diagonal_generator_stack_hidden_dim=51,
            mask_generator_stack_hidden_dim=61,
            weight_generator_stack_activation=ActivationOptions.SILU,
            bias_generator_stack_activation=ActivationOptions.MISH,
            diagonal_generator_stack_activation=ActivationOptions.TANH,
            mask_generator_stack_activation=ActivationOptions.RELU,
        )

        augmentation_config = self._augmentation_config(cfg)

        self.assertEqual(augmentation_config.model_config.hidden_dim, 21)
        for component_config in (
            augmentation_config.weight_config,
            augmentation_config.bias_config,
            augmentation_config.diagonal_config,
            augmentation_config.mask_config,
        ):
            self.assertIsNone(component_config.model_config)

    def test_adaptive_component_stack_overrides_apply_when_independent(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            weight_option_flag=True,
            weight_option=DualModelDynamicWeightConfig,
            bias_option_flag=True,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option_flag=True,
            diagonal_option=CombinedDynamicDiagonalConfig,
            mask_option_flag=True,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            weight_generator_stack_independent_flag=True,
            weight_generator_stack_hidden_dim=31,
            weight_generator_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            weight_generator_stack_num_layers=4,
            weight_generator_stack_activation=ActivationOptions.SILU,
            weight_generator_stack_residual_connection_option=(
                ResidualConnectionOptions.DISABLED
            ),
            weight_generator_stack_dropout_probability=0.15,
            weight_generator_stack_last_layer_bias_option=(
                LastLayerBiasOptions.DISABLED
            ),
            weight_generator_stack_apply_output_pipeline_flag=True,
            weight_generator_stack_bias_flag=False,
            bias_generator_stack_independent_flag=True,
            bias_generator_stack_hidden_dim=41,
            bias_generator_stack_activation=ActivationOptions.MISH,
            diagonal_generator_stack_independent_flag=True,
            diagonal_generator_stack_hidden_dim=51,
            diagonal_generator_stack_activation=ActivationOptions.TANH,
            mask_generator_stack_independent_flag=True,
            mask_generator_stack_hidden_dim=61,
            mask_generator_stack_activation=ActivationOptions.RELU,
        )

        augmentation_config = self._augmentation_config(cfg)

        self._assert_generator_stack(
            augmentation_config.weight_config.model_config,
            hidden_dim=31,
            num_layers=4,
            activation=ActivationOptions.SILU,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            dropout_probability=0.15,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=True,
            bias_flag=False,
        )
        self.assertEqual(augmentation_config.bias_config.model_config.hidden_dim, 41)
        self.assertEqual(
            augmentation_config.bias_config.model_config.layer_config.activation,
            ActivationOptions.MISH,
        )
        self.assertEqual(
            augmentation_config.diagonal_config.model_config.hidden_dim,
            51,
        )
        self.assertEqual(
            augmentation_config.diagonal_config.model_config.layer_config.activation,
            ActivationOptions.TANH,
        )
        self.assertEqual(augmentation_config.mask_config.model_config.hidden_dim, 61)
        self.assertEqual(
            augmentation_config.mask_config.model_config.layer_config.activation,
            ActivationOptions.RELU,
        )

    def test_partial_adaptive_component_stack_overrides_fall_back_to_shared(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            bias_option_flag=True,
            bias_option=AdditiveDynamicBiasConfig,
            adaptive_generator_stack_hidden_dim=21,
            adaptive_generator_stack_num_layers=3,
            adaptive_generator_stack_activation=ActivationOptions.RELU,
            adaptive_generator_stack_layer_norm_position=(
                LayerNormPositionOptions.AFTER
            ),
            adaptive_generator_stack_dropout_probability=0.2,
            adaptive_generator_stack_last_layer_bias_option=(
                LastLayerBiasOptions.DISABLED
            ),
            adaptive_generator_stack_apply_output_pipeline_flag=True,
            adaptive_generator_stack_bias_flag=False,
            bias_generator_stack_independent_flag=True,
            bias_generator_stack_hidden_dim=41,
        )

        bias_stack = self._augmentation_config(cfg).bias_config.model_config

        self._assert_generator_stack(
            bias_stack,
            hidden_dim=41,
            num_layers=3,
            activation=ActivationOptions.RELU,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            dropout_probability=0.2,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=True,
            bias_flag=False,
        )

    def test_all_adaptive_component_stacks_can_forward_one_fake_batch(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=8,
            output_dim=4,
            stack_num_layers=2,
            weight_option_flag=True,
            weight_option=DualModelDynamicWeightConfig,
            bias_option_flag=True,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option_flag=True,
            diagonal_option=CombinedDynamicDiagonalConfig,
            mask_option_flag=True,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            weight_generator_stack_independent_flag=True,
            weight_generator_stack_hidden_dim=17,
            bias_generator_stack_independent_flag=True,
            bias_generator_stack_hidden_dim=19,
            diagonal_generator_stack_independent_flag=True,
            diagonal_generator_stack_hidden_dim=23,
            mask_generator_stack_independent_flag=True,
            mask_generator_stack_hidden_dim=29,
        )
        model = Model(cfg)

        logits = model(torch.randn(2, 1, 2, 4))

        self.assertEqual(logits.shape, (2, 4))

    def test_disabled_adaptive_component_options_ignore_independent_stacks(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=16,
            output_dim=4,
            weight_generator_stack_independent_flag=True,
            weight_generator_stack_hidden_dim=31,
            bias_generator_stack_independent_flag=True,
            bias_generator_stack_hidden_dim=41,
            diagonal_generator_stack_independent_flag=True,
            diagonal_generator_stack_hidden_dim=51,
            mask_generator_stack_independent_flag=True,
            mask_generator_stack_hidden_dim=61,
        )

        augmentation_config = self._augmentation_config(cfg)

        self.assertIsNone(augmentation_config.weight_config)
        self.assertIsNone(augmentation_config.bias_config)
        self.assertIsNone(augmentation_config.diagonal_config)
        self.assertIsNone(augmentation_config.mask_config)

    def test_individual_adaptive_presets_wire_expected_config_type(self):
        cases_by_family = {
            "weight": [
                (
                    ExperimentPreset.SINGLE_MODEL_WEIGHT,
                    SingleModelDynamicWeightConfig,
                ),
                (
                    ExperimentPreset.DUAL_MODEL_WEIGHT,
                    DualModelDynamicWeightConfig,
                ),
                (
                    ExperimentPreset.LOW_RANK_WEIGHT,
                    LowRankDynamicWeightConfig,
                ),
                (
                    ExperimentPreset.HYPERNETWORK_WEIGHT,
                    HypernetworkDynamicWeightConfig,
                ),
                (
                    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT,
                    LayeredWeightedBankDynamicWeightConfig,
                ),
                (
                    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT,
                    SoftWeightedBankDynamicWeightConfig,
                ),
            ],
            "bias": [
                (
                    ExperimentPreset.AFFINE_TRANSFORM_BIAS,
                    AffineTransformDynamicBiasConfig,
                ),
                (ExperimentPreset.ADDITIVE_BIAS, AdditiveDynamicBiasConfig),
                (ExperimentPreset.GENERATOR_BIAS, GeneratorDynamicBiasConfig),
                (
                    ExperimentPreset.MULTIPLICATIVE_BIAS,
                    MultiplicativeDynamicBiasConfig,
                ),
                (
                    ExperimentPreset.SIGMOID_GATED_BIAS,
                    SigmoidGatedDynamicBiasConfig,
                ),
                (ExperimentPreset.TANH_GATED_BIAS, TanhGatedDynamicBiasConfig),
                (ExperimentPreset.WEIGHTED_BANK_BIAS, WeightedBankDynamicBiasConfig),
            ],
            "diagonal": [
                (
                    ExperimentPreset.STANDARD_DIAGONAL,
                    StandardDynamicDiagonalConfig,
                ),
                (ExperimentPreset.ANTI_DIAGONAL, AntiDynamicDiagonalConfig),
                (
                    ExperimentPreset.COMBINED_DIAGONAL,
                    CombinedDynamicDiagonalConfig,
                ),
            ],
            "mask": [
                (ExperimentPreset.DIAGONAL_AXIS_MASK, DiagonalAxisMaskConfig),
                (ExperimentPreset.OUTER_PRODUCT_MASK, OuterProductMaskConfig),
                (ExperimentPreset.PER_AXIS_SCORE_MASK, PerAxisScoreMaskConfig),
                (ExperimentPreset.TOP_SLICE_AXIS_MASK, TopSliceAxisMaskConfig),
                (
                    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK,
                    WeightInformedScoreAxisMaskConfig,
                ),
            ],
        }

        for family, cases in cases_by_family.items():
            expected_field = f"{family}_config"
            for preset, expected_type in cases:
                with self.subTest(
                    family=family,
                    preset=preset.name,
                    expected_config_role=expected_field,
                ):
                    cfg = ExperimentPresets().get_config(preset)[0]
                    augmentation_config = self._augmentation_config(cfg)

                    for field in (
                        "weight_config",
                        "bias_config",
                        "diagonal_config",
                        "mask_config",
                    ):
                        with self.subTest(config_role=field):
                            value = getattr(augmentation_config, field)
                            if field == expected_field:
                                self.assertIsInstance(value, expected_type)
                            else:
                                self.assertIsNone(value)

    def test_specialized_weight_presets_wire_expected_knobs(self):
        cases = [
            (
                ExperimentPreset.DECAY_EXPONENTIAL_WEIGHT,
                {
                    "decay_schedule": WeightDecayScheduleOptions.EXPONENTIAL,
                    "decay_rate": 1e-3,
                    "decay_warmup_batches": 500,
                },
            ),
            (
                ExperimentPreset.NORM_L2_WEIGHT,
                {
                    "normalization_option": WeightNormalizationOptions.L2_SCALE,
                },
            ),
            (
                ExperimentPreset.DEEP_GENERATOR,
                {
                    "generator_depth": DynamicDepthOptions.DEPTH_OF_EIGHT,
                },
            ),
        ]

        for preset, expected_values in cases:
            with self.subTest(preset=preset.name):
                cfg = ExperimentPresets().get_config(preset)[0]
                weight_config = self._augmentation_config(cfg).weight_config

                self.assertIsInstance(weight_config, DualModelDynamicWeightConfig)
                for field, expected in expected_values.items():
                    self.assertEqual(getattr(weight_config, field), expected)

    def test_preset_locks_are_exposed_with_reasons(self):
        presets = ExperimentPresets()

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                expected_locks = presets.locks_for_preset(preset)
                locks = presets.locked_fields(preset)

                self.assertEqual(set(locks), set(expected_locks))
                self.assertEqual(set(locks), set(presets.overrides_for_preset(preset)))
                for field, lock in locks.items():
                    expected = expected_locks[field]
                    expected_value = (
                        expected.value if isinstance(expected, PresetLock) else expected
                    )
                    self.assertEqual(lock.value, expected_value)
                    self.assertIn(preset.name, lock.reason)

    def test_preset_definitions_keep_values_and_descriptions_together(self):
        presets = ExperimentPresets()

        self.assertNotIsInstance(ExperimentPreset.BASELINE.value, str)
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, len(ExperimentPreset) + 1)),
        )
        self.assertEqual(set(_PRESET_DEFINITIONS), set(ExperimentPreset))
        for preset, definition in _PRESET_DEFINITIONS.items():
            with self.subTest(preset=preset.name):
                self.assertIsInstance(definition, PresetDefinition)
                self.assertEqual(presets.definition_for_preset(preset), definition)
                self.assertEqual(
                    presets.overrides_for_preset(preset),
                    definition.preset_values,
                )
                self.assertEqual(
                    presets.description_for_preset(preset),
                    definition.description,
                )
                self.assertIsInstance(definition.description, str)
                self.assertTrue(definition.description)

    def test_experiment_presets_do_not_alias_names(self):
        self.assertEqual(
            len(ExperimentPreset.__members__),
            len(set(ExperimentPreset.__members__.values())),
        )
        self.assertNotIn("ADAPTIVE_HALTING", ExperimentPreset.__members__)
        self.assertIn("DUAL_WEIGHT_HALTING", ExperimentPreset.names())

    def test_adaptive_option_preset_locks_include_component_flags(self):
        presets = ExperimentPresets()
        option_flags = {
            "weight_option": "weight_option_flag",
            "bias_option": "bias_option_flag",
            "diagonal_option": "diagonal_option_flag",
            "row_mask_option": "mask_option_flag",
        }

        for preset in ExperimentPreset:
            overrides = presets.overrides_for_preset(preset)
            locks = presets.locked_fields(preset)
            for option_key, flag_key in option_flags.items():
                if option_key not in overrides:
                    continue
                with self.subTest(preset=preset.name, option_key=option_key):
                    self.assertIn(flag_key, overrides)
                    self.assertTrue(overrides[flag_key])
                    self.assertIn(flag_key, locks)
                    self.assertTrue(locks[flag_key].value)
                    self.assertIn(preset.name, locks[flag_key].reason)

    def test_weight_bias_diagonal_presets_do_not_include_masks(self):
        expected_configs = {
            ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
                SingleModelDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                CombinedDynamicDiagonalConfig,
            ),
            ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
                DualModelDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                CombinedDynamicDiagonalConfig,
            ),
            (
                ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL
            ): (
                LayeredWeightedBankDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                CombinedDynamicDiagonalConfig,
            ),
            ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
                LowRankDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                CombinedDynamicDiagonalConfig,
            ),
            ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
                SingleModelDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                StandardDynamicDiagonalConfig,
            ),
            ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
                DualModelDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                StandardDynamicDiagonalConfig,
            ),
            (
                ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL
            ): (
                LayeredWeightedBankDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                StandardDynamicDiagonalConfig,
            ),
            ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
                LowRankDynamicWeightConfig,
                AdditiveDynamicBiasConfig,
                StandardDynamicDiagonalConfig,
            ),
        }

        for preset, expected_types in expected_configs.items():
            with self.subTest(preset=preset.name):
                cfg = ExperimentPresets().get_config(preset)[0]
                augmentation_config = self._augmentation_config(cfg)

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
        cfg = self.adaptive_preset(
            stack_gate_flag=True,
            gate_option=LayerGateOptions.MULTIPLIER,
            gate_stack_independent_flag=True,
            gate_stack_hidden_dim=32,
            gate_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            gate_stack_num_layers=3,
            gate_stack_activation=ActivationOptions.SILU,
            gate_stack_residual_connection_option=ResidualConnectionOptions.DISABLED,
            gate_stack_dropout_probability=0.1,
            gate_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            gate_stack_apply_output_pipeline_flag=True,
            gate_stack_bias_flag=False,
        )

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
        cfg = self.adaptive_preset(shared_gate_config=shared_gate_config)
        model_cfg = cfg.experiment_config.model_config

        self.assertIs(model_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(model_cfg.layer_config.gate_config)

    def test_shared_gate_config_rejects_enabled_stack_gate(self):
        cfg = self.adaptive_preset(
            stack_gate_flag=True,
            shared_gate_config=self.shared_gate_config(),
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Model(cfg)

    def test_shared_gate_config_allows_absent_stack_gate(self):
        shared_gate_config = self.shared_gate_config()
        cfg = self.adaptive_preset(
            stack_gate_flag=False,
            shared_gate_config=shared_gate_config,
        )
        model_cfg = cfg.experiment_config.model_config

        self.assertIs(model_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(model_cfg.layer_config.gate_config)

    def test_halting_config_uses_builder_overrides(self):
        cfg = self.adaptive_preset(
            stack_halting_flag=True,
            halting_threshold=0.5,
            halting_dropout=0.2,
            halting_stack_independent_flag=True,
            halting_stack_hidden_dim=48,
            halting_stack_layer_norm_position=LayerNormPositionOptions.BEFORE,
            halting_stack_num_layers=5,
            halting_stack_activation=ActivationOptions.MISH,
            halting_stack_residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_stack_dropout_probability=0.3,
            halting_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            halting_stack_apply_output_pipeline_flag=True,
            halting_stack_bias_flag=False,
        )

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

    def test_controller_stack_defaults_use_submodule_stack_options(self):
        cfg = self.adaptive_preset(
            stack_gate_flag=True,
            stack_halting_flag=True,
            memory_flag=True,
        )

        model_cfg = cfg.experiment_config.model_config
        expected_last_layer_bias_options = {
            "gate": LastLayerBiasOptions.DEFAULT,
            "halting": LastLayerBiasOptions.DISABLED,
            "memory": LastLayerBiasOptions.DEFAULT,
        }

        for role, stack_cfg in self._controller_stack_configs(model_cfg).items():
            with self.subTest(config_role=role):
                self._assert_controller_stack_options(
                    stack_cfg,
                    hidden_dim=config.STACK_HIDDEN_DIM,
                    num_layers=2,
                    activation=ActivationOptions.GELU,
                    layer_norm_position=config.STACK_LAYER_NORM_POSITION,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
                    dropout_probability=0.0,
                    last_layer_bias_option=expected_last_layer_bias_options[role],
                    apply_output_pipeline_flag=False,
                    bias_flag=True,
                )

    def test_controller_stacks_inherit_submodule_defaults_when_overrides_are_none(
        self,
    ):
        cfg = self.adaptive_preset(
            stack_gate_flag=True,
            stack_halting_flag=True,
            memory_flag=True,
            submodule_stack_hidden_dim=37,
            submodule_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            submodule_stack_num_layers=4,
            submodule_stack_activation=ActivationOptions.MISH,
            submodule_stack_residual_connection_option=(
                ResidualConnectionOptions.RESIDUAL
            ),
            submodule_stack_dropout_probability=0.12,
            submodule_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            submodule_stack_apply_output_pipeline_flag=True,
            submodule_stack_bias_flag=False,
            gate_stack_activation=None,
            gate_stack_apply_output_pipeline_flag=None,
            gate_stack_bias_flag=None,
            halting_stack_layer_norm_position=None,
            halting_stack_last_layer_bias_option=None,
        )

        model_cfg = cfg.experiment_config.model_config

        for role, stack_cfg in self._controller_stack_configs(model_cfg).items():
            with self.subTest(config_role=role):
                self._assert_controller_stack_options(
                    stack_cfg,
                    hidden_dim=37,
                    num_layers=4,
                    activation=ActivationOptions.MISH,
                    layer_norm_position=LayerNormPositionOptions.AFTER,
                    residual_connection_option=ResidualConnectionOptions.RESIDUAL,
                    dropout_probability=0.12,
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                    apply_output_pipeline_flag=True,
                    bias_flag=False,
                )

    def test_controller_stack_overrides_require_independent_flags(self):
        cfg = self.adaptive_preset(
            stack_gate_flag=True,
            stack_halting_flag=True,
            memory_flag=True,
            submodule_stack_hidden_dim=11,
            submodule_stack_activation=ActivationOptions.RELU,
            gate_stack_hidden_dim=22,
            gate_stack_activation=ActivationOptions.SILU,
            halting_stack_hidden_dim=33,
            halting_stack_activation=ActivationOptions.MISH,
            memory_stack_hidden_dim=44,
            memory_stack_activation=ActivationOptions.TANH,
        )

        model_cfg = cfg.experiment_config.model_config

        self.assertEqual(model_cfg.layer_config.gate_config.model_config.hidden_dim, 11)
        self.assertEqual(
            model_cfg.layer_config.gate_config.model_config.layer_config.activation,
            ActivationOptions.RELU,
        )
        self.assertEqual(
            model_cfg.layer_config.halting_config.halting_gate_config.hidden_dim,
            11,
        )
        self.assertEqual(
            model_cfg.layer_config.halting_config.halting_gate_config.layer_config.activation,
            ActivationOptions.RELU,
        )
        self.assertEqual(model_cfg.shared_memory_config.model_config.hidden_dim, 11)
        self.assertEqual(
            model_cfg.shared_memory_config.model_config.layer_config.activation,
            ActivationOptions.RELU,
        )

    def test_controller_stack_independent_flags_enable_controller_options(self):
        cfg = self.adaptive_preset(
            stack_gate_flag=True,
            stack_halting_flag=True,
            memory_flag=True,
            submodule_stack_hidden_dim=11,
            submodule_stack_activation=ActivationOptions.RELU,
            gate_stack_independent_flag=True,
            gate_stack_hidden_dim=22,
            gate_stack_activation=ActivationOptions.SILU,
            halting_stack_independent_flag=True,
            halting_stack_hidden_dim=33,
            halting_stack_activation=ActivationOptions.MISH,
            memory_stack_independent_flag=True,
            memory_stack_hidden_dim=44,
            memory_stack_activation=ActivationOptions.TANH,
        )

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
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=8,
            output_dim=4,
            memory_flag=True,
        )

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
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=8,
            output_dim=4,
            memory_flag=True,
            memory_option=WeightedDynamicMemoryConfig,
            memory_position_option=MemoryPositionOptions.BEFORE_AFFINE,
            memory_test_time_training_learning_rate=0.02,
            memory_test_time_training_num_inner_steps=2,
            memory_stack_independent_flag=True,
            memory_stack_hidden_dim=12,
            memory_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            memory_stack_num_layers=3,
            memory_stack_activation=ActivationOptions.SILU,
            memory_stack_residual_connection_option=(
                ResidualConnectionOptions.DISABLED
            ),
            memory_stack_dropout_probability=0.1,
            memory_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            memory_stack_apply_output_pipeline_flag=True,
            memory_stack_bias_flag=False,
        )

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
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=8,
            output_dim=4,
            stack_num_layers=2,
            memory_flag=True,
        )
        model = Model(cfg)

        logits = model(torch.randn(2, 1, 2, 4))

        self.assertEqual(logits.shape, (2, 4))

    def test_memory_enabled_backward_produces_memory_gradients(self):
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=8,
            output_dim=4,
            stack_num_layers=2,
            memory_flag=True,
        )
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
        cfg = self.adaptive_preset(
            input_dim=8,
            stack_hidden_dim=8,
            output_dim=4,
            recurrent_flag=True,
            memory_flag=True,
        )

        recurrent_cfg = cfg.experiment_config.model_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertIsNotNone(recurrent_cfg.block_config.shared_memory_config)
        self.assertIsNone(recurrent_cfg.memory_config)

    def test_recurrent_layer_norm_position_defaults_disabled_and_uses_override(self):
        default_cfg = self.adaptive_preset(recurrent_flag=True)
        default_recurrent_cfg = default_cfg.experiment_config.model_config

        self.assertIsInstance(default_recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            default_recurrent_cfg.recurrent_layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )

        cfg = self.adaptive_preset(
            recurrent_flag=True,
            recurrent_layer_norm_position=LayerNormPositionOptions.BEFORE,
        )
        recurrent_cfg = cfg.experiment_config.model_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            recurrent_cfg.recurrent_layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )

    def test_recurrent_controllers_use_resolved_gate_and_halting_builders(self):
        cfg = self.adaptive_preset(
            recurrent_flag=True,
            recurrent_gate_flag=True,
            recurrent_gate_option=LayerGateOptions.MULTIPLIER,
            recurrent_halting_flag=True,
            submodule_stack_hidden_dim=29,
            submodule_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            submodule_stack_num_layers=3,
            submodule_stack_activation=ActivationOptions.MISH,
            submodule_stack_residual_connection_option=(
                ResidualConnectionOptions.RESIDUAL
            ),
            submodule_stack_dropout_probability=0.18,
            submodule_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            submodule_stack_apply_output_pipeline_flag=True,
            submodule_stack_bias_flag=False,
            gate_stack_activation=None,
            gate_stack_apply_output_pipeline_flag=None,
            gate_stack_bias_flag=None,
            halting_stack_layer_norm_position=None,
            halting_stack_last_layer_bias_option=None,
        )

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
                self.assertFalse(stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_recurrent_controller_stack_overrides_require_independent_flags(self):
        cfg = self.adaptive_preset(
            recurrent_flag=True,
            recurrent_gate_flag=True,
            recurrent_halting_flag=True,
            submodule_stack_hidden_dim=19,
            submodule_stack_activation=ActivationOptions.RELU,
            recurrent_gate_stack_hidden_dim=64,
            recurrent_gate_stack_activation=ActivationOptions.SILU,
            recurrent_halting_stack_hidden_dim=72,
            recurrent_halting_stack_activation=ActivationOptions.MISH,
        )

        recurrent_cfg = cfg.experiment_config.model_config
        gate_cfg = recurrent_cfg.gate_config.model_config
        halting_stack_cfg = recurrent_cfg.halting_config.halting_gate_config

        self.assertEqual(gate_cfg.hidden_dim, 19)
        self.assertEqual(gate_cfg.layer_config.activation, ActivationOptions.RELU)
        self.assertEqual(halting_stack_cfg.hidden_dim, 19)
        self.assertEqual(
            halting_stack_cfg.layer_config.activation,
            ActivationOptions.RELU,
        )

    def test_recurrent_gate_config_uses_recurrent_overrides(self):
        cfg = self.adaptive_preset(
            recurrent_flag=True,
            recurrent_gate_flag=True,
            recurrent_gate_option=LayerGateOptions.MULTIPLIER,
            recurrent_gate_stack_independent_flag=True,
            recurrent_gate_stack_hidden_dim=64,
            recurrent_gate_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            recurrent_gate_stack_num_layers=4,
            recurrent_gate_stack_activation=ActivationOptions.SILU,
            recurrent_gate_stack_residual_connection_option=(
                ResidualConnectionOptions.DISABLED
            ),
            recurrent_gate_stack_dropout_probability=0.15,
            recurrent_gate_stack_last_layer_bias_option=(LastLayerBiasOptions.DISABLED),
            recurrent_gate_stack_apply_output_pipeline_flag=False,
            recurrent_gate_stack_bias_flag=False,
        )

        recurrent_cfg = cfg.experiment_config.model_config
        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(recurrent_cfg.gate_config.option, LayerGateOptions.MULTIPLIER)
        self.assertIsNone(recurrent_cfg.block_config.layer_config.gate_config)

        gate_cfg = recurrent_cfg.gate_config.model_config
        self.assertEqual(gate_cfg.hidden_dim, 64)
        self.assertEqual(gate_cfg.num_layers, 4)
        self.assertEqual(gate_cfg.last_layer_bias_option, LastLayerBiasOptions.DISABLED)
        self.assertFalse(gate_cfg.apply_output_pipeline_flag)
        self.assertEqual(
            gate_cfg.layer_config.layer_norm_position, LayerNormPositionOptions.AFTER
        )
        self.assertEqual(gate_cfg.layer_config.activation, ActivationOptions.SILU)
        self.assertEqual(
            gate_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(gate_cfg.layer_config.dropout_probability, 0.15)
        self.assertFalse(gate_cfg.layer_config.layer_model_config.bias_flag)

    def test_recurrent_halting_config_uses_recurrent_overrides(self):
        cfg = self.adaptive_preset(
            output_dim=13,
            recurrent_flag=True,
            recurrent_halting_flag=True,
            recurrent_halting_threshold=0.65,
            recurrent_halting_dropout=0.25,
            recurrent_halting_hidden_state_mode=(
                HaltingHiddenStateModeOptions.ACCUMULATED
            ),
            recurrent_halting_stack_independent_flag=True,
            recurrent_halting_stack_hidden_dim=72,
            recurrent_halting_stack_layer_norm_position=LayerNormPositionOptions.BEFORE,
            recurrent_halting_stack_num_layers=4,
            recurrent_halting_stack_activation=ActivationOptions.MISH,
            recurrent_halting_stack_residual_connection_option=(
                ResidualConnectionOptions.DISABLED
            ),
            recurrent_halting_stack_dropout_probability=0.2,
            recurrent_halting_stack_last_layer_bias_option=(
                LastLayerBiasOptions.DISABLED
            ),
            recurrent_halting_stack_apply_output_pipeline_flag=True,
            recurrent_halting_stack_bias_flag=False,
        )

        recurrent_cfg = cfg.experiment_config.model_config
        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        halting_cfg = recurrent_cfg.halting_config
        halting_stack_cfg = halting_cfg.halting_gate_config

        self.assertEqual(halting_cfg.threshold, 0.65)
        self.assertEqual(halting_cfg.halting_dropout, 0.25)
        self.assertEqual(
            halting_cfg.hidden_state_mode,
            HaltingHiddenStateModeOptions.ACCUMULATED,
        )
        self.assertEqual(halting_stack_cfg.hidden_dim, 72)
        self.assertEqual(halting_stack_cfg.output_dim, 2)
        self.assertEqual(halting_stack_cfg.num_layers, 4)
        self.assertEqual(
            halting_stack_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertTrue(halting_stack_cfg.apply_output_pipeline_flag)
        self.assertEqual(
            halting_stack_cfg.layer_config.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertEqual(
            halting_stack_cfg.layer_config.activation,
            ActivationOptions.MISH,
        )
        self.assertEqual(
            halting_stack_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(halting_stack_cfg.layer_config.dropout_probability, 0.2)
        self.assertFalse(halting_stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_recurrent_presets_wire_optional_controllers(self):
        expected_controllers = {
            ExperimentPreset.RECURRENT: (False, False, False),
            ExperimentPreset.RECURRENT_GATING: (True, False, False),
            ExperimentPreset.RECURRENT_HALTING: (False, True, False),
            ExperimentPreset.RECURRENT_MEMORY: (False, False, True),
            ExperimentPreset.RECURRENT_GATING_HALTING: (True, True, False),
            ExperimentPreset.RECURRENT_GATING_MEMORY: (True, False, True),
            ExperimentPreset.RECURRENT_HALTING_MEMORY: (False, True, True),
            ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: (True, True, True),
            ExperimentPreset.RECURRENT_RESIDUAL: (False, False, False),
            ExperimentPreset.RECURRENT_POST_NORM: (False, False, False),
            ExperimentPreset.FULL_STACK_RECURRENT: (False, False, False),
        }

        for preset, (
            expected_gate,
            expected_halting,
            expected_memory,
        ) in expected_controllers.items():
            with self.subTest(preset=preset.name):
                cfg = ExperimentPresets().get_config(preset)[0]
                recurrent_cfg = cfg.experiment_config.model_config

                self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
                self.assertEqual(recurrent_cfg.gate_config is not None, expected_gate)
                self.assertEqual(
                    recurrent_cfg.halting_config is not None,
                    expected_halting,
                )
                self.assertEqual(
                    recurrent_cfg.block_config.shared_memory_config is not None,
                    expected_memory,
                )

    def test_linear_parity_presets_build_valid_configs(self):
        parity_presets = [
            ExperimentPreset.GATING,
            ExperimentPreset.HALTING,
            ExperimentPreset.MEMORY,
            ExperimentPreset.GATING_HALTING,
            ExperimentPreset.GATING_MEMORY,
            ExperimentPreset.HALTING_MEMORY,
            ExperimentPreset.GATING_HALTING_MEMORY,
            ExperimentPreset.RESIDUAL,
            ExperimentPreset.POST_NORM,
            ExperimentPreset.RESIDUAL_POST_NORM,
            ExperimentPreset.RESIDUAL_GATING,
            ExperimentPreset.RESIDUAL_HALTING,
            ExperimentPreset.RESIDUAL_MEMORY,
            ExperimentPreset.RECURRENT_MEMORY,
            ExperimentPreset.RECURRENT_GATING_MEMORY,
            ExperimentPreset.RECURRENT_HALTING_MEMORY,
            ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY,
            ExperimentPreset.RECURRENT_RESIDUAL,
            ExperimentPreset.RECURRENT_POST_NORM,
        ]

        for preset in parity_presets:
            with self.subTest(preset=preset.name):
                cfg = ExperimentPresets().get_config(preset)[0]
                self.assertIsNotNone(cfg.experiment_config.model_config)

    def test_controller_parity_presets_wire_expected_layer_configs(self):
        cases = {
            ExperimentPreset.GATING: (True, False, False),
            ExperimentPreset.HALTING: (False, True, False),
            ExperimentPreset.MEMORY: (False, False, True),
            ExperimentPreset.GATING_HALTING: (True, True, False),
            ExperimentPreset.GATING_MEMORY: (True, False, True),
            ExperimentPreset.HALTING_MEMORY: (False, True, True),
            ExperimentPreset.GATING_HALTING_MEMORY: (True, True, True),
        }

        for preset, (expected_gate, expected_halting, expected_memory) in cases.items():
            with self.subTest(preset=preset.name):
                cfg = ExperimentPresets().get_config(preset)[0]
                model_cfg = cfg.experiment_config.model_config
                layer_cfg = model_cfg.layer_config

                self.assertEqual(layer_cfg.gate_config is not None, expected_gate)
                self.assertEqual(
                    layer_cfg.halting_config is not None,
                    expected_halting,
                )
                self.assertEqual(
                    model_cfg.shared_memory_config is not None,
                    expected_memory,
                )

    def test_residual_and_post_norm_parity_presets_wire_layer_config(self):
        cases = {
            ExperimentPreset.RESIDUAL: (
                ResidualConnectionOptions.RESIDUAL,
                config.STACK_LAYER_NORM_POSITION,
                False,
                False,
                False,
            ),
            ExperimentPreset.POST_NORM: (
                ResidualConnectionOptions.DISABLED,
                LayerNormPositionOptions.AFTER,
                False,
                False,
                False,
            ),
            ExperimentPreset.RESIDUAL_POST_NORM: (
                ResidualConnectionOptions.RESIDUAL,
                LayerNormPositionOptions.AFTER,
                False,
                False,
                False,
            ),
            ExperimentPreset.RESIDUAL_GATING: (
                ResidualConnectionOptions.RESIDUAL,
                config.STACK_LAYER_NORM_POSITION,
                True,
                False,
                False,
            ),
            ExperimentPreset.RESIDUAL_HALTING: (
                ResidualConnectionOptions.RESIDUAL,
                config.STACK_LAYER_NORM_POSITION,
                False,
                True,
                False,
            ),
            ExperimentPreset.RESIDUAL_MEMORY: (
                ResidualConnectionOptions.RESIDUAL,
                config.STACK_LAYER_NORM_POSITION,
                False,
                False,
                True,
            ),
        }

        for preset, (
            expected_residual,
            expected_norm,
            expected_gate,
            expected_halting,
            expected_memory,
        ) in cases.items():
            with self.subTest(preset=preset.name):
                cfg = ExperimentPresets().get_config(preset)[0]
                model_cfg = cfg.experiment_config.model_config
                layer_cfg = model_cfg.layer_config

                self.assertEqual(
                    layer_cfg.residual_connection_option,
                    expected_residual,
                )
                self.assertEqual(layer_cfg.layer_norm_position, expected_norm)
                self.assertEqual(layer_cfg.gate_config is not None, expected_gate)
                self.assertEqual(
                    layer_cfg.halting_config is not None,
                    expected_halting,
                )
                self.assertEqual(
                    model_cfg.shared_memory_config is not None,
                    expected_memory,
                )

    def test_recurrent_residual_and_post_norm_presets_wire_block_config(self):
        cases = {
            ExperimentPreset.RECURRENT_RESIDUAL: (
                ResidualConnectionOptions.RESIDUAL,
                config.STACK_LAYER_NORM_POSITION,
            ),
            ExperimentPreset.RECURRENT_POST_NORM: (
                ResidualConnectionOptions.DISABLED,
                LayerNormPositionOptions.AFTER,
            ),
        }

        for preset, (expected_residual, expected_norm) in cases.items():
            with self.subTest(preset=preset.name):
                cfg = ExperimentPresets().get_config(preset)[0]
                recurrent_cfg = cfg.experiment_config.model_config

                self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
                self.assertEqual(
                    recurrent_cfg.block_config.layer_config.residual_connection_option,
                    expected_residual,
                )
                self.assertEqual(
                    recurrent_cfg.block_config.layer_config.layer_norm_position,
                    expected_norm,
                )

    def test_new_adaptive_combination_presets_wire_config(self):
        presets = ExperimentPresets()
        cases = [
            {
                "preset": ExperimentPreset.DUAL_WEIGHT_GATING,
                "config_role": "dual weight gate",
                "weight": DualModelDynamicWeightConfig,
                "gate": True,
            },
            {
                "preset": ExperimentPreset.DUAL_WEIGHT_HALTING,
                "config_role": "dual weight halting",
                "weight": DualModelDynamicWeightConfig,
                "halting": True,
            },
            {
                "preset": ExperimentPreset.DUAL_WEIGHT_GATING_HALTING,
                "config_role": "dual weight gate halting",
                "weight": DualModelDynamicWeightConfig,
                "gate": True,
                "halting": True,
            },
            {
                "preset": ExperimentPreset.DUAL_WEIGHT_MEMORY,
                "config_role": "dual weight memory",
                "weight": DualModelDynamicWeightConfig,
                "memory": True,
            },
            {
                "preset": ExperimentPreset.DUAL_WEIGHT_GATING_MEMORY,
                "config_role": "dual weight gate memory",
                "weight": DualModelDynamicWeightConfig,
                "gate": True,
                "memory": True,
            },
            {
                "preset": ExperimentPreset.DUAL_WEIGHT_HALTING_MEMORY,
                "config_role": "dual weight halting memory",
                "weight": DualModelDynamicWeightConfig,
                "halting": True,
                "memory": True,
            },
            {
                "preset": ExperimentPreset.FULL_STACK_GATING,
                "config_role": "full stack gate",
                "full_stack": True,
                "gate": True,
            },
            {
                "preset": ExperimentPreset.FULL_STACK_HALTING,
                "config_role": "full stack halting",
                "full_stack": True,
                "halting": True,
            },
            {
                "preset": ExperimentPreset.FULL_STACK_MEMORY,
                "config_role": "full stack memory",
                "full_stack": True,
                "memory": True,
            },
            {
                "preset": ExperimentPreset.FULL_STACK_GATING_HALTING,
                "config_role": "full stack gate halting",
                "full_stack": True,
                "gate": True,
                "halting": True,
            },
            {
                "preset": ExperimentPreset.FULL_STACK_RECURRENT,
                "config_role": "full stack recurrent",
                "full_stack": True,
                "recurrent": True,
            },
            {
                "preset": ExperimentPreset.BANK_WEIGHT_MASK,
                "config_role": "bank weight mask",
                "weight": LayeredWeightedBankDynamicWeightConfig,
                "mask": WeightInformedScoreAxisMaskConfig,
                "bias_none": True,
                "diagonal_none": True,
            },
            {
                "preset": ExperimentPreset.LOW_RANK_POST_NORM,
                "config_role": "low rank post norm",
                "weight": LowRankDynamicWeightConfig,
                "layer_norm": LayerNormPositionOptions.AFTER,
            },
        ]

        for case in cases:
            preset = case["preset"]
            with self.subTest(
                preset=preset.name,
                expected_config_role=case["config_role"],
            ):
                cfg = presets.get_config(preset)[0]
                model_cfg = cfg.experiment_config.model_config
                augmentation_config = self._augmentation_config(cfg)
                if case.get("recurrent"):
                    self.assertIsInstance(model_cfg, RecurrentLayerConfig)
                    layer_cfg = model_cfg.block_config.layer_config
                else:
                    layer_cfg = model_cfg.layer_config

                if case.get("full_stack"):
                    self._assert_full_stack_augmentation(augmentation_config)
                if "weight" in case:
                    self.assertIsInstance(
                        augmentation_config.weight_config,
                        case["weight"],
                    )
                if "mask" in case:
                    self.assertIsInstance(augmentation_config.mask_config, case["mask"])
                if case.get("bias_none"):
                    self.assertIsNone(augmentation_config.bias_config)
                if case.get("diagonal_none"):
                    self.assertIsNone(augmentation_config.diagonal_config)
                if case.get("gate"):
                    self.assertIsNotNone(layer_cfg.gate_config)
                if case.get("halting"):
                    self.assertIsNotNone(layer_cfg.halting_config)
                if case.get("memory"):
                    self.assertIsNotNone(model_cfg.shared_memory_config)
                if "layer_norm" in case:
                    self.assertEqual(layer_cfg.layer_norm_position, case["layer_norm"])

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

    def _assert_generator_stack(
        self,
        stack_config,
        *,
        hidden_dim: int,
        num_layers: int,
        activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        dropout_probability: float,
        last_layer_bias_option: LastLayerBiasOptions,
        apply_output_pipeline_flag: bool,
        bias_flag: bool,
    ) -> None:
        self.assertEqual(stack_config.hidden_dim, hidden_dim)
        self.assertEqual(stack_config.num_layers, num_layers)
        self.assertEqual(stack_config.layer_config.activation, activation)
        self.assertEqual(
            stack_config.layer_config.layer_norm_position,
            layer_norm_position,
        )
        self.assertEqual(
            stack_config.layer_config.dropout_probability,
            dropout_probability,
        )
        self.assertEqual(
            stack_config.last_layer_bias_option,
            last_layer_bias_option,
        )
        self.assertEqual(
            stack_config.apply_output_pipeline_flag,
            apply_output_pipeline_flag,
        )
        self.assertEqual(
            stack_config.layer_config.layer_model_config.bias_flag,
            bias_flag,
        )

    def _member_name(self, member) -> str:
        if member is None:
            return "None"
        return member.__name__

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )


if __name__ == "__main__":
    unittest.main()

import inspect
import unittest

import models.experts.linear_adaptive.config as config
import models.experts.linear_adaptive.dataset_options as dataset_options
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
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from emperor.memory.config import (
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from emperor.memory.options import MemoryPositionOptions
from models.experts.linear_adaptive._router_controller_config import (
    RouterControllerModelConfig,
)
from models.experts.linear_adaptive.config_builder import (
    LinearAdaptiveConfigBuilder,
)
from models.experts.linear_adaptive.model import Model
from models.experts.linear_adaptive.presets import (
    ExperimentPreset,
    ExperimentPresets,
)
from models.experts.linear_adaptive.runtime_defaults import runtime_from_flat
from models.experts.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    RuntimeOptions,
)

from model_runtime.packages import RandomSearch


class TestLinearAdaptiveExpertModel(unittest.TestCase):
    def experts_preset(self, **kwargs):
        return ExperimentPresets()._preset(**kwargs)

    def test_all_presets_forward_one_mnist_batch(self):
        batch_size = 2
        presets = ExperimentPresets()
        dataset = dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ][0]

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                output = model(X)
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_local_preset_catalog_has_stable_endpoints(self):
        self.assertEqual(ExperimentPreset.BASELINE.value, 1)
        self.assertEqual(ExperimentPreset.ADAPTIVE_SHARED_ROUTER.value, 9)
        self.assertGreaterEqual(len(ExperimentPreset.names()), 60)

    def test_baseline_forwards_all_datasets(self):
        batch_size = 2
        presets = ExperimentPresets()

        for dataset in dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]:
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(ExperimentPreset.BASELINE, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                output = model(X)
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_auxiliary_loss_preset_returns_finite_loss(self):
        batch_size = 2
        dataset = dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ][0]
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.ADAPTIVE_TOP1_SWITCH,
            dataset,
        )[0]
        model = Model(cfg)
        X = self._fake_batch(dataset, batch_size)

        output = model(X)

        self.assertIsInstance(output, tuple)
        logits, auxiliary_loss = output
        self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
        self.assertTrue(torch.isfinite(auxiliary_loss).item())

    def test_boundary_configs_are_separate_adaptive_linear_layers(self):
        cfg = self.experts_preset(
            stack_activation=ActivationOptions.MISH,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            stack_dropout_probability=0.13,
            stack_bias_flag=False,
            stack_gate_flag=True,
            stack_halting_flag=True,
            memory_flag=True,
        )
        exp_cfg = cfg.experiment_config
        stack_cfg = exp_cfg.model_config.stack_config

        self.assertIsNot(exp_cfg.input_model_config, exp_cfg.model_config)
        self.assertIsNot(exp_cfg.output_model_config, exp_cfg.model_config)
        self.assertIsNot(exp_cfg.input_model_config, exp_cfg.output_model_config)
        self.assertIsNotNone(stack_cfg.layer_config.gate_config)
        self.assertIsNotNone(stack_cfg.layer_config.halting_config)
        self.assertIsNotNone(stack_cfg.shared_memory_config)

        cases = (
            ("input", exp_cfg.input_model_config, ActivationOptions.MISH),
            ("output", exp_cfg.output_model_config, ActivationOptions.DISABLED),
        )
        for label, boundary_cfg, expected_activation in cases:
            with self.subTest(label=label):
                self.assertIsInstance(boundary_cfg, LayerConfig)
                self.assertEqual(boundary_cfg.activation, expected_activation)
                self.assertEqual(
                    boundary_cfg.layer_norm_position,
                    LayerNormPositionOptions.DISABLED,
                )
                self.assertEqual(
                    boundary_cfg.residual_connection_option,
                    ResidualConnectionOptions.DISABLED,
                )
                self.assertEqual(boundary_cfg.dropout_probability, 0.0)
                self.assertIsNone(boundary_cfg.gate_config)
                self.assertIsNone(boundary_cfg.halting_config)
                self.assertIsNone(boundary_cfg.memory_config)
                self.assertIsInstance(
                    boundary_cfg.layer_model_config,
                    AdaptiveLinearLayerConfig,
                )
                self.assertFalse(boundary_cfg.layer_model_config.bias_flag)
                self._assert_empty_adaptive_augmentation(
                    boundary_cfg.layer_model_config.adaptive_augmentation_config
                )

    def test_main_expert_and_router_stacks_use_adaptive_layer_configs(self):
        cfg = self.experts_preset(
            weight_option=DualModelDynamicWeightConfig,
            router_weight_option=LayeredWeightedBankDynamicWeightConfig,
            expert_bias_flag=False,
            router_bias_flag=False,
        )

        moe_layer_cfg = self._moe_layer_config(cfg)
        expert_stack_cfg = self._expert_stack_config(cfg)
        expert_layer_model_cfg = expert_stack_cfg.layer_config.layer_model_config
        router_layer_model_cfg = self._moe_model_config(
            cfg
        ).sampler_config.router_config.model_config.layer_config.layer_model_config

        self.assertIsInstance(moe_layer_cfg, MixtureOfExpertsConfig)
        self.assertIsInstance(expert_layer_model_cfg, AdaptiveLinearLayerConfig)
        self.assertIsInstance(router_layer_model_cfg, AdaptiveLinearLayerConfig)
        self.assertFalse(expert_layer_model_cfg.bias_flag)
        self.assertFalse(router_layer_model_cfg.bias_flag)
        self.assertIsInstance(
            expert_layer_model_cfg.adaptive_augmentation_config.weight_config,
            DualModelDynamicWeightConfig,
        )
        self.assertIsInstance(
            router_layer_model_cfg.adaptive_augmentation_config.weight_config,
            LayeredWeightedBankDynamicWeightConfig,
        )

    def test_preset_accepts_search_flags(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            RandomSearch(num_samples=2),
        )

        self.assertEqual(len(configs), 2)

    def test_option_group_build_matches_flat_kwargs(self):
        stack_options = ExpertsStackOptions(
            hidden_dim=16,
            bias_flag=False,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            num_layers=3,
            activation=ActivationOptions.MISH,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.13,
            last_layer_bias_option=LastLayerBiasOptions.ENABLED,
            apply_output_pipeline_flag=True,
        )
        submodule_stack_options = ExpertsSubmoduleStackOptions(
            hidden_dim=16,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            activation=ActivationOptions.ELU,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.06,
            bias_flag=True,
        )
        mixture_options = ExpertsMixtureOptions(
            top_k=2,
            num_experts=5,
            capacity_factor=1.25,
            dropped_token_behavior=DroppedTokenOptions.IDENTITY,
            compute_expert_mixture_flag=True,
            weighted_parameters_flag=True,
            weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
            routing_initialization_mode=RoutingInitializationMode.SHARED,
        )
        expert_stack_options = ExpertsSubmoduleStackOptions(
            hidden_dim=16,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=True,
            activation=ActivationOptions.GELU,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.05,
            bias_flag=False,
        )
        sampler_options = ExpertsSamplerOptions(
            threshold=0.17,
            filter_above_threshold=True,
            num_topk_samples=3,
            normalize_probabilities_flag=False,
            noisy_topk_flag=True,
            coefficient_of_variation_loss_weight=0.11,
            switch_loss_weight=0.12,
            zero_centred_loss_weight=0.13,
            mutual_information_loss_weight=0.14,
        )
        router_options = ExpertsRouterOptions(noisy_topk_flag=True)
        router_stack_options = ExpertsSubmoduleStackOptions(
            hidden_dim=16,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            activation=ActivationOptions.SILU,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.07,
            bias_flag=True,
        )
        gate_stack_options = ExpertsSubmoduleStackOptions(
            hidden_dim=18,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            activation=ActivationOptions.TANH,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.03,
            bias_flag=False,
        )
        halting_stack_options = ExpertsSubmoduleStackOptions(
            hidden_dim=20,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=False,
            activation=ActivationOptions.RELU,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.04,
            bias_flag=False,
        )
        memory_stack_options = ExpertsSubmoduleStackOptions(
            hidden_dim=22,
            num_layers=3,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=True,
            activation=ActivationOptions.SILU,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.02,
            bias_flag=False,
        )

        layer_controller_options = ExpertsLayerControllerOptions(
            stack_gate_flag=True,
            gate_option=LayerGateOptions.ADDITION,
            gate_activation=ActivationOptions.TANH,
            gate_stack_source=self.experts_stack_source(gate_stack_options),
            stack_halting_flag=True,
            halting_threshold=0.63,
            halting_dropout=0.08,
            halting_hidden_state_mode=HaltingHiddenStateModeOptions.ACCUMULATED,
            halting_stack_source=self.experts_stack_source(halting_stack_options),
            halting_output_dim=2,
        )
        dynamic_memory_options = ExpertsDynamicMemoryOptions(
            memory_flag=True,
            memory_option=WeightedDynamicMemoryConfig,
            memory_position_option=MemoryPositionOptions.BEFORE_AFFINE,
            memory_test_time_training_learning_rate=0.02,
            memory_test_time_training_num_inner_steps=2,
            memory_stack_source=self.experts_stack_source(memory_stack_options),
        )
        recurrent_controller_options = ExpertsRecurrentControllerOptions(
            recurrent_flag=True,
            recurrent_max_steps=3,
            recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            recurrent_gate_flag=True,
            recurrent_gate_option=LayerGateOptions.MULTIPLIER,
            recurrent_gate_activation=ActivationOptions.SIGMOID,
            recurrent_gate_stack_source=self.experts_stack_source(gate_stack_options),
            recurrent_halting_flag=True,
            recurrent_halting_threshold=0.71,
            recurrent_halting_dropout=0.09,
            recurrent_halting_hidden_state_mode=(
                HaltingHiddenStateModeOptions.ACCUMULATED
            ),
            recurrent_halting_stack_source=self.experts_stack_source(
                halting_stack_options
            ),
        )
        adaptive_generator_stack_options = AdaptiveGeneratorStackOptions(
            hidden_dim=12,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            num_layers=2,
            activation=ActivationOptions.ELU,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.09,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=False,
            bias_flag=False,
        )
        hidden_adaptive_weight_options = HiddenAdaptiveWeightOptions(
            generator_depth=DynamicDepthOptions.DEPTH_OF_FOUR,
            option_flag=True,
            option=DualModelDynamicWeightConfig,
            normalization_option=WeightNormalizationOptions.RMS,
            normalization_position_option=(
                WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
            ),
            decay_schedule=WeightDecayScheduleOptions.LINEAR,
            decay_rate=0.21,
            decay_warmup_batches=7,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_THREE,
            generator_stack_source=self.adaptive_generator_stack_source(
                hidden_dim=14,
                num_layers=3,
                activation=ActivationOptions.RELU,
            ),
        )
        hidden_adaptive_bias_options = HiddenAdaptiveBiasOptions(
            option_flag=True,
            option=WeightedBankDynamicBiasConfig,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.31,
            decay_warmup_batches=9,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_FOUR,
            generator_stack_source=self.adaptive_generator_stack_source(
                hidden_dim=15,
                activation=ActivationOptions.MISH,
            ),
        )
        hidden_adaptive_diagonal_options = HiddenAdaptiveDiagonalOptions(
            option_flag=True,
            option=CombinedDynamicDiagonalConfig,
            generator_stack_source=self.adaptive_generator_stack_source(
                hidden_dim=17,
                activation=ActivationOptions.TANH,
            ),
        )
        hidden_adaptive_mask_options = HiddenAdaptiveMaskOptions(
            option_flag=True,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            mask_dimension_option=MaskDimensionOptions.ROW,
            mask_threshold=0.67,
            mask_surrogate_scale=4.0,
            mask_floor=0.05,
            mask_transition_width=0.2,
            generator_stack_source=self.adaptive_generator_stack_source(
                hidden_dim=19,
                activation=ActivationOptions.SILU,
            ),
        )
        router_adaptive_weight_options = HiddenAdaptiveWeightOptions(
            generator_depth=DynamicDepthOptions.DEPTH_OF_THREE,
            option_flag=True,
            option=LayeredWeightedBankDynamicWeightConfig,
            normalization_option=WeightNormalizationOptions.RMS,
            normalization_position_option=(
                WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
            ),
            decay_schedule=WeightDecayScheduleOptions.MULTIPLICATIVE,
            decay_rate=0.11,
            decay_warmup_batches=5,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
            generator_stack_source=self.adaptive_generator_stack_source(
                hidden_dim=13,
                activation=ActivationOptions.GELU,
            ),
        )

        flat_kwargs = {
            "batch_size": 3,
            "learning_rate": 0.02,
            "input_dim": 8,
            "output_dim": 4,
            "hidden_dim": stack_options.hidden_dim,
            "stack_bias_flag": stack_options.bias_flag,
            "layer_norm_position": stack_options.layer_norm_position,
            "stack_num_layers": stack_options.num_layers,
            "stack_activation": stack_options.activation,
            "stack_residual_connection_option": (
                stack_options.residual_connection_option
            ),
            "stack_dropout_probability": stack_options.dropout_probability,
            "stack_last_layer_bias_option": (stack_options.last_layer_bias_option),
            "stack_apply_output_pipeline_flag": (
                stack_options.apply_output_pipeline_flag
            ),
            **self.flat_stack_kwargs(
                "submodule_stack",
                submodule_stack_options,
                bias_name="submodule_stack_bias_flag",
            ),
            "top_k": mixture_options.top_k,
            "num_experts": mixture_options.num_experts,
            "capacity_factor": mixture_options.capacity_factor,
            "dropped_token_behavior": mixture_options.dropped_token_behavior,
            "compute_expert_mixture_flag": (
                mixture_options.compute_expert_mixture_flag
            ),
            "weighted_parameters_flag": mixture_options.weighted_parameters_flag,
            "weighting_position_option": (mixture_options.weighting_position_option),
            "routing_initialization_mode": (
                mixture_options.routing_initialization_mode
            ),
            **self.flat_stack_kwargs(
                "expert_stack",
                expert_stack_options,
                bias_name="expert_bias_flag",
            ),
            "sampler_threshold": sampler_options.threshold,
            "sampler_filter_above_threshold": (sampler_options.filter_above_threshold),
            "sampler_num_topk_samples": sampler_options.num_topk_samples,
            "sampler_normalize_probabilities_flag": (
                sampler_options.normalize_probabilities_flag
            ),
            "sampler_noisy_topk_flag": sampler_options.noisy_topk_flag,
            "sampler_coefficient_of_variation_loss_weight": (
                sampler_options.coefficient_of_variation_loss_weight
            ),
            "sampler_switch_loss_weight": sampler_options.switch_loss_weight,
            "sampler_zero_centred_loss_weight": (
                sampler_options.zero_centred_loss_weight
            ),
            "sampler_mutual_information_loss_weight": (
                sampler_options.mutual_information_loss_weight
            ),
            "router_noisy_topk_flag": router_options.noisy_topk_flag,
            **self.flat_stack_kwargs(
                "router_stack",
                router_stack_options,
                bias_name="router_bias_flag",
            ),
            "stack_gate_flag": layer_controller_options.stack_gate_flag,
            "gate_option": layer_controller_options.gate_option,
            "gate_activation": layer_controller_options.gate_activation,
            "gate_stack_independent_flag": True,
            **self.flat_stack_kwargs(
                "gate_stack",
                gate_stack_options,
                bias_name="gate_stack_bias_flag",
            ),
            "stack_halting_flag": layer_controller_options.stack_halting_flag,
            "halting_threshold": layer_controller_options.halting_threshold,
            "halting_dropout": layer_controller_options.halting_dropout,
            "halting_hidden_state_mode": (
                layer_controller_options.halting_hidden_state_mode
            ),
            "halting_output_dim": layer_controller_options.halting_output_dim,
            "halting_stack_independent_flag": True,
            **self.flat_stack_kwargs(
                "halting_stack",
                halting_stack_options,
                bias_name="halting_stack_bias_flag",
            ),
            "memory_flag": dynamic_memory_options.memory_flag,
            "memory_option": dynamic_memory_options.memory_option,
            "memory_position_option": dynamic_memory_options.memory_position_option,
            "memory_test_time_training_learning_rate": (
                dynamic_memory_options.memory_test_time_training_learning_rate
            ),
            "memory_test_time_training_num_inner_steps": (
                dynamic_memory_options.memory_test_time_training_num_inner_steps
            ),
            "memory_stack_independent_flag": True,
            **self.flat_stack_kwargs(
                "memory_stack",
                memory_stack_options,
                bias_name="memory_stack_bias_flag",
            ),
            "recurrent_flag": recurrent_controller_options.recurrent_flag,
            "recurrent_max_steps": (recurrent_controller_options.recurrent_max_steps),
            "recurrent_layer_norm_position": (
                recurrent_controller_options.recurrent_layer_norm_position
            ),
            "recurrent_gate_flag": (recurrent_controller_options.recurrent_gate_flag),
            "recurrent_gate_option": (
                recurrent_controller_options.recurrent_gate_option
            ),
            "recurrent_gate_activation": (
                recurrent_controller_options.recurrent_gate_activation
            ),
            "recurrent_gate_stack_independent_flag": True,
            **self.flat_stack_kwargs(
                "recurrent_gate_stack",
                gate_stack_options,
                bias_name="recurrent_gate_stack_bias_flag",
            ),
            "recurrent_halting_flag": (
                recurrent_controller_options.recurrent_halting_flag
            ),
            "recurrent_halting_threshold": (
                recurrent_controller_options.recurrent_halting_threshold
            ),
            "recurrent_halting_dropout": (
                recurrent_controller_options.recurrent_halting_dropout
            ),
            "recurrent_halting_hidden_state_mode": (
                recurrent_controller_options.recurrent_halting_hidden_state_mode
            ),
            "recurrent_halting_stack_independent_flag": True,
            **self.flat_stack_kwargs(
                "recurrent_halting_stack",
                halting_stack_options,
                bias_name="recurrent_halting_stack_bias_flag",
            ),
            "adaptive_generator_stack_hidden_dim": (
                adaptive_generator_stack_options.hidden_dim
            ),
            "adaptive_generator_stack_layer_norm_position": (
                adaptive_generator_stack_options.layer_norm_position
            ),
            "adaptive_generator_stack_num_layers": (
                adaptive_generator_stack_options.num_layers
            ),
            "adaptive_generator_stack_activation": (
                adaptive_generator_stack_options.activation
            ),
            "adaptive_generator_stack_residual_connection_option": (
                adaptive_generator_stack_options.residual_connection_option
            ),
            "adaptive_generator_stack_dropout_probability": (
                adaptive_generator_stack_options.dropout_probability
            ),
            "adaptive_generator_stack_last_layer_bias_option": (
                adaptive_generator_stack_options.last_layer_bias_option
            ),
            "adaptive_generator_stack_apply_output_pipeline_flag": (
                adaptive_generator_stack_options.apply_output_pipeline_flag
            ),
            "adaptive_generator_stack_bias_flag": (
                adaptive_generator_stack_options.bias_flag
            ),
            **self.flat_adaptive_weight_kwargs(
                hidden_adaptive_weight_options,
                option_prefix="weight",
                stack_prefix="weight_generator_stack",
            ),
            **self.flat_adaptive_bias_kwargs(
                hidden_adaptive_bias_options,
                option_prefix="bias",
                stack_prefix="bias_generator_stack",
            ),
            **self.flat_adaptive_diagonal_kwargs(
                hidden_adaptive_diagonal_options,
                option_prefix="diagonal",
                stack_prefix="diagonal_generator_stack",
            ),
            **self.flat_adaptive_mask_kwargs(
                hidden_adaptive_mask_options,
                option_prefix="",
                stack_prefix="mask_generator_stack",
            ),
            **self.flat_adaptive_weight_kwargs(
                router_adaptive_weight_options,
                option_prefix="router_weight",
                stack_prefix="router_weight_generator_stack",
            ),
        }

        flat_cfg = self.experts_preset(**flat_kwargs)
        grouped_cfg = LinearAdaptiveConfigBuilder(
            batch_size=3,
            learning_rate=0.02,
            input_dim=8,
            output_dim=4,
            stack_options=stack_options,
            submodule_stack_options=submodule_stack_options,
            mixture_options=mixture_options,
            expert_stack_options=expert_stack_options,
            sampler_options=sampler_options,
            router_options=router_options,
            router_stack_options=router_stack_options,
            layer_controller_options=layer_controller_options,
            dynamic_memory_options=dynamic_memory_options,
            recurrent_controller_options=recurrent_controller_options,
            adaptive_generator_stack_options=adaptive_generator_stack_options,
            hidden_adaptive_weight_options=hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=hidden_adaptive_diagonal_options,
            hidden_adaptive_mask_options=hidden_adaptive_mask_options,
            router_adaptive_weight_options=router_adaptive_weight_options,
        ).build()

        self.assertEqual(flat_cfg, grouped_cfg)

    def test_runtime_is_primary_and_removed_legacy_names_are_rejected(self):
        parameters = inspect.signature(LinearAdaptiveConfigBuilder.__init__).parameters
        old_names = {
            "router_stack_independent_flag",
            "sampler_stack_options",
            "sampler_stack_independent_flag",
            "sampler_stack_hidden_dim",
            "sampler_bias_flag",
        }

        self.assertIn("runtime", parameters)
        self.assertLessEqual(
            set(parameters),
            {"self", "legacy_args", "runtime", "legacy_options"},
        )

        for name in old_names:
            with self.subTest(name=name):
                self.assertNotIn(name, parameters)

        for kwargs in (
            {"router_stack_independent_flag": True},
            {
                "sampler_stack_options": ExpertsSubmoduleStackOptions(
                    hidden_dim=8,
                    num_layers=1,
                    last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                    apply_output_pipeline_flag=False,
                    activation=ActivationOptions.RELU,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
                    dropout_probability=0.0,
                    bias_flag=True,
                )
            },
            {"sampler_stack_independent_flag": True},
            {"sampler_stack_hidden_dim": 13},
            {"sampler_bias_flag": False},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    LinearAdaptiveConfigBuilder(**kwargs)

                with self.assertRaises(TypeError):
                    self.experts_preset(**kwargs)

    def test_typed_runtime_builds_the_model_config(self):
        runtime = runtime_from_flat(
            {
                "batch_size": 3,
                "learning_rate": 0.02,
                "input_dim": 8,
                "hidden_dim": 16,
                "output_dim": 4,
                "stack_num_layers": 1,
                "submodule_stack_hidden_dim": 16,
                "num_experts": 2,
                "top_k": 1,
            },
            config,
        )

        self.assertIsInstance(runtime, RuntimeOptions)
        cfg = LinearAdaptiveConfigBuilder(runtime=runtime).build()

        self.assertEqual(cfg.batch_size, 3)
        self.assertEqual(cfg.learning_rate, 0.02)
        self.assertEqual(cfg.input_dim, 8)
        self.assertEqual(cfg.hidden_dim, 16)
        self.assertEqual(cfg.output_dim, 4)

    def test_shared_gate_config_is_stored_on_stack_config(self):
        shared_gate_config = self.shared_gate_config()
        cfg = self.experts_preset(shared_gate_config=shared_gate_config)
        stack_cfg = cfg.experiment_config.model_config.stack_config

        self.assertIs(stack_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(stack_cfg.layer_config.gate_config)

    def test_shared_gate_config_rejects_enabled_stack_gate(self):
        cfg = self.experts_preset(
            stack_gate_flag=True,
            shared_gate_config=self.shared_gate_config(),
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Model(cfg)

    def test_shared_gate_config_allows_absent_stack_gate(self):
        shared_gate_config = self.shared_gate_config()
        cfg = self.experts_preset(
            stack_gate_flag=False,
            shared_gate_config=shared_gate_config,
        )
        stack_cfg = cfg.experiment_config.model_config.stack_config

        self.assertIs(stack_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(stack_cfg.layer_config.gate_config)

    def test_controller_stack_overrides_use_canonical_names(self):
        cfg = self.experts_preset(
            stack_gate_flag=True,
            gate_stack_independent_flag=True,
            gate_stack_hidden_dim=32,
            gate_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            gate_stack_bias_flag=False,
            stack_halting_flag=True,
            halting_stack_independent_flag=True,
            halting_stack_hidden_dim=48,
            halting_stack_layer_norm_position=LayerNormPositionOptions.BEFORE,
            halting_stack_bias_flag=False,
        )
        stack_cfg = cfg.experiment_config.model_config.stack_config
        gate_stack = stack_cfg.layer_config.gate_config.model_config
        halting_stack = stack_cfg.layer_config.halting_config.halting_gate_config

        self.assertEqual(gate_stack.hidden_dim, 32)
        self.assertEqual(
            gate_stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertFalse(gate_stack.layer_config.layer_model_config.bias_flag)
        self.assertEqual(halting_stack.hidden_dim, 48)
        self.assertEqual(
            halting_stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertFalse(halting_stack.layer_config.layer_model_config.bias_flag)

    def test_router_stack_defaults_do_not_inherit_submodule_overrides(self):
        cfg = self.experts_preset(
            submodule_stack_hidden_dim=44,
            submodule_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            submodule_stack_apply_output_pipeline_flag=True,
            submodule_stack_bias_flag=False,
        )
        router_stack = (
            cfg.experiment_config.model_config.sampler_config.router_config.model_config
        )

        self.assertEqual(router_stack.hidden_dim, config.ROUTER_STACK_HIDDEN_DIM)
        self.assertEqual(router_stack.num_layers, config.ROUTER_STACK_NUM_LAYERS)
        self.assertEqual(
            router_stack.layer_config.activation,
            config.ROUTER_STACK_ACTIVATION,
        )
        self.assertEqual(
            router_stack.layer_config.residual_connection_option,
            config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION,
        )
        self.assertEqual(
            router_stack.layer_config.dropout_probability,
            config.ROUTER_STACK_DROPOUT_PROBABILITY,
        )
        self.assertEqual(
            router_stack.layer_config.layer_norm_position,
            config.ROUTER_STACK_LAYER_NORM_POSITION,
        )
        self.assertEqual(
            router_stack.last_layer_bias_option,
            config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION,
        )
        self.assertEqual(
            router_stack.apply_output_pipeline_flag,
            config.ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        )
        self.assertEqual(
            router_stack.layer_config.layer_model_config.bias_flag,
            config.ROUTER_BIAS_FLAG,
        )

    def test_router_stack_overrides_apply_without_independent_flag(self):
        cfg = self.experts_preset(
            submodule_stack_hidden_dim=44,
            submodule_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            submodule_stack_apply_output_pipeline_flag=False,
            submodule_stack_bias_flag=True,
            router_stack_hidden_dim=88,
            router_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            router_stack_apply_output_pipeline_flag=True,
            router_bias_flag=False,
        )
        router_stack = (
            cfg.experiment_config.model_config.sampler_config.router_config.model_config
        )

        self.assertEqual(router_stack.hidden_dim, 88)
        self.assertEqual(
            router_stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertTrue(router_stack.apply_output_pipeline_flag)
        self.assertFalse(router_stack.layer_config.layer_model_config.bias_flag)

    def test_router_controller_flags_build_stable_trunk_and_logits_head(self):
        cfg = self.experts_preset(
            router_gate_flag=True,
            router_halting_flag=True,
            router_memory_flag=True,
            router_recurrent_flag=True,
            router_recurrent_gate_flag=True,
            router_recurrent_halting_flag=True,
            router_weight_option=LayeredWeightedBankDynamicWeightConfig,
        )
        router_model_cfg = self._moe_model_config(
            cfg
        ).sampler_config.router_config.model_config

        self.assertIsInstance(router_model_cfg, RouterControllerModelConfig)
        self.assertEqual(router_model_cfg.input_dim, config.HIDDEN_DIM)
        self.assertEqual(router_model_cfg.hidden_dim, config.SUBMODULE_STACK_HIDDEN_DIM)
        self.assertEqual(router_model_cfg.output_dim, config.EXPERT_NUM_EXPERTS)
        self.assertIsInstance(router_model_cfg.trunk_config, RecurrentLayerConfig)
        self.assertIsInstance(router_model_cfg.head_config, LayerConfig)

        recurrent_cfg = router_model_cfg.trunk_config
        trunk_stack = recurrent_cfg.block_config
        self.assertIsInstance(trunk_stack, LayerStackConfig)
        self.assertEqual(trunk_stack.input_dim, router_model_cfg.hidden_dim)
        self.assertEqual(trunk_stack.hidden_dim, router_model_cfg.hidden_dim)
        self.assertEqual(trunk_stack.output_dim, router_model_cfg.hidden_dim)
        self.assertIsNotNone(trunk_stack.layer_config.gate_config)
        self.assertIsNotNone(trunk_stack.layer_config.halting_config)
        self.assertIsNotNone(trunk_stack.shared_memory_config)
        self.assertIsNotNone(recurrent_cfg.gate_config)
        self.assertIsNotNone(recurrent_cfg.halting_config)
        self.assertIsNone(router_model_cfg.head_config.gate_config)
        self.assertIsNone(router_model_cfg.head_config.halting_config)
        self.assertIsInstance(
            trunk_stack.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )
        self.assertIsInstance(
            router_model_cfg.head_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )
        self.assertIsInstance(
            trunk_stack.layer_config.layer_model_config.adaptive_augmentation_config.weight_config,
            LayeredWeightedBankDynamicWeightConfig,
        )
        self.assertIsInstance(
            router_model_cfg.head_config.layer_model_config.adaptive_augmentation_config.weight_config,
            LayeredWeightedBankDynamicWeightConfig,
        )

    def test_router_controller_stacks_inherit_submodule_defaults(self):
        cfg = self.experts_preset(
            submodule_stack_hidden_dim=32,
            submodule_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            submodule_stack_apply_output_pipeline_flag=False,
            submodule_stack_bias_flag=False,
            router_stack_hidden_dim=64,
            router_stack_layer_norm_position=LayerNormPositionOptions.BEFORE,
            router_stack_apply_output_pipeline_flag=True,
            router_bias_flag=True,
            router_gate_flag=True,
            router_halting_flag=True,
            router_memory_flag=True,
            router_recurrent_flag=True,
            router_recurrent_gate_flag=True,
            router_recurrent_halting_flag=True,
        )
        router_model_cfg = self._moe_model_config(
            cfg
        ).sampler_config.router_config.model_config
        recurrent_cfg = router_model_cfg.trunk_config
        trunk_stack = recurrent_cfg.block_config

        self.assertEqual(router_model_cfg.hidden_dim, 64)
        self.assertEqual(trunk_stack.hidden_dim, 64)
        self.assertEqual(
            trunk_stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertTrue(trunk_stack.apply_output_pipeline_flag)
        self.assertTrue(trunk_stack.layer_config.layer_model_config.bias_flag)

        inherited_stacks = (
            trunk_stack.layer_config.gate_config.model_config,
            trunk_stack.layer_config.halting_config.halting_gate_config,
            trunk_stack.shared_memory_config.model_config,
            recurrent_cfg.gate_config.model_config,
            recurrent_cfg.halting_config.halting_gate_config,
        )
        for stack_cfg in inherited_stacks:
            with self.subTest(stack_cfg=stack_cfg):
                self.assertEqual(stack_cfg.hidden_dim, 32)
                self.assertEqual(
                    stack_cfg.layer_config.layer_norm_position,
                    LayerNormPositionOptions.AFTER,
                )
                self.assertFalse(stack_cfg.apply_output_pipeline_flag)
                self.assertFalse(stack_cfg.layer_config.layer_model_config.bias_flag)

        halting_stacks = (
            trunk_stack.layer_config.halting_config.halting_gate_config,
            recurrent_cfg.halting_config.halting_gate_config,
        )
        for stack_cfg in halting_stacks:
            with self.subTest(halting_stack=stack_cfg):
                self.assertEqual(
                    stack_cfg.last_layer_bias_option,
                    LastLayerBiasOptions.DISABLED,
                )

    def test_flat_router_controller_kwargs_reach_controller_stacks(self):
        cfg = self.experts_preset(
            router_gate_flag=True,
            router_gate_stack_independent_flag=True,
            router_gate_stack_hidden_dim=41,
            router_halting_flag=True,
            router_halting_stack_independent_flag=True,
            router_halting_stack_hidden_dim=45,
            router_memory_flag=True,
            router_memory_stack_independent_flag=True,
            router_memory_stack_hidden_dim=43,
            router_recurrent_flag=True,
            router_recurrent_gate_flag=True,
            router_recurrent_gate_stack_independent_flag=True,
            router_recurrent_gate_stack_hidden_dim=47,
            router_recurrent_halting_flag=True,
            router_recurrent_halting_stack_independent_flag=True,
            router_recurrent_halting_stack_hidden_dim=49,
        )
        router_model_cfg = self._moe_model_config(
            cfg
        ).sampler_config.router_config.model_config
        recurrent_cfg = router_model_cfg.trunk_config
        trunk_stack = recurrent_cfg.block_config

        self.assertEqual(
            trunk_stack.layer_config.gate_config.model_config.hidden_dim,
            41,
        )
        self.assertEqual(
            trunk_stack.layer_config.halting_config.halting_gate_config.hidden_dim,
            45,
        )
        self.assertEqual(
            trunk_stack.shared_memory_config.model_config.hidden_dim,
            43,
        )
        self.assertEqual(
            recurrent_cfg.gate_config.model_config.hidden_dim,
            47,
        )
        self.assertEqual(
            recurrent_cfg.halting_config.halting_gate_config.hidden_dim,
            49,
        )

    def test_router_recurrent_forwards_when_hidden_dim_differs_from_num_experts(self):
        cfg = self.experts_preset(
            hidden_dim=32,
            submodule_stack_hidden_dim=32,
            num_experts=7,
            top_k=2,
            router_recurrent_flag=True,
        )
        model = Model(cfg)
        X = self._fake_batch(
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            batch_size=2,
        )

        output = model(X)
        logits = output[0] if isinstance(output, tuple) else output

        self.assertEqual(
            logits.shape,
            (
                2,
                dataset_options.DATASET_OPTIONS_BY_TASK[
                    dataset_options.DEFAULT_EXPERIMENT_TASK
                ][0].num_classes,
            ),
        )

    def test_disabled_memory_is_absent_from_stack_and_layer_configs(self):
        cfg = LinearAdaptiveConfigBuilder().build()
        stack_cfg = cfg.experiment_config.model_config.stack_config

        self.assertIsNone(stack_cfg.shared_memory_config)
        self.assertIsNone(stack_cfg.layer_config.memory_config)

    def test_memory_config_is_shared_on_stack_config(self):
        cfg = self.experts_preset(memory_flag=True)
        stack_cfg = cfg.experiment_config.model_config.stack_config
        memory_cfg = stack_cfg.shared_memory_config

        self.assertIsInstance(memory_cfg, GatedResidualDynamicMemoryConfig)
        self.assertEqual(memory_cfg.input_dim, config.HIDDEN_DIM)
        self.assertEqual(memory_cfg.output_dim, config.HIDDEN_DIM)
        self.assertEqual(
            memory_cfg.memory_position_option,
            MemoryPositionOptions.AFTER_AFFINE,
        )
        self.assertIsNone(memory_cfg.test_time_training_learning_rate)
        self.assertIsNone(memory_cfg.test_time_training_num_inner_steps)
        self.assertIsNone(stack_cfg.layer_config.memory_config)
        self.assertIsInstance(
            memory_cfg.model_config.layer_config.layer_model_config,
            LinearLayerConfig,
        )

    def test_memory_stack_overrides_require_independent_flag(self):
        cfg = self.experts_preset(
            memory_flag=True,
            submodule_stack_hidden_dim=11,
            submodule_stack_activation=ActivationOptions.RELU,
            memory_stack_hidden_dim=44,
            memory_stack_activation=ActivationOptions.TANH,
        )
        stack_cfg = cfg.experiment_config.model_config.stack_config
        memory_stack = stack_cfg.shared_memory_config.model_config

        self.assertEqual(memory_stack.hidden_dim, 11)
        self.assertEqual(
            memory_stack.layer_config.activation,
            ActivationOptions.RELU,
        )

    def test_memory_config_uses_overrides_when_independent(self):
        cfg = self.experts_preset(
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
        memory_cfg = (
            cfg.experiment_config.model_config.stack_config.shared_memory_config
        )
        memory_stack = memory_cfg.model_config

        self.assertIsInstance(memory_cfg, WeightedDynamicMemoryConfig)
        self.assertEqual(
            memory_cfg.memory_position_option,
            MemoryPositionOptions.BEFORE_AFFINE,
        )
        self.assertEqual(memory_cfg.test_time_training_learning_rate, 0.02)
        self.assertEqual(memory_cfg.test_time_training_num_inner_steps, 2)
        self.assertEqual(memory_stack.hidden_dim, 12)
        self.assertEqual(memory_stack.num_layers, 3)
        self.assertEqual(
            memory_stack.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertTrue(memory_stack.apply_output_pipeline_flag)
        self.assertEqual(
            memory_stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(memory_stack.layer_config.activation, ActivationOptions.SILU)
        self.assertFalse(memory_stack.layer_config.layer_model_config.bias_flag)
        self.assertIsNone(memory_stack.layer_config.memory_config)

    def test_controller_stack_overrides_require_independent_flags(self):
        cfg = self.experts_preset(
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
        stack_cfg = cfg.experiment_config.model_config.stack_config

        self.assertEqual(stack_cfg.layer_config.gate_config.model_config.hidden_dim, 11)
        self.assertEqual(
            stack_cfg.layer_config.gate_config.model_config.layer_config.activation,
            ActivationOptions.RELU,
        )
        self.assertEqual(
            stack_cfg.layer_config.halting_config.halting_gate_config.hidden_dim,
            11,
        )
        self.assertEqual(
            stack_cfg.layer_config.halting_config.halting_gate_config.layer_config.activation,
            ActivationOptions.RELU,
        )
        self.assertEqual(stack_cfg.shared_memory_config.model_config.hidden_dim, 11)
        self.assertEqual(
            stack_cfg.shared_memory_config.model_config.layer_config.activation,
            ActivationOptions.RELU,
        )

    def test_controller_stack_independent_flags_enable_controller_options(self):
        cfg = self.experts_preset(
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
        stack_cfg = cfg.experiment_config.model_config.stack_config

        self.assertEqual(stack_cfg.layer_config.gate_config.model_config.hidden_dim, 22)
        self.assertEqual(
            stack_cfg.layer_config.gate_config.model_config.layer_config.activation,
            ActivationOptions.SILU,
        )
        self.assertEqual(
            stack_cfg.layer_config.halting_config.halting_gate_config.hidden_dim,
            33,
        )
        self.assertEqual(
            stack_cfg.layer_config.halting_config.halting_gate_config.layer_config.activation,
            ActivationOptions.MISH,
        )
        self.assertEqual(stack_cfg.shared_memory_config.model_config.hidden_dim, 44)
        self.assertEqual(
            stack_cfg.shared_memory_config.model_config.layer_config.activation,
            ActivationOptions.TANH,
        )

    def test_recurrent_controllers_use_separate_stack_and_scalar_overrides(self):
        cfg = self.experts_preset(
            recurrent_flag=True,
            stack_gate_flag=True,
            gate_stack_independent_flag=True,
            gate_stack_hidden_dim=32,
            recurrent_gate_flag=True,
            recurrent_gate_stack_independent_flag=True,
            recurrent_gate_stack_hidden_dim=64,
            stack_halting_flag=True,
            halting_stack_independent_flag=True,
            halting_threshold=0.55,
            halting_stack_hidden_dim=40,
            recurrent_halting_flag=True,
            recurrent_halting_threshold=0.75,
            recurrent_halting_dropout=0.2,
            recurrent_halting_hidden_state_mode=(
                HaltingHiddenStateModeOptions.ACCUMULATED
            ),
            recurrent_halting_stack_independent_flag=True,
            recurrent_halting_stack_hidden_dim=72,
        )
        recurrent_cfg = cfg.experiment_config.model_config
        block_stack = recurrent_cfg.block_config.stack_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            block_stack.layer_config.gate_config.model_config.hidden_dim, 32
        )
        self.assertEqual(recurrent_cfg.gate_config.model_config.hidden_dim, 64)
        self.assertEqual(block_stack.layer_config.halting_config.threshold, 0.55)
        self.assertEqual(recurrent_cfg.halting_config.threshold, 0.75)
        self.assertEqual(recurrent_cfg.halting_config.halting_dropout, 0.2)
        self.assertEqual(
            recurrent_cfg.halting_config.hidden_state_mode,
            HaltingHiddenStateModeOptions.ACCUMULATED,
        )
        self.assertEqual(
            recurrent_cfg.halting_config.halting_gate_config.hidden_dim,
            72,
        )

    def test_expert_gate_config_is_internal_to_expert_model_config(self):
        cfg = self.experts_preset(expert_gate_flag=True)
        outer_stack = cfg.experiment_config.model_config.stack_config
        expert_cfg = self._expert_model_config(cfg)

        self.assertIsInstance(expert_cfg, LayerStackConfig)
        self.assertIsNone(outer_stack.layer_config.gate_config)
        self.assertIsNotNone(expert_cfg.layer_config.gate_config)
        self.assertEqual(
            expert_cfg.layer_config.gate_config.option,
            config.EXPERT_GATE_OPTION,
        )

    def test_expert_halting_config_is_internal_to_expert_model_config(self):
        cfg = self.experts_preset(expert_halting_flag=True)
        outer_stack = cfg.experiment_config.model_config.stack_config
        expert_cfg = self._expert_model_config(cfg)

        self.assertIsInstance(expert_cfg, LayerStackConfig)
        self.assertIsNone(outer_stack.layer_config.halting_config)
        self.assertIsNotNone(expert_cfg.layer_config.halting_config)
        self.assertEqual(
            expert_cfg.layer_config.halting_config.threshold,
            config.EXPERT_HALTING_THRESHOLD,
        )

    def test_expert_memory_config_is_internal_to_expert_model_config(self):
        cfg = self.experts_preset(expert_memory_flag=True)
        outer_stack = cfg.experiment_config.model_config.stack_config
        expert_cfg = self._expert_model_config(cfg)

        self.assertIsInstance(expert_cfg, LayerStackConfig)
        self.assertIsNone(outer_stack.shared_memory_config)
        self.assertIsNone(outer_stack.layer_config.memory_config)
        self.assertIsInstance(
            expert_cfg.shared_memory_config,
            GatedResidualDynamicMemoryConfig,
        )
        self.assertIsNone(expert_cfg.layer_config.memory_config)

    def test_expert_recurrent_wraps_only_expert_model_config(self):
        cfg = self.experts_preset(
            expert_recurrent_flag=True,
            expert_recurrent_gate_flag=True,
            expert_recurrent_halting_flag=True,
            expert_recurrent_halting_threshold=0.77,
            expert_recurrent_gate_stack_independent_flag=True,
            expert_recurrent_gate_stack_hidden_dim=25,
            expert_recurrent_halting_stack_independent_flag=True,
            expert_recurrent_halting_stack_hidden_dim=27,
        )
        model_cfg = cfg.experiment_config.model_config
        expert_cfg = self._expert_model_config(cfg)

        self.assertIsInstance(model_cfg, MixtureOfExpertsModelConfig)
        self.assertIsInstance(expert_cfg, RecurrentLayerConfig)
        self.assertIsInstance(expert_cfg.block_config, LayerStackConfig)
        self.assertEqual(expert_cfg.max_steps, config.EXPERT_RECURRENT_MAX_STEPS)
        self.assertEqual(expert_cfg.gate_config.model_config.hidden_dim, 25)
        self.assertEqual(expert_cfg.halting_config.threshold, 0.77)
        self.assertEqual(
            expert_cfg.halting_config.halting_gate_config.hidden_dim,
            27,
        )

    def test_expert_controller_stacks_inherit_expert_stack_by_default(self):
        inherited_cfg = self.experts_preset(
            expert_stack_hidden_dim=31,
            expert_gate_flag=True,
            expert_gate_stack_hidden_dim=62,
            expert_recurrent_flag=True,
            expert_recurrent_gate_flag=True,
            expert_recurrent_gate_stack_hidden_dim=93,
        )
        inherited_expert_cfg = self._expert_model_config(inherited_cfg)

        self.assertIsInstance(inherited_expert_cfg, RecurrentLayerConfig)
        self.assertEqual(
            inherited_expert_cfg.block_config.layer_config.gate_config.model_config.hidden_dim,
            31,
        )
        self.assertEqual(
            inherited_expert_cfg.gate_config.model_config.hidden_dim,
            31,
        )

        custom_cfg = self.experts_preset(
            expert_stack_hidden_dim=31,
            expert_gate_flag=True,
            expert_gate_stack_independent_flag=True,
            expert_gate_stack_hidden_dim=62,
            expert_recurrent_flag=True,
            expert_recurrent_gate_flag=True,
            expert_recurrent_gate_stack_independent_flag=True,
            expert_recurrent_gate_stack_hidden_dim=93,
        )
        custom_expert_cfg = self._expert_model_config(custom_cfg)

        self.assertIsInstance(custom_expert_cfg, RecurrentLayerConfig)
        self.assertEqual(
            custom_expert_cfg.block_config.layer_config.gate_config.model_config.hidden_dim,
            62,
        )
        self.assertEqual(
            custom_expert_cfg.gate_config.model_config.hidden_dim,
            93,
        )

    def test_memory_enabled_forward_pass(self):
        cfg = self.experts_preset(
            input_dim=8,
            hidden_dim=8,
            output_dim=4,
            stack_num_layers=2,
            memory_flag=True,
        )
        model = Model(cfg)

        output = model(torch.randn(2, 1, 2, 4))
        logits = output[0] if isinstance(output, tuple) else output

        self.assertEqual(logits.shape, (2, 4))

    def test_controller_memory_presets_wire_expected_stack_configs(self):
        expected_controllers = {
            ExperimentPreset.MEMORY: (False, False, True),
            ExperimentPreset.GATING_MEMORY: (True, False, True),
            ExperimentPreset.HALTING_MEMORY: (False, True, True),
            ExperimentPreset.GATING_HALTING_MEMORY: (True, True, True),
        }

        for preset, (
            expected_gate,
            expected_halting,
            expected_memory,
        ) in expected_controllers.items():
            with self.subTest(preset=preset.name):
                cfg = ExperimentPresets().get_config(preset)[0]
                stack_cfg = cfg.experiment_config.model_config.stack_config

                self.assertEqual(
                    stack_cfg.layer_config.gate_config is not None,
                    expected_gate,
                )
                self.assertEqual(
                    stack_cfg.layer_config.halting_config is not None,
                    expected_halting,
                )
                self.assertEqual(
                    stack_cfg.shared_memory_config is not None,
                    expected_memory,
                )

    def test_residual_and_post_norm_presets_wire_expert_stack_config(self):
        cases = {
            ExperimentPreset.RESIDUAL: (
                ResidualConnectionOptions.RESIDUAL,
                config.STACK_LAYER_NORM_POSITION,
                False,
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
                False,
            ),
            ExperimentPreset.RESIDUAL_POST_NORM: (
                ResidualConnectionOptions.RESIDUAL,
                LayerNormPositionOptions.AFTER,
                False,
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
                False,
            ),
            ExperimentPreset.RESIDUAL_HALTING: (
                ResidualConnectionOptions.RESIDUAL,
                config.STACK_LAYER_NORM_POSITION,
                False,
                True,
                False,
                False,
            ),
            ExperimentPreset.RESIDUAL_MEMORY: (
                ResidualConnectionOptions.RESIDUAL,
                config.STACK_LAYER_NORM_POSITION,
                False,
                False,
                True,
                False,
            ),
            ExperimentPreset.RECURRENT_RESIDUAL: (
                ResidualConnectionOptions.RESIDUAL,
                config.STACK_LAYER_NORM_POSITION,
                False,
                False,
                False,
                True,
            ),
            ExperimentPreset.RECURRENT_POST_NORM: (
                ResidualConnectionOptions.DISABLED,
                LayerNormPositionOptions.AFTER,
                False,
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
            expected_recurrent,
        ) in cases.items():
            with self.subTest(preset=preset.name):
                cfg = ExperimentPresets().get_config(preset)[0]
                model_cfg = cfg.experiment_config.model_config
                stack_cfg = self._moe_model_config(cfg).stack_config
                layer_cfg = stack_cfg.layer_config

                self.assertEqual(
                    isinstance(model_cfg, RecurrentLayerConfig),
                    expected_recurrent,
                )
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
                    stack_cfg.shared_memory_config is not None,
                    expected_memory,
                )

    def test_gate_options_propagate_to_outer_stack_and_recurrent_wrapper(self):
        cfg = self.experts_preset(
            recurrent_flag=True,
            stack_gate_flag=True,
            recurrent_gate_flag=True,
            gate_option=LayerGateOptions.MULTIPLIER,
            recurrent_gate_option=LayerGateOptions.MULTIPLIER,
        )
        recurrent_cfg = cfg.experiment_config.model_config
        stack_cfg = recurrent_cfg.block_config.stack_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            stack_cfg.layer_config.gate_config.option,
            LayerGateOptions.MULTIPLIER,
        )
        self.assertEqual(recurrent_cfg.gate_config.option, LayerGateOptions.MULTIPLIER)

    def test_recurrent_layer_norm_position_defaults_disabled_and_uses_override(self):
        default_cfg = self.experts_preset(recurrent_flag=True)
        default_recurrent_cfg = default_cfg.experiment_config.model_config

        self.assertIsInstance(default_recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            default_recurrent_cfg.recurrent_layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )

        cfg = self.experts_preset(
            recurrent_flag=True,
            recurrent_layer_norm_position=LayerNormPositionOptions.DEFAULT,
        )
        recurrent_cfg = cfg.experiment_config.model_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            recurrent_cfg.recurrent_layer_norm_position,
            LayerNormPositionOptions.DEFAULT,
        )

    def test_recurrent_presets_wrap_full_moe_model(self):
        expected_controllers = {
            ExperimentPreset.RECURRENT: (False, False, False),
            ExperimentPreset.RECURRENT_GATING: (True, False, False),
            ExperimentPreset.RECURRENT_HALTING: (False, True, False),
            ExperimentPreset.RECURRENT_GATING_HALTING: (True, True, False),
            ExperimentPreset.RECURRENT_MEMORY: (False, False, True),
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
                self.assertEqual(recurrent_cfg.max_steps, config.RECURRENT_MAX_STEPS)
                self.assertIsInstance(
                    recurrent_cfg.block_config,
                    MixtureOfExpertsModelConfig,
                )
                self.assertEqual(recurrent_cfg.gate_config is not None, expected_gate)
                self.assertEqual(
                    recurrent_cfg.halting_config is not None,
                    expected_halting,
                )
                self.assertEqual(
                    recurrent_cfg.block_config.stack_config.shared_memory_config
                    is not None,
                    expected_memory,
                )
                inner_layer_config = (
                    recurrent_cfg.block_config.stack_config.layer_config
                )
                self.assertIsNone(inner_layer_config.gate_config)
                self.assertIsNone(inner_layer_config.halting_config)
                self.assertIsInstance(
                    inner_layer_config.layer_model_config.expert_model_config,
                    LayerStackConfig,
                )

    def test_hidden_expert_adaptive_options_build_expected_config_types(self):
        cfg = self.experts_preset(
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        augmentation_config = self._expert_augmentation_config(cfg)

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
        self.assertEqual(
            augmentation_config.mask_config.mask_dimension_option,
            MaskDimensionOptions.ROW,
        )

    def test_individual_linear_adaptive_presets_wire_expert_augmentation_type(self):
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
                    augmentation_config = self._expert_augmentation_config(cfg)

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

    def test_specialized_weight_presets_wire_expert_weight_knobs(self):
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
                weight_config = self._expert_augmentation_config(cfg).weight_config

                self.assertIsInstance(weight_config, DualModelDynamicWeightConfig)
                for field, expected in expected_values.items():
                    self.assertEqual(getattr(weight_config, field), expected)

    def test_linear_adaptive_combination_presets_wire_expert_config(self):
        presets = ExperimentPresets()
        cases = [
            {
                "preset": ExperimentPreset.FULL_STACK,
                "config_role": "full stack",
                "full_stack": True,
            },
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
                moe_model_cfg = self._moe_model_config(cfg)
                layer_cfg = moe_model_cfg.stack_config.layer_config
                augmentation_config = self._expert_augmentation_config(cfg)

                if case.get("recurrent"):
                    self.assertIsInstance(model_cfg, RecurrentLayerConfig)
                if case.get("full_stack"):
                    self._assert_full_adaptive_augmentation(augmentation_config)
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
                    self.assertIsNotNone(
                        moe_model_cfg.stack_config.shared_memory_config
                    )
                if "layer_norm" in case:
                    self.assertEqual(layer_cfg.layer_norm_position, case["layer_norm"])

    def test_router_adaptive_options_are_independent_from_hidden_options(self):
        cfg = self.experts_preset(
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            router_weight_option=LayeredWeightedBankDynamicWeightConfig,
        )

        hidden_augmentation = self._expert_augmentation_config(cfg)
        router_augmentation = self._router_augmentation_config(cfg)

        self.assertIsInstance(
            hidden_augmentation.weight_config,
            DualModelDynamicWeightConfig,
        )
        self.assertIsInstance(
            hidden_augmentation.bias_config,
            AdditiveDynamicBiasConfig,
        )
        self.assertIsInstance(
            hidden_augmentation.diagonal_config,
            CombinedDynamicDiagonalConfig,
        )
        self.assertIsInstance(
            hidden_augmentation.mask_config,
            WeightInformedScoreAxisMaskConfig,
        )
        self.assertIsInstance(
            router_augmentation.weight_config,
            LayeredWeightedBankDynamicWeightConfig,
        )
        self.assertIsNone(router_augmentation.bias_config)
        self.assertIsNone(router_augmentation.diagonal_config)
        self.assertIsNone(router_augmentation.mask_config)

    def test_boundary_adaptive_options_are_independent_from_hidden_and_router(self):
        cfg = self.experts_preset(
            weight_option=LowRankDynamicWeightConfig,
            router_weight_option=LayeredWeightedBankDynamicWeightConfig,
            input_layer_weight_option=DualModelDynamicWeightConfig,
            output_layer_bias_option=AdditiveDynamicBiasConfig,
        )

        input_augmentation = (
            cfg.experiment_config.input_model_config.layer_model_config.adaptive_augmentation_config
        )
        output_augmentation = (
            cfg.experiment_config.output_model_config.layer_model_config.adaptive_augmentation_config
        )
        hidden_augmentation = self._expert_augmentation_config(cfg)
        router_augmentation = self._router_augmentation_config(cfg)

        self.assertIsInstance(
            hidden_augmentation.weight_config,
            LowRankDynamicWeightConfig,
        )
        self.assertIsInstance(
            router_augmentation.weight_config,
            LayeredWeightedBankDynamicWeightConfig,
        )
        self.assertIsInstance(
            input_augmentation.weight_config,
            DualModelDynamicWeightConfig,
        )
        self.assertIsNone(input_augmentation.bias_config)
        self.assertIsNone(output_augmentation.weight_config)
        self.assertIsInstance(
            output_augmentation.bias_config,
            AdditiveDynamicBiasConfig,
        )

    def test_adaptive_generator_stack_overrides_respect_independent_flags(self):
        inherited_cfg = self.experts_preset(
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
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
            weight_generator_stack_hidden_dim=31,
            bias_generator_stack_hidden_dim=41,
            diagonal_generator_stack_hidden_dim=51,
            mask_generator_stack_hidden_dim=61,
        )
        inherited_augmentation = self._expert_augmentation_config(inherited_cfg)

        self._assert_generator_stack(
            inherited_augmentation.model_config,
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
            inherited_augmentation.weight_config,
            inherited_augmentation.bias_config,
            inherited_augmentation.diagonal_config,
            inherited_augmentation.mask_config,
        ):
            self.assertIsNone(component_config.model_config)

        custom_cfg = self.experts_preset(
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            adaptive_generator_stack_hidden_dim=21,
            adaptive_generator_stack_num_layers=3,
            adaptive_generator_stack_activation=ActivationOptions.RELU,
            adaptive_generator_stack_layer_norm_position=(
                LayerNormPositionOptions.AFTER
            ),
            weight_generator_stack_independent_flag=True,
            weight_generator_stack_hidden_dim=31,
            weight_generator_stack_num_layers=4,
            weight_generator_stack_activation=ActivationOptions.SILU,
            weight_generator_stack_layer_norm_position=LayerNormPositionOptions.BEFORE,
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
            mask_generator_stack_activation=ActivationOptions.ELU,
        )
        custom_augmentation = self._expert_augmentation_config(custom_cfg)

        self._assert_generator_stack(
            custom_augmentation.weight_config.model_config,
            hidden_dim=31,
            num_layers=4,
            activation=ActivationOptions.SILU,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            dropout_probability=0.15,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=True,
            bias_flag=False,
        )
        self.assertEqual(custom_augmentation.bias_config.model_config.hidden_dim, 41)
        self.assertEqual(
            custom_augmentation.bias_config.model_config.layer_config.activation,
            ActivationOptions.MISH,
        )
        self.assertEqual(
            custom_augmentation.diagonal_config.model_config.hidden_dim,
            51,
        )
        self.assertEqual(
            custom_augmentation.diagonal_config.model_config.layer_config.activation,
            ActivationOptions.TANH,
        )
        self.assertEqual(custom_augmentation.mask_config.model_config.hidden_dim, 61)
        self.assertEqual(
            custom_augmentation.mask_config.model_config.layer_config.activation,
            ActivationOptions.ELU,
        )

    def test_router_adaptive_generator_stacks_inherit_adaptive_defaults(self):
        inherited_cfg = self.experts_preset(
            router_weight_option=DualModelDynamicWeightConfig,
            router_bias_option=AdditiveDynamicBiasConfig,
            router_diagonal_option=CombinedDynamicDiagonalConfig,
            router_row_mask_option=WeightInformedScoreAxisMaskConfig,
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
            router_stack_hidden_dim=64,
            router_stack_num_layers=5,
            router_stack_activation=ActivationOptions.MISH,
            router_weight_generator_stack_hidden_dim=31,
            router_bias_generator_stack_hidden_dim=41,
            router_diagonal_generator_stack_hidden_dim=51,
            router_mask_generator_stack_hidden_dim=61,
        )
        inherited_augmentation = self._router_augmentation_config(inherited_cfg)

        self._assert_generator_stack(
            inherited_augmentation.model_config,
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
            inherited_augmentation.weight_config,
            inherited_augmentation.bias_config,
            inherited_augmentation.diagonal_config,
            inherited_augmentation.mask_config,
        ):
            self.assertIsNone(component_config.model_config)

        custom_cfg = self.experts_preset(
            router_weight_option=DualModelDynamicWeightConfig,
            router_bias_option=AdditiveDynamicBiasConfig,
            router_diagonal_option=CombinedDynamicDiagonalConfig,
            router_row_mask_option=WeightInformedScoreAxisMaskConfig,
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
            router_stack_hidden_dim=64,
            router_stack_num_layers=5,
            router_stack_activation=ActivationOptions.MISH,
            router_weight_generator_stack_independent_flag=True,
            router_weight_generator_stack_hidden_dim=31,
            router_weight_generator_stack_num_layers=4,
            router_weight_generator_stack_activation=ActivationOptions.SILU,
            router_weight_generator_stack_layer_norm_position=(
                LayerNormPositionOptions.BEFORE
            ),
            router_weight_generator_stack_dropout_probability=0.15,
            router_weight_generator_stack_last_layer_bias_option=(
                LastLayerBiasOptions.DISABLED
            ),
            router_weight_generator_stack_apply_output_pipeline_flag=True,
            router_weight_generator_stack_bias_flag=False,
            router_bias_generator_stack_independent_flag=True,
            router_bias_generator_stack_hidden_dim=41,
            router_bias_generator_stack_activation=ActivationOptions.MISH,
            router_diagonal_generator_stack_independent_flag=True,
            router_diagonal_generator_stack_hidden_dim=51,
            router_diagonal_generator_stack_activation=ActivationOptions.TANH,
            router_mask_generator_stack_independent_flag=True,
            router_mask_generator_stack_hidden_dim=61,
            router_mask_generator_stack_activation=ActivationOptions.ELU,
        )
        custom_augmentation = self._router_augmentation_config(custom_cfg)

        self._assert_generator_stack(
            custom_augmentation.weight_config.model_config,
            hidden_dim=31,
            num_layers=4,
            activation=ActivationOptions.SILU,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            dropout_probability=0.15,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=True,
            bias_flag=False,
        )
        self.assertEqual(custom_augmentation.bias_config.model_config.hidden_dim, 41)
        self.assertEqual(
            custom_augmentation.bias_config.model_config.layer_config.activation,
            ActivationOptions.MISH,
        )
        self.assertEqual(
            custom_augmentation.bias_config.model_config.num_layers,
            3,
        )
        self.assertEqual(
            custom_augmentation.diagonal_config.model_config.hidden_dim,
            51,
        )
        self.assertEqual(
            custom_augmentation.diagonal_config.model_config.layer_config.activation,
            ActivationOptions.TANH,
        )
        self.assertEqual(custom_augmentation.mask_config.model_config.hidden_dim, 61)
        self.assertEqual(
            custom_augmentation.mask_config.model_config.layer_config.activation,
            ActivationOptions.ELU,
        )

    def test_new_adaptive_moe_presets_wire_config(self):
        presets = ExperimentPresets()
        cases = [
            {
                "preset": ExperimentPreset.ADAPTIVE_SHARED_ROUTER,
                "config_role": "adaptive shared router",
                "model_routing": RoutingInitializationMode.SHARED,
                "layer_routing": RoutingInitializationMode.SHARED,
                "expert_weight": DualModelDynamicWeightConfig,
            },
            {
                "preset": ExperimentPreset.ADAPTIVE_AFTER_WEIGHT,
                "config_role": "adaptive after weight",
                "weighting_position": (ExpertWeightingPositionOptions.AFTER_EXPERTS),
                "expert_weight": DualModelDynamicWeightConfig,
            },
            {
                "preset": ExperimentPreset.ADAPTIVE_TOP1_SWITCH,
                "config_role": "adaptive top1 switch",
                "top_k": 1,
                "normalize_probabilities": False,
                "switch_loss_weight": 0.1,
                "expert_weight": DualModelDynamicWeightConfig,
            },
            {
                "preset": ExperimentPreset.ADAPTIVE_FULL_SHARED,
                "config_role": "adaptive full shared",
                "model_routing": RoutingInitializationMode.SHARED,
                "layer_routing": RoutingInitializationMode.SHARED,
                "full_adaptive": True,
            },
            {
                "preset": ExperimentPreset.ADAPTIVE_FULL_CAPACITY,
                "config_role": "adaptive full capacity",
                "top_k": 1,
                "capacity_factor": 1.0,
                "normalize_probabilities": False,
                "dropped_token_behavior": DroppedTokenOptions.ZEROS,
                "full_adaptive": True,
            },
            {
                "preset": ExperimentPreset.ADAPTIVE_BANK_ROUTER,
                "config_role": "adaptive bank router",
                "model_routing": RoutingInitializationMode.SHARED,
                "layer_routing": RoutingInitializationMode.SHARED,
                "router_weight": LayeredWeightedBankDynamicWeightConfig,
            },
        ]

        for case in cases:
            preset = case["preset"]
            with self.subTest(
                preset=preset.name,
                expected_config_role=case["config_role"],
            ):
                cfg = presets.get_config(preset)[0]
                moe_model_cfg = cfg.experiment_config.model_config
                moe_layer_cfg = self._moe_layer_config(cfg)
                expert_augmentation_config = self._expert_augmentation_config(cfg)

                if "model_routing" in case:
                    self.assertEqual(
                        moe_model_cfg.routing_initialization_mode,
                        case["model_routing"],
                    )
                if "layer_routing" in case:
                    self.assertEqual(
                        moe_layer_cfg.routing_initialization_mode,
                        case["layer_routing"],
                    )
                if "weighting_position" in case:
                    self.assertEqual(
                        moe_layer_cfg.weighting_position_option,
                        case["weighting_position"],
                    )
                if "top_k" in case:
                    self.assertEqual(moe_layer_cfg.top_k, case["top_k"])
                    self.assertEqual(moe_model_cfg.top_k, case["top_k"])
                if "capacity_factor" in case:
                    self.assertEqual(
                        moe_layer_cfg.capacity_factor,
                        case["capacity_factor"],
                    )
                if "normalize_probabilities" in case:
                    self.assertEqual(
                        moe_layer_cfg.sampler_config.normalize_probabilities_flag,
                        case["normalize_probabilities"],
                    )
                if "switch_loss_weight" in case:
                    self.assertEqual(
                        moe_layer_cfg.sampler_config.switch_loss_weight,
                        case["switch_loss_weight"],
                    )
                if "dropped_token_behavior" in case:
                    self.assertEqual(
                        moe_layer_cfg.dropped_token_behavior,
                        case["dropped_token_behavior"],
                    )
                if "expert_weight" in case:
                    self.assertIsInstance(
                        expert_augmentation_config.weight_config,
                        case["expert_weight"],
                    )
                if case.get("full_adaptive"):
                    self._assert_full_adaptive_augmentation(expert_augmentation_config)
                if "router_weight" in case:
                    router_augmentation_config = self._router_augmentation_config(cfg)
                    self.assertIsInstance(
                        router_augmentation_config.weight_config,
                        case["router_weight"],
                    )

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
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

    def experts_stack_source(
        self,
        options: ExpertsSubmoduleStackOptions,
        *,
        independent_flag: bool = True,
    ) -> ExpertsSubmoduleStackSource:
        return ExpertsSubmoduleStackSource(
            independent_flag=independent_flag,
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            residual_connection_option=options.residual_connection_option,
            dropout_probability=options.dropout_probability,
            bias_flag=options.bias_flag,
        )

    def adaptive_generator_stack_source(
        self,
        *,
        independent_flag: bool = True,
        hidden_dim: int | None = None,
        layer_norm_position: LayerNormPositionOptions | None = None,
        num_layers: int | None = None,
        activation: ActivationOptions | None = None,
        residual_connection_option: ResidualConnectionOptions | None = None,
        dropout_probability: float | None = None,
        last_layer_bias_option: LastLayerBiasOptions | None = None,
        apply_output_pipeline_flag: bool | None = None,
        bias_flag: bool | None = None,
    ) -> AdaptiveGeneratorStackSource:
        return AdaptiveGeneratorStackSource(
            independent_flag=independent_flag,
            hidden_dim=hidden_dim,
            layer_norm_position=layer_norm_position,
            num_layers=num_layers,
            activation=activation,
            residual_connection_option=residual_connection_option,
            dropout_probability=dropout_probability,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            bias_flag=bias_flag,
        )

    def flat_stack_kwargs(
        self,
        prefix: str,
        options: ExpertsSubmoduleStackOptions,
        *,
        bias_name: str,
    ) -> dict[str, object]:
        return {
            f"{prefix}_hidden_dim": options.hidden_dim,
            f"{prefix}_num_layers": options.num_layers,
            f"{prefix}_last_layer_bias_option": options.last_layer_bias_option,
            f"{prefix}_apply_output_pipeline_flag": (
                options.apply_output_pipeline_flag
            ),
            f"{prefix}_activation": options.activation,
            f"{prefix}_layer_norm_position": options.layer_norm_position,
            f"{prefix}_residual_connection_option": (
                options.residual_connection_option
            ),
            f"{prefix}_dropout_probability": options.dropout_probability,
            bias_name: options.bias_flag,
        }

    def flat_adaptive_source_kwargs(
        self,
        prefix: str,
        source: AdaptiveGeneratorStackSource,
    ) -> dict[str, object]:
        return {
            f"{prefix}_independent_flag": source.independent_flag,
            f"{prefix}_hidden_dim": source.hidden_dim,
            f"{prefix}_layer_norm_position": source.layer_norm_position,
            f"{prefix}_num_layers": source.num_layers,
            f"{prefix}_activation": source.activation,
            f"{prefix}_residual_connection_option": (source.residual_connection_option),
            f"{prefix}_dropout_probability": source.dropout_probability,
            f"{prefix}_last_layer_bias_option": source.last_layer_bias_option,
            f"{prefix}_apply_output_pipeline_flag": (source.apply_output_pipeline_flag),
            f"{prefix}_bias_flag": source.bias_flag,
        }

    def flat_adaptive_weight_kwargs(
        self,
        options: HiddenAdaptiveWeightOptions,
        *,
        option_prefix: str,
        stack_prefix: str,
    ) -> dict[str, object]:
        return {
            f"{option_prefix}_generator_depth": options.generator_depth,
            f"{option_prefix}_option_flag": options.option_flag,
            f"{option_prefix}_option": options.option,
            f"{option_prefix}_normalization_option": options.normalization_option,
            f"{option_prefix}_normalization_position_option": (
                options.normalization_position_option
            ),
            f"{option_prefix}_decay_schedule": options.decay_schedule,
            f"{option_prefix}_decay_rate": options.decay_rate,
            f"{option_prefix}_decay_warmup_batches": (options.decay_warmup_batches),
            f"{option_prefix}_bank_expansion_factor": (options.bank_expansion_factor),
            **self.flat_adaptive_source_kwargs(
                stack_prefix,
                options.generator_stack_source,
            ),
        }

    def flat_adaptive_bias_kwargs(
        self,
        options: HiddenAdaptiveBiasOptions,
        *,
        option_prefix: str,
        stack_prefix: str,
    ) -> dict[str, object]:
        return {
            f"{option_prefix}_option_flag": options.option_flag,
            f"{option_prefix}_option": options.option,
            f"{option_prefix}_decay_schedule": options.decay_schedule,
            f"{option_prefix}_decay_rate": options.decay_rate,
            f"{option_prefix}_decay_warmup_batches": options.decay_warmup_batches,
            f"{option_prefix}_bank_expansion_factor": (options.bank_expansion_factor),
            **self.flat_adaptive_source_kwargs(
                stack_prefix,
                options.generator_stack_source,
            ),
        }

    def flat_adaptive_diagonal_kwargs(
        self,
        options: HiddenAdaptiveDiagonalOptions,
        *,
        option_prefix: str,
        stack_prefix: str,
    ) -> dict[str, object]:
        return {
            f"{option_prefix}_option_flag": options.option_flag,
            f"{option_prefix}_option": options.option,
            **self.flat_adaptive_source_kwargs(
                stack_prefix,
                options.generator_stack_source,
            ),
        }

    def flat_adaptive_mask_kwargs(
        self,
        options: HiddenAdaptiveMaskOptions,
        *,
        option_prefix: str,
        stack_prefix: str,
    ) -> dict[str, object]:
        prefix = f"{option_prefix}_" if option_prefix else ""
        return {
            f"{prefix}mask_option_flag": options.option_flag,
            f"{prefix}row_mask_option": options.row_mask_option,
            f"{prefix}mask_dimension_option": options.mask_dimension_option,
            f"{prefix}mask_threshold": options.mask_threshold,
            f"{prefix}mask_surrogate_scale": options.mask_surrogate_scale,
            f"{prefix}mask_floor": options.mask_floor,
            f"{prefix}mask_transition_width": options.mask_transition_width,
            **self.flat_adaptive_source_kwargs(
                stack_prefix,
                options.generator_stack_source,
            ),
        }

    def _moe_model_config(self, cfg) -> MixtureOfExpertsModelConfig:
        model_config = cfg.experiment_config.model_config
        if isinstance(model_config, RecurrentLayerConfig):
            model_config = model_config.block_config
        return model_config

    def _moe_layer_config(self, cfg) -> MixtureOfExpertsConfig:
        return self._moe_model_config(cfg).stack_config.layer_config.layer_model_config

    def _expert_model_config(self, cfg):
        return self._moe_layer_config(cfg).expert_model_config

    def _expert_stack_config(self, cfg) -> LayerStackConfig:
        expert_cfg = self._expert_model_config(cfg)
        if isinstance(expert_cfg, RecurrentLayerConfig):
            expert_cfg = expert_cfg.block_config
        return expert_cfg

    def _expert_augmentation_config(self, cfg):
        return self._expert_stack_config(
            cfg
        ).layer_config.layer_model_config.adaptive_augmentation_config

    def _router_augmentation_config(self, cfg):
        return self._moe_model_config(
            cfg
        ).sampler_config.router_config.model_config.layer_config.layer_model_config.adaptive_augmentation_config

    def _assert_empty_adaptive_augmentation(self, augmentation_config) -> None:
        self.assertIsNone(augmentation_config.weight_config)
        self.assertIsNone(augmentation_config.bias_config)
        self.assertIsNone(augmentation_config.diagonal_config)
        self.assertIsNone(augmentation_config.mask_config)

    def _assert_full_adaptive_augmentation(self, augmentation_config) -> None:
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
        stack_cfg: LayerStackConfig,
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
        self.assertEqual(stack_cfg.hidden_dim, hidden_dim)
        self.assertEqual(stack_cfg.num_layers, num_layers)
        self.assertEqual(stack_cfg.layer_config.activation, activation)
        self.assertEqual(
            stack_cfg.layer_config.layer_norm_position,
            layer_norm_position,
        )
        self.assertEqual(
            stack_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
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


if __name__ == "__main__":
    unittest.main()

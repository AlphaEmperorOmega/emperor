import inspect
import unittest

import torch
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
from emperor.experiments.base import RandomSearch
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.config import (
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from emperor.memory.options import MemoryPositionOptions

import models.experts.linear.config as config
from models.experts.linear.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
    RuntimeOptions,
)
from models.experts.linear.config_builder import LinearConfigBuilder
from models.experts.linear.model import Model
from models.experts.linear.presets import ExperimentPreset, ExperimentPresets
from models.experts.linear.runtime_defaults import runtime_from_flat
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


import models.experts.linear.dataset_options as dataset_options
class TestLinearModel(unittest.TestCase):
    def experts_preset(self, **kwargs):
        return ExperimentPresets()._preset(**kwargs)

    def test_all_presets_forward_one_mnist_batch(self):
        batch_size = 4
        presets = ExperimentPresets()
        dataset = dataset_options.DATASET_OPTIONS_BY_TASK[dataset_options.DEFAULT_EXPERIMENT_TASK][0]

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                output = model(X)
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_baseline_forwards_all_datasets(self):
        batch_size = 4
        presets = ExperimentPresets()

        for dataset in dataset_options.DATASET_OPTIONS_BY_TASK[dataset_options.DEFAULT_EXPERIMENT_TASK]:
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(ExperimentPreset.BASELINE, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                output = model(X)
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_all_presets_train_one_epoch(self):
        presets = ExperimentPresets()
        dataset = dataset_options.DATASET_OPTIONS_BY_TASK[dataset_options.DEFAULT_EXPERIMENT_TASK][0]

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset, dataset)[0]
                model = Model(cfg)
                datamodule = RandomImageClassificationDataModule(dataset)

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def test_local_preset_catalog_has_stable_endpoints(self):
        self.assertEqual(ExperimentPreset.BASELINE.value, 1)
        self.assertEqual(ExperimentPreset.RECURRENT_POST_NORM.value, 32)
        self.assertEqual(len(ExperimentPreset.names()), 32)

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )

    def test_boundary_configs_are_separate_linear_layers(self):
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

        self.assertIsNot(exp_cfg.input_model_config, exp_cfg.model_config)
        self.assertIsNot(exp_cfg.output_model_config, exp_cfg.model_config)
        self.assertIsNot(exp_cfg.input_model_config, exp_cfg.output_model_config)

        cases = (
            (
                "input",
                exp_cfg.input_model_config,
                ActivationOptions.MISH,
                LayerNormPositionOptions.AFTER,
                0.13,
            ),
            (
                "output",
                exp_cfg.output_model_config,
                ActivationOptions.DISABLED,
                LayerNormPositionOptions.DISABLED,
                0.0,
            ),
        )
        for (
            label,
            boundary_cfg,
            expected_activation,
            expected_norm,
            expected_dropout,
        ) in cases:
            with self.subTest(label=label):
                self.assertIsInstance(boundary_cfg, LayerConfig)
                self.assertEqual(boundary_cfg.activation, expected_activation)
                self.assertEqual(boundary_cfg.layer_norm_position, expected_norm)
                self.assertEqual(
                    boundary_cfg.residual_connection_option,
                    ResidualConnectionOptions.DISABLED,
                )
                self.assertEqual(boundary_cfg.dropout_probability, expected_dropout)
                self.assertIsNone(boundary_cfg.gate_config)
                self.assertIsNone(boundary_cfg.halting_config)
                self.assertIsNone(boundary_cfg.memory_config)
                self.assertIsInstance(
                    boundary_cfg.layer_model_config,
                    LinearLayerConfig,
                )
                self.assertTrue(boundary_cfg.layer_model_config.bias_flag)

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

    def test_preset_accepts_search_flags(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            dataset_options.DATASET_OPTIONS_BY_TASK[dataset_options.DEFAULT_EXPERIMENT_TASK][0],
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
            hidden_dim=stack_options.hidden_dim,
            num_layers=4,
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
            hidden_dim=stack_options.hidden_dim,
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
            hidden_dim=stack_options.hidden_dim,
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

        def stack_source(
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

        layer_controller_options = ExpertsLayerControllerOptions(
            stack_gate_flag=True,
            gate_option=LayerGateOptions.ADDITION,
            gate_activation=ActivationOptions.TANH,
            gate_stack_source=stack_source(gate_stack_options),
            stack_halting_flag=True,
            halting_threshold=0.63,
            halting_dropout=0.08,
            halting_hidden_state_mode=HaltingHiddenStateModeOptions.ACCUMULATED,
            halting_stack_source=stack_source(halting_stack_options),
            halting_output_dim=2,
        )
        dynamic_memory_options = ExpertsDynamicMemoryOptions(
            memory_flag=True,
            memory_option=WeightedDynamicMemoryConfig,
            memory_position_option=MemoryPositionOptions.BEFORE_AFFINE,
            memory_test_time_training_learning_rate=0.02,
            memory_test_time_training_num_inner_steps=2,
            memory_stack_source=stack_source(memory_stack_options),
        )
        recurrent_controller_options = ExpertsRecurrentControllerOptions(
            recurrent_flag=True,
            recurrent_max_steps=3,
            recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            recurrent_gate_flag=True,
            recurrent_gate_option=LayerGateOptions.MULTIPLIER,
            recurrent_gate_activation=ActivationOptions.SIGMOID,
            recurrent_gate_stack_source=stack_source(gate_stack_options),
            recurrent_halting_flag=True,
            recurrent_halting_threshold=0.71,
            recurrent_halting_dropout=0.09,
            recurrent_halting_hidden_state_mode=(
                HaltingHiddenStateModeOptions.ACCUMULATED
            ),
            recurrent_halting_stack_source=stack_source(halting_stack_options),
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
            "submodule_stack_hidden_dim": submodule_stack_options.hidden_dim,
            "submodule_stack_num_layers": submodule_stack_options.num_layers,
            "submodule_stack_last_layer_bias_option": (
                submodule_stack_options.last_layer_bias_option
            ),
            "submodule_stack_apply_output_pipeline_flag": (
                submodule_stack_options.apply_output_pipeline_flag
            ),
            "submodule_stack_activation": submodule_stack_options.activation,
            "submodule_stack_layer_norm_position": (
                submodule_stack_options.layer_norm_position
            ),
            "submodule_stack_residual_connection_option": (
                submodule_stack_options.residual_connection_option
            ),
            "submodule_stack_dropout_probability": (
                submodule_stack_options.dropout_probability
            ),
            "submodule_stack_bias_flag": submodule_stack_options.bias_flag,
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
            "expert_stack_num_layers": expert_stack_options.num_layers,
            "expert_stack_activation": expert_stack_options.activation,
            "expert_stack_residual_connection_option": (
                expert_stack_options.residual_connection_option
            ),
            "expert_stack_dropout_probability": (
                expert_stack_options.dropout_probability
            ),
            "expert_stack_layer_norm_position": (
                expert_stack_options.layer_norm_position
            ),
            "expert_stack_last_layer_bias_option": (
                expert_stack_options.last_layer_bias_option
            ),
            "expert_stack_apply_output_pipeline_flag": (
                expert_stack_options.apply_output_pipeline_flag
            ),
            "expert_bias_flag": expert_stack_options.bias_flag,
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
            "router_stack_hidden_dim": router_stack_options.hidden_dim,
            "router_stack_num_layers": router_stack_options.num_layers,
            "router_stack_activation": router_stack_options.activation,
            "router_stack_residual_connection_option": (
                router_stack_options.residual_connection_option
            ),
            "router_stack_dropout_probability": (
                router_stack_options.dropout_probability
            ),
            "router_stack_layer_norm_position": (
                router_stack_options.layer_norm_position
            ),
            "router_stack_last_layer_bias_option": (
                router_stack_options.last_layer_bias_option
            ),
            "router_stack_apply_output_pipeline_flag": (
                router_stack_options.apply_output_pipeline_flag
            ),
            "router_bias_flag": router_stack_options.bias_flag,
            "stack_gate_flag": layer_controller_options.stack_gate_flag,
            "gate_option": layer_controller_options.gate_option,
            "gate_activation": layer_controller_options.gate_activation,
            "gate_stack_independent_flag": True,
            "gate_stack_hidden_dim": gate_stack_options.hidden_dim,
            "gate_stack_layer_norm_position": (gate_stack_options.layer_norm_position),
            "gate_stack_num_layers": gate_stack_options.num_layers,
            "gate_stack_activation": gate_stack_options.activation,
            "gate_stack_residual_connection_option": (
                gate_stack_options.residual_connection_option
            ),
            "gate_stack_dropout_probability": (gate_stack_options.dropout_probability),
            "gate_stack_last_layer_bias_option": (
                gate_stack_options.last_layer_bias_option
            ),
            "gate_stack_apply_output_pipeline_flag": (
                gate_stack_options.apply_output_pipeline_flag
            ),
            "gate_stack_bias_flag": gate_stack_options.bias_flag,
            "stack_halting_flag": layer_controller_options.stack_halting_flag,
            "halting_threshold": layer_controller_options.halting_threshold,
            "halting_dropout": layer_controller_options.halting_dropout,
            "halting_hidden_state_mode": (
                layer_controller_options.halting_hidden_state_mode
            ),
            "halting_stack_independent_flag": True,
            "halting_stack_hidden_dim": halting_stack_options.hidden_dim,
            "halting_output_dim": layer_controller_options.halting_output_dim,
            "halting_stack_layer_norm_position": (
                halting_stack_options.layer_norm_position
            ),
            "halting_stack_num_layers": halting_stack_options.num_layers,
            "halting_stack_activation": halting_stack_options.activation,
            "halting_stack_residual_connection_option": (
                halting_stack_options.residual_connection_option
            ),
            "halting_stack_dropout_probability": (
                halting_stack_options.dropout_probability
            ),
            "halting_stack_last_layer_bias_option": (
                halting_stack_options.last_layer_bias_option
            ),
            "halting_stack_apply_output_pipeline_flag": (
                halting_stack_options.apply_output_pipeline_flag
            ),
            "halting_stack_bias_flag": halting_stack_options.bias_flag,
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
            "memory_stack_hidden_dim": memory_stack_options.hidden_dim,
            "memory_stack_layer_norm_position": (
                memory_stack_options.layer_norm_position
            ),
            "memory_stack_num_layers": memory_stack_options.num_layers,
            "memory_stack_activation": memory_stack_options.activation,
            "memory_stack_residual_connection_option": (
                memory_stack_options.residual_connection_option
            ),
            "memory_stack_dropout_probability": (
                memory_stack_options.dropout_probability
            ),
            "memory_stack_last_layer_bias_option": (
                memory_stack_options.last_layer_bias_option
            ),
            "memory_stack_apply_output_pipeline_flag": (
                memory_stack_options.apply_output_pipeline_flag
            ),
            "memory_stack_bias_flag": memory_stack_options.bias_flag,
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
            "recurrent_gate_stack_hidden_dim": gate_stack_options.hidden_dim,
            "recurrent_gate_stack_layer_norm_position": (
                gate_stack_options.layer_norm_position
            ),
            "recurrent_gate_stack_num_layers": gate_stack_options.num_layers,
            "recurrent_gate_stack_activation": gate_stack_options.activation,
            "recurrent_gate_stack_residual_connection_option": (
                gate_stack_options.residual_connection_option
            ),
            "recurrent_gate_stack_dropout_probability": (
                gate_stack_options.dropout_probability
            ),
            "recurrent_gate_stack_last_layer_bias_option": (
                gate_stack_options.last_layer_bias_option
            ),
            "recurrent_gate_stack_apply_output_pipeline_flag": (
                gate_stack_options.apply_output_pipeline_flag
            ),
            "recurrent_gate_stack_bias_flag": gate_stack_options.bias_flag,
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
            "recurrent_halting_stack_hidden_dim": halting_stack_options.hidden_dim,
            "recurrent_halting_stack_layer_norm_position": (
                halting_stack_options.layer_norm_position
            ),
            "recurrent_halting_stack_num_layers": halting_stack_options.num_layers,
            "recurrent_halting_stack_activation": halting_stack_options.activation,
            "recurrent_halting_stack_residual_connection_option": (
                halting_stack_options.residual_connection_option
            ),
            "recurrent_halting_stack_dropout_probability": (
                halting_stack_options.dropout_probability
            ),
            "recurrent_halting_stack_last_layer_bias_option": (
                halting_stack_options.last_layer_bias_option
            ),
            "recurrent_halting_stack_apply_output_pipeline_flag": (
                halting_stack_options.apply_output_pipeline_flag
            ),
            "recurrent_halting_stack_bias_flag": halting_stack_options.bias_flag,
        }

        flat_cfg = self.experts_preset(**flat_kwargs)
        grouped_builder = LinearConfigBuilder(
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
        )

        self.assertEqual(
            grouped_builder.submodule_stack_options,
            submodule_stack_options,
        )
        grouped_cfg = grouped_builder.build()
        self.assertEqual(flat_cfg, grouped_cfg)

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

    def test_runtime_is_primary_and_flat_builder_kwargs_are_rejected(self):
        parameters = inspect.signature(LinearConfigBuilder.__init__).parameters
        flat_names = {
            "memory_flag",
            "top_k",
            "gate_stack_independent_flag",
            "gate_stack_hidden_dim",
            "router_stack_independent_flag",
            "sampler_stack_independent_flag",
            "sampler_stack_options",
            "sampler_bias_flag",
            "shared_gate_config",
        }

        self.assertIn("runtime", parameters)
        self.assertLessEqual(
            set(parameters),
            {"self", "legacy_args", "runtime", "legacy_options"},
        )

        for name in flat_names:
            with self.subTest(name=name):
                self.assertNotIn(name, parameters)

        for kwargs in (
            {"memory_flag": True},
            {"top_k": 1},
            {"router_stack_independent_flag": True},
            {"sampler_stack_options": ExpertsSubmoduleStackOptions(
                hidden_dim=8,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                activation=ActivationOptions.RELU,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                bias_flag=True,
            )},
            {"shared_gate_config": self.shared_gate_config()},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    LinearConfigBuilder(**kwargs)

        for kwargs in (
            {"router_stack_independent_flag": True},
            {"sampler_stack_independent_flag": True},
            {"sampler_stack_hidden_dim": 13},
            {"sampler_bias_flag": False},
        ):
            with self.subTest(kwargs=kwargs):
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
        cfg = LinearConfigBuilder(runtime=runtime).build()

        self.assertEqual(cfg.batch_size, 3)
        self.assertEqual(cfg.learning_rate, 0.02)
        self.assertEqual(cfg.input_dim, 8)
        self.assertEqual(cfg.hidden_dim, 16)
        self.assertEqual(cfg.output_dim, 4)

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

    def test_disabled_memory_is_absent_from_stack_and_layer_configs(self):
        cfg = LinearConfigBuilder().build()
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

    def test_linear_family_presets_wire_expert_stack_config(self):
        presets = ExperimentPresets()
        cases = [
            {
                "preset": ExperimentPreset.RESIDUAL,
                "config_role": "expert stack residual",
                "residual": ResidualConnectionOptions.RESIDUAL,
            },
            {
                "preset": ExperimentPreset.POST_NORM,
                "config_role": "expert stack post norm",
                "layer_norm": LayerNormPositionOptions.AFTER,
            },
            {
                "preset": ExperimentPreset.RESIDUAL_POST_NORM,
                "config_role": "expert stack residual post norm",
                "residual": ResidualConnectionOptions.RESIDUAL,
                "layer_norm": LayerNormPositionOptions.AFTER,
            },
            {
                "preset": ExperimentPreset.RESIDUAL_GATING,
                "config_role": "expert stack residual gating",
                "residual": ResidualConnectionOptions.RESIDUAL,
                "gate": True,
            },
            {
                "preset": ExperimentPreset.RESIDUAL_HALTING,
                "config_role": "expert stack residual halting",
                "residual": ResidualConnectionOptions.RESIDUAL,
                "halting": True,
            },
            {
                "preset": ExperimentPreset.RESIDUAL_MEMORY,
                "config_role": "expert stack residual memory",
                "residual": ResidualConnectionOptions.RESIDUAL,
                "memory": True,
            },
            {
                "preset": ExperimentPreset.RECURRENT_RESIDUAL,
                "config_role": "recurrent expert stack residual",
                "recurrent": True,
                "residual": ResidualConnectionOptions.RESIDUAL,
            },
            {
                "preset": ExperimentPreset.RECURRENT_POST_NORM,
                "config_role": "recurrent expert stack post norm",
                "recurrent": True,
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
                if case.get("recurrent"):
                    self.assertIsInstance(model_cfg, RecurrentLayerConfig)
                    moe_model_cfg = model_cfg.block_config
                else:
                    self.assertIsInstance(model_cfg, MixtureOfExpertsModelConfig)
                    moe_model_cfg = model_cfg
                stack_cfg = moe_model_cfg.stack_config
                layer_cfg = stack_cfg.layer_config

                if "residual" in case:
                    self.assertEqual(
                        layer_cfg.residual_connection_option,
                        case["residual"],
                    )
                if "layer_norm" in case:
                    self.assertEqual(
                        layer_cfg.layer_norm_position,
                        case["layer_norm"],
                    )
                if case.get("gate"):
                    self.assertIsNotNone(layer_cfg.gate_config)
                if case.get("halting"):
                    self.assertIsNotNone(layer_cfg.halting_config)
                if case.get("memory"):
                    self.assertIsNotNone(stack_cfg.shared_memory_config)

    def test_new_moe_combination_presets_wire_config(self):
        presets = ExperimentPresets()
        cases = [
            {
                "preset": ExperimentPreset.SHARED_ROUTER_AFTER_WEIGHT,
                "config_role": "shared router after weight",
                "model_routing": RoutingInitializationMode.SHARED,
                "layer_routing": RoutingInitializationMode.SHARED,
                "weighting_position": (ExpertWeightingPositionOptions.AFTER_EXPERTS),
            },
            {
                "preset": ExperimentPreset.TOP1_SWITCH_AUX,
                "config_role": "top1 switch auxiliary loss",
                "top_k": 1,
                "normalize_probabilities": False,
                "switch_loss_weight": 0.1,
            },
            {
                "preset": ExperimentPreset.TOP2_BALANCED_AUX,
                "config_role": "top2 balanced auxiliary loss",
                "top_k": 2,
                "coefficient_of_variation_loss_weight": 0.1,
            },
            {
                "preset": ExperimentPreset.CAPACITY_TOP1_ZERO,
                "config_role": "capacity top1 zeros",
                "top_k": 1,
                "capacity_factor": 1.0,
                "normalize_probabilities": False,
                "dropped_token_behavior": DroppedTokenOptions.ZEROS,
            },
            {
                "preset": ExperimentPreset.CAPACITY_TOP1_IDENTITY,
                "config_role": "capacity top1 identity",
                "top_k": 1,
                "capacity_factor": 1.0,
                "normalize_probabilities": False,
                "dropped_token_behavior": DroppedTokenOptions.IDENTITY,
            },
            {
                "preset": ExperimentPreset.NOISY_SHARED_ROUTER,
                "config_role": "noisy shared router",
                "model_routing": RoutingInitializationMode.SHARED,
                "noisy_topk": True,
            },
            {
                "preset": ExperimentPreset.RESIDUAL_SHARED_ROUTER,
                "config_role": "residual shared router",
                "model_routing": RoutingInitializationMode.SHARED,
                "residual": ResidualConnectionOptions.RESIDUAL,
            },
            {
                "preset": ExperimentPreset.POST_NORM_AFTER_WEIGHT,
                "config_role": "post norm after weight",
                "layer_norm": LayerNormPositionOptions.AFTER,
                "weighting_position": (ExpertWeightingPositionOptions.AFTER_EXPERTS),
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
                layer_cfg = moe_model_cfg.stack_config.layer_config
                moe_layer_cfg = self._moe_layer_config(cfg)

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
                if "coefficient_of_variation_loss_weight" in case:
                    self.assertEqual(
                        moe_layer_cfg.sampler_config.coefficient_of_variation_loss_weight,
                        case["coefficient_of_variation_loss_weight"],
                    )
                if "capacity_factor" in case:
                    self.assertEqual(
                        moe_layer_cfg.capacity_factor,
                        case["capacity_factor"],
                    )
                if "dropped_token_behavior" in case:
                    self.assertEqual(
                        moe_layer_cfg.dropped_token_behavior,
                        case["dropped_token_behavior"],
                    )
                if case.get("noisy_topk"):
                    self.assertTrue(moe_model_cfg.sampler_config.noisy_topk_flag)
                    self.assertTrue(
                        moe_model_cfg.sampler_config.router_config.noisy_topk_flag
                    )
                    self.assertTrue(moe_layer_cfg.sampler_config.noisy_topk_flag)
                    self.assertTrue(
                        moe_layer_cfg.sampler_config.router_config.noisy_topk_flag
                    )
                if "residual" in case:
                    self.assertEqual(
                        layer_cfg.residual_connection_option,
                        case["residual"],
                    )
                if "layer_norm" in case:
                    self.assertEqual(
                        layer_cfg.layer_norm_position,
                        case["layer_norm"],
                    )

    def test_auxiliary_loss_presets_return_finite_loss(self):
        batch_size = 4
        dataset = dataset_options.DATASET_OPTIONS_BY_TASK[dataset_options.DEFAULT_EXPERIMENT_TASK][0]
        presets = ExperimentPresets()

        for preset in (
            ExperimentPreset.TOP1_SWITCH_AUX,
            ExperimentPreset.TOP2_BALANCED_AUX,
        ):
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                output = model(X)

                self.assertIsInstance(output, tuple)
                logits, auxiliary_loss = output
                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
                self.assertTrue(torch.isfinite(auxiliary_loss).item())

    def _moe_layer_config(self, cfg):
        model_config = cfg.experiment_config.model_config
        if isinstance(model_config, RecurrentLayerConfig):
            model_config = model_config.block_config
        return model_config.stack_config.layer_config.layer_model_config

    def _expert_model_config(self, cfg):
        return self._moe_layer_config(cfg).expert_model_config


if __name__ == "__main__":
    unittest.main()

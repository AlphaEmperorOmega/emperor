import inspect
import unittest

import torch

import models.experts.experts_linear.config as config

from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.layer import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.experiments.base import RandomSearch
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
from models.experts._builder_options import (
    ExpertsControllerStackOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
)
from models.experts.experts_linear.config_builder import ExpertsLinearConfigBuilder
from models.experts.experts_linear.model import Model
from models.experts.experts_linear.presets import ExperimentPreset, ExperimentPresets
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestExpertsLinearModel(unittest.TestCase):
    def test_all_presets_forward_one_mnist_batch(self):
        batch_size = 4
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
        batch_size = 4
        presets = ExperimentPresets()

        for dataset in config.DATASET_OPTIONS:
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(ExperimentPreset.BASELINE, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                output = model(X)
                logits = output[0] if isinstance(output, tuple) else output

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

    def test_preset_accepts_search_flags(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
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
        expert_stack_options = ExpertsControllerStackOptions(
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
        sampler_stack_options = ExpertsControllerStackOptions(
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
        gate_stack_options = ExpertsControllerStackOptions(
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
        halting_stack_options = ExpertsControllerStackOptions(
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
        layer_controller_options = ExpertsLayerControllerOptions(
            stack_gate_flag=True,
            gate_option=LayerGateOptions.ADDITION,
            gate_activation=ActivationOptions.TANH,
            gate_stack_options=gate_stack_options,
            stack_halting_flag=True,
            halting_threshold=0.63,
            halting_dropout=0.08,
            halting_hidden_state_mode=HaltingHiddenStateModeOptions.ACCUMULATED,
            halting_stack_options=halting_stack_options,
            halting_output_dim=2,
        )
        recurrent_controller_options = ExpertsRecurrentControllerOptions(
            recurrent_flag=True,
            recurrent_max_steps=3,
            recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            recurrent_gate_flag=True,
            recurrent_gate_option=LayerGateOptions.MULTIPLIER,
            recurrent_gate_activation=ActivationOptions.SIGMOID,
            recurrent_halting_flag=True,
        )
        flat_kwargs = {
            "batch_size": 3,
            "learning_rate": 0.02,
            "input_dim": 8,
            "output_dim": 4,
            "stack_hidden_dim": stack_options.hidden_dim,
            "stack_bias_flag": stack_options.bias_flag,
            "layer_norm_position": stack_options.layer_norm_position,
            "stack_num_layers": stack_options.num_layers,
            "stack_activation": stack_options.activation,
            "stack_residual_connection_option": (
                stack_options.residual_connection_option
            ),
            "stack_dropout_probability": stack_options.dropout_probability,
            "stack_last_layer_bias_option": (
                stack_options.last_layer_bias_option
            ),
            "stack_apply_output_pipeline_flag": (
                stack_options.apply_output_pipeline_flag
            ),
            "top_k": mixture_options.top_k,
            "num_experts": mixture_options.num_experts,
            "capacity_factor": mixture_options.capacity_factor,
            "dropped_token_behavior": mixture_options.dropped_token_behavior,
            "compute_expert_mixture_flag": (
                mixture_options.compute_expert_mixture_flag
            ),
            "weighted_parameters_flag": mixture_options.weighted_parameters_flag,
            "weighting_position_option": (
                mixture_options.weighting_position_option
            ),
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
            "sampler_filter_above_threshold": (
                sampler_options.filter_above_threshold
            ),
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
            "sampler_stack_num_layers": sampler_stack_options.num_layers,
            "sampler_stack_activation": sampler_stack_options.activation,
            "sampler_stack_residual_connection_option": (
                sampler_stack_options.residual_connection_option
            ),
            "sampler_stack_dropout_probability": (
                sampler_stack_options.dropout_probability
            ),
            "sampler_stack_layer_norm_position": (
                sampler_stack_options.layer_norm_position
            ),
            "sampler_stack_last_layer_bias_option": (
                sampler_stack_options.last_layer_bias_option
            ),
            "sampler_stack_apply_output_pipeline_flag": (
                sampler_stack_options.apply_output_pipeline_flag
            ),
            "sampler_bias_flag": sampler_stack_options.bias_flag,
            "stack_gate_flag": layer_controller_options.stack_gate_flag,
            "gate_option": layer_controller_options.gate_option,
            "gate_activation": layer_controller_options.gate_activation,
            "gate_stack_hidden_dim": gate_stack_options.hidden_dim,
            "gate_stack_layer_norm_position": (
                gate_stack_options.layer_norm_position
            ),
            "gate_stack_num_layers": gate_stack_options.num_layers,
            "gate_stack_activation": gate_stack_options.activation,
            "gate_stack_residual_connection_option": (
                gate_stack_options.residual_connection_option
            ),
            "gate_stack_dropout_probability": (
                gate_stack_options.dropout_probability
            ),
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
            "recurrent_flag": recurrent_controller_options.recurrent_flag,
            "recurrent_max_steps": (
                recurrent_controller_options.recurrent_max_steps
            ),
            "recurrent_layer_norm_position": (
                recurrent_controller_options.recurrent_layer_norm_position
            ),
            "recurrent_gate_flag": (
                recurrent_controller_options.recurrent_gate_flag
            ),
            "recurrent_gate_option": (
                recurrent_controller_options.recurrent_gate_option
            ),
            "recurrent_gate_activation": (
                recurrent_controller_options.recurrent_gate_activation
            ),
            "recurrent_halting_flag": (
                recurrent_controller_options.recurrent_halting_flag
            ),
        }

        flat_cfg = ExpertsLinearConfigBuilder(**flat_kwargs).build()
        grouped_cfg = ExpertsLinearConfigBuilder(
            batch_size=3,
            learning_rate=0.02,
            input_dim=8,
            output_dim=4,
            stack_options=stack_options,
            mixture_options=mixture_options,
            expert_stack_options=expert_stack_options,
            sampler_options=sampler_options,
            router_options=router_options,
            sampler_stack_options=sampler_stack_options,
            layer_controller_options=layer_controller_options,
            recurrent_controller_options=recurrent_controller_options,
        ).build()

        self.assertEqual(flat_cfg, grouped_cfg)

    def test_shared_gate_config_is_stored_on_stack_config(self):
        shared_gate_config = self.shared_gate_config()
        cfg = ExpertsLinearConfigBuilder(shared_gate_config=shared_gate_config).build()
        stack_cfg = cfg.experiment_config.model_config.stack_config

        self.assertIs(stack_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(stack_cfg.layer_config.gate_config)

    def test_shared_gate_config_rejects_enabled_stack_gate(self):
        cfg = ExpertsLinearConfigBuilder(
            stack_gate_flag=True,
            shared_gate_config=self.shared_gate_config(),
        ).build()

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Model(cfg)

    def test_shared_gate_config_allows_absent_stack_gate(self):
        shared_gate_config = self.shared_gate_config()
        cfg = ExpertsLinearConfigBuilder(
            stack_gate_flag=False,
            shared_gate_config=shared_gate_config,
        ).build()
        stack_cfg = cfg.experiment_config.model_config.stack_config

        self.assertIs(stack_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(stack_cfg.layer_config.gate_config)

    def test_controller_stack_builder_kwargs_are_canonical(self):
        parameters = inspect.signature(ExpertsLinearConfigBuilder.__init__).parameters
        expected_names = {
            "gate_stack_hidden_dim",
            "gate_stack_layer_norm_position",
            "gate_stack_bias_flag",
            "halting_stack_hidden_dim",
            "halting_stack_layer_norm_position",
            "halting_stack_bias_flag",
        }
        legacy_names = {name.replace("_stack_", "_") for name in expected_names}

        for name in expected_names:
            with self.subTest(name=name):
                self.assertIn(name, parameters)

        for name in legacy_names:
            with self.subTest(name=name):
                self.assertNotIn(name, parameters)

        legacy_gate_hidden_dim = "gate" + "_hidden_dim"
        with self.assertRaises(TypeError):
            ExpertsLinearConfigBuilder(**{legacy_gate_hidden_dim: 32})

    def test_controller_stack_overrides_use_canonical_names(self):
        cfg = ExpertsLinearConfigBuilder(
            stack_gate_flag=True,
            gate_stack_hidden_dim=32,
            gate_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            gate_stack_bias_flag=False,
            stack_halting_flag=True,
            halting_stack_hidden_dim=48,
            halting_stack_layer_norm_position=LayerNormPositionOptions.BEFORE,
            halting_stack_bias_flag=False,
        ).build()
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

    def test_gate_options_propagate_to_outer_stack_and_recurrent_wrapper(self):
        cfg = ExpertsLinearConfigBuilder(
            recurrent_flag=True,
            stack_gate_flag=True,
            recurrent_gate_flag=True,
            gate_option=LayerGateOptions.MULTIPLIER,
            recurrent_gate_option=LayerGateOptions.MULTIPLIER,
        ).build()
        recurrent_cfg = cfg.experiment_config.model_config
        stack_cfg = recurrent_cfg.block_config.stack_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            stack_cfg.layer_config.gate_config.option,
            LayerGateOptions.MULTIPLIER,
        )
        self.assertEqual(recurrent_cfg.gate_config.option, LayerGateOptions.MULTIPLIER)

    def test_recurrent_layer_norm_position_defaults_disabled_and_uses_override(self):
        default_cfg = ExpertsLinearConfigBuilder(recurrent_flag=True).build()
        default_recurrent_cfg = default_cfg.experiment_config.model_config

        self.assertIsInstance(default_recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            default_recurrent_cfg.recurrent_layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )

        cfg = ExpertsLinearConfigBuilder(
            recurrent_flag=True,
            recurrent_layer_norm_position=LayerNormPositionOptions.DEFAULT,
        ).build()
        recurrent_cfg = cfg.experiment_config.model_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            recurrent_cfg.recurrent_layer_norm_position,
            LayerNormPositionOptions.DEFAULT,
        )

    def test_recurrent_presets_wrap_full_moe_model(self):
        expected_controllers = {
            ExperimentPreset.RECURRENT: (False, False),
            ExperimentPreset.RECURRENT_GATING: (True, False),
            ExperimentPreset.RECURRENT_HALTING: (False, True),
            ExperimentPreset.RECURRENT_GATING_HALTING: (True, True),
        }

        for preset, (expected_gate, expected_halting) in expected_controllers.items():
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
                inner_layer_config = (
                    recurrent_cfg.block_config.stack_config.layer_config
                )
                self.assertIsNone(inner_layer_config.gate_config)
                self.assertIsNone(inner_layer_config.halting_config)

    def test_new_moe_combination_presets_wire_config(self):
        presets = ExperimentPresets()
        cases = [
            {
                "preset": ExperimentPreset.SHARED_ROUTER_AFTER_WEIGHT,
                "config_role": "shared router after weight",
                "model_routing": RoutingInitializationMode.SHARED,
                "layer_routing": RoutingInitializationMode.SHARED,
                "weighting_position": (
                    ExpertWeightingPositionOptions.AFTER_EXPERTS
                ),
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
                "weighting_position": (
                    ExpertWeightingPositionOptions.AFTER_EXPERTS
                ),
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
        dataset = config.DATASET_OPTIONS[0]
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


if __name__ == "__main__":
    unittest.main()

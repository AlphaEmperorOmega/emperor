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
from emperor.linears.core.config import LinearLayerConfig
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

    def test_shared_gate_config_is_stored_on_stack_config(self):
        shared_gate_config = self.shared_gate_config()
        cfg = ExpertsLinearConfigBuilder(shared_gate_config=shared_gate_config).build()
        stack_cfg = cfg.experiment_config.model_config.stack_config

        self.assertIs(stack_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(stack_cfg.layer_config.gate_config)

    def test_shared_gate_config_rejects_enabled_stack_gate(self):
        with self.assertRaises(ValueError):
            ExpertsLinearConfigBuilder(
                stack_gate_flag=True,
                shared_gate_config=self.shared_gate_config(),
            ).build()

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

        cfg = presets.get_config(ExperimentPreset.SHARED_ROUTER_AFTER_WEIGHT)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        moe_layer_cfg = moe_model_cfg.stack_config.layer_config.layer_model_config
        self.assertEqual(
            moe_model_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self.assertEqual(
            moe_layer_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self.assertEqual(
            moe_layer_cfg.weighting_position_option,
            ExpertWeightingPositionOptions.AFTER_EXPERTS,
        )

        cfg = presets.get_config(ExperimentPreset.TOP1_SWITCH_AUX)[0]
        moe_layer_cfg = self._moe_layer_config(cfg)
        self.assertEqual(moe_layer_cfg.top_k, 1)
        self.assertFalse(moe_layer_cfg.sampler_config.normalize_probabilities_flag)
        self.assertEqual(moe_layer_cfg.sampler_config.switch_loss_weight, 0.1)

        cfg = presets.get_config(ExperimentPreset.TOP2_BALANCED_AUX)[0]
        moe_layer_cfg = self._moe_layer_config(cfg)
        self.assertEqual(moe_layer_cfg.top_k, 2)
        self.assertEqual(
            moe_layer_cfg.sampler_config.coefficient_of_variation_loss_weight,
            0.1,
        )

        expected_capacity = {
            ExperimentPreset.CAPACITY_TOP1_ZERO: DroppedTokenOptions.ZEROS,
            ExperimentPreset.CAPACITY_TOP1_IDENTITY: DroppedTokenOptions.IDENTITY,
        }
        for preset, expected_dropped_behavior in expected_capacity.items():
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset)[0]
                moe_layer_cfg = self._moe_layer_config(cfg)

                self.assertEqual(moe_layer_cfg.top_k, 1)
                self.assertEqual(moe_layer_cfg.capacity_factor, 1.0)
                self.assertFalse(
                    moe_layer_cfg.sampler_config.normalize_probabilities_flag
                )
                self.assertEqual(
                    moe_layer_cfg.dropped_token_behavior,
                    expected_dropped_behavior,
                )

        cfg = presets.get_config(ExperimentPreset.NOISY_SHARED_ROUTER)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        moe_layer_cfg = self._moe_layer_config(cfg)
        self.assertEqual(
            moe_model_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self.assertTrue(moe_model_cfg.sampler_config.noisy_topk_flag)
        self.assertTrue(moe_model_cfg.sampler_config.router_config.noisy_topk_flag)
        self.assertTrue(moe_layer_cfg.sampler_config.noisy_topk_flag)
        self.assertTrue(moe_layer_cfg.sampler_config.router_config.noisy_topk_flag)

        cfg = presets.get_config(ExperimentPreset.RESIDUAL_SHARED_ROUTER)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        layer_cfg = moe_model_cfg.stack_config.layer_config
        self.assertEqual(
            moe_model_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self.assertEqual(
            layer_cfg.residual_connection_option, ResidualConnectionOptions.RESIDUAL
        )

        cfg = presets.get_config(ExperimentPreset.POST_NORM_AFTER_WEIGHT)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        layer_cfg = moe_model_cfg.stack_config.layer_config
        moe_layer_cfg = layer_cfg.layer_model_config
        self.assertEqual(
            layer_cfg.layer_norm_position,
            config.LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(
            moe_layer_cfg.weighting_position_option,
            ExpertWeightingPositionOptions.AFTER_EXPERTS,
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

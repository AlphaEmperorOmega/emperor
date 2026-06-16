from emperor.base.layer.residual import ResidualConnectionOptions
import unittest

import torch

import models.linears.linear.config as config

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
from emperor.experiments.base import PresetLock, RandomSearch
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.config import (
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from emperor.memory.options import MemoryPositionOptions
from models.linears.linear.model import Model
from models.linears.linear.config_builder import LinearConfigBuilder
from models.linears.linear.presets import ExperimentOptions, ExperimentPresets
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestLinearModel(unittest.TestCase):
    def test_all_options_forward_one_mnist_batch(self):
        batch_size = 4
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
        batch_size = 4
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

    def test_controller_presets_wire_expected_layer_configs(self):
        presets = ExperimentPresets()
        cases = [
            (ExperimentOptions.GATING, True, False),
            (ExperimentOptions.HALTING, False, True),
            (ExperimentOptions.GATING_HALTING, True, True),
        ]

        for option, expect_gate, expect_halting in cases:
            with self.subTest(option=option.name):
                cfg = presets.get_config(option)[0]
                layer_config = cfg.experiment_config.model_config.layer_config

                self.assertEqual(layer_config.gate_config is not None, expect_gate)
                self.assertEqual(
                    layer_config.halting_config is not None,
                    expect_halting,
                )

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

    def test_gate_config_uses_builder_overrides(self):
        cfg = LinearConfigBuilder(
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
        cfg = LinearConfigBuilder(shared_gate_config=shared_gate_config).build()
        model_cfg = cfg.experiment_config.model_config

        self.assertIs(model_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(model_cfg.layer_config.gate_config)

    def test_shared_gate_config_rejects_enabled_stack_gate(self):
        with self.assertRaises(ValueError):
            LinearConfigBuilder(
                stack_gate_flag=True,
                shared_gate_config=self.shared_gate_config(),
            ).build()

    def test_shared_gate_config_allows_absent_stack_gate(self):
        shared_gate_config = self.shared_gate_config()
        cfg = LinearConfigBuilder(
            stack_gate_flag=False,
            shared_gate_config=shared_gate_config,
        ).build()
        model_cfg = cfg.experiment_config.model_config

        self.assertIs(model_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(model_cfg.layer_config.gate_config)

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
            halting_stack_residual_connection_option=ResidualConnectionOptions.DISABLED,
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
        self.assertEqual(
            halting_stack_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(halting_stack_cfg.layer_config.dropout_probability, 0.3)
        self.assertFalse(halting_stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_controller_stack_defaults_preserve_current_behavior(self):
        cfg = LinearConfigBuilder(
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
            config.LAYER_NORM_POSITION,
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
            config.LAYER_NORM_POSITION,
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
        cfg = LinearConfigBuilder(
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
        cfg = LinearConfigBuilder(
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

    def test_memory_config_uses_builder_defaults(self):
        cfg = LinearConfigBuilder(
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
        cfg = LinearConfigBuilder(
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
        cfg = LinearConfigBuilder(
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
        cfg = LinearConfigBuilder(
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
        cfg = LinearConfigBuilder(
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
        default_cfg = LinearConfigBuilder(recurrent_flag=True).build()
        default_recurrent_cfg = default_cfg.experiment_config.model_config

        self.assertIsInstance(default_recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            default_recurrent_cfg.recurrent_layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )

        cfg = LinearConfigBuilder(
            recurrent_flag=True,
            recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
        ).build()
        recurrent_cfg = cfg.experiment_config.model_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            recurrent_cfg.recurrent_layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )

    def test_recurrent_gate_config_uses_recurrent_overrides(self):
        cfg = LinearConfigBuilder(
            recurrent_flag=True,
            recurrent_gate_flag=True,
            recurrent_gate_option=LayerGateOptions.MULTIPLIER,
            recurrent_gate_hidden_dim=64,
            recurrent_gate_layer_norm_position=LayerNormPositionOptions.AFTER,
            recurrent_gate_stack_num_layers=4,
            recurrent_gate_stack_activation=ActivationOptions.SILU,
            recurrent_gate_stack_residual_connection_option=ResidualConnectionOptions.DISABLED,
            recurrent_gate_stack_dropout_probability=0.15,
            recurrent_gate_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            recurrent_gate_stack_apply_output_pipeline_flag=False,
            recurrent_gate_bias_flag=False,
        ).build()

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
        cfg = LinearConfigBuilder(
            output_dim=13,
            recurrent_flag=True,
            recurrent_halting_flag=True,
            recurrent_halting_threshold=0.65,
            recurrent_halting_dropout=0.25,
            recurrent_halting_hidden_state_mode=HaltingHiddenStateModeOptions.ACCUMULATED,
            recurrent_halting_hidden_dim=72,
            recurrent_halting_output_dim=3,
            recurrent_halting_layer_norm_position=LayerNormPositionOptions.BEFORE,
            recurrent_halting_stack_num_layers=4,
            recurrent_halting_stack_activation=ActivationOptions.MISH,
            recurrent_halting_stack_residual_connection_option=ResidualConnectionOptions.DISABLED,
            recurrent_halting_stack_dropout_probability=0.35,
            recurrent_halting_stack_last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            recurrent_halting_stack_apply_output_pipeline_flag=True,
            recurrent_halting_bias_flag=False,
        ).build()

        recurrent_cfg = cfg.experiment_config.model_config
        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertIsNone(recurrent_cfg.block_config.layer_config.halting_config)

        halting_cfg = recurrent_cfg.halting_config
        halting_stack_cfg = halting_cfg.halting_gate_config
        self.assertEqual(halting_cfg.threshold, 0.65)
        self.assertEqual(halting_cfg.halting_dropout, 0.25)
        self.assertEqual(
            halting_cfg.hidden_state_mode, HaltingHiddenStateModeOptions.ACCUMULATED
        )
        self.assertEqual(halting_stack_cfg.hidden_dim, 72)
        self.assertEqual(halting_stack_cfg.output_dim, 3)
        self.assertEqual(halting_stack_cfg.num_layers, 4)
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
        self.assertEqual(
            halting_stack_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.DISABLED,
        )
        self.assertEqual(halting_stack_cfg.layer_config.dropout_probability, 0.35)
        self.assertFalse(halting_stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_recurrent_controller_stacks_inherit_through_controller_defaults(self):
        cfg = LinearConfigBuilder(
            recurrent_flag=True,
            recurrent_gate_flag=True,
            recurrent_halting_flag=True,
            gate_hidden_dim=31,
            gate_layer_norm_position=LayerNormPositionOptions.AFTER,
            gate_stack_num_layers=3,
            gate_stack_activation=ActivationOptions.SILU,
            gate_stack_residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            gate_stack_dropout_probability=0.11,
            gate_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            gate_stack_apply_output_pipeline_flag=False,
            gate_bias_flag=False,
            halting_hidden_dim=41,
            halting_layer_norm_position=LayerNormPositionOptions.BEFORE,
            halting_stack_num_layers=4,
            halting_stack_activation=ActivationOptions.MISH,
            halting_stack_residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            halting_stack_dropout_probability=0.21,
            halting_stack_last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            halting_stack_apply_output_pipeline_flag=True,
            halting_bias_flag=False,
        ).build()

        recurrent_cfg = cfg.experiment_config.model_config
        gate_cfg = recurrent_cfg.gate_config.model_config
        halting_stack_cfg = recurrent_cfg.halting_config.halting_gate_config

        self.assertEqual(gate_cfg.hidden_dim, 31)
        self.assertEqual(gate_cfg.num_layers, 3)
        self.assertEqual(gate_cfg.last_layer_bias_option, LastLayerBiasOptions.DISABLED)
        self.assertFalse(gate_cfg.apply_output_pipeline_flag)
        self.assertEqual(
            gate_cfg.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(gate_cfg.layer_config.activation, ActivationOptions.SILU)
        self.assertEqual(
            gate_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertEqual(gate_cfg.layer_config.dropout_probability, 0.11)
        self.assertFalse(gate_cfg.layer_config.layer_model_config.bias_flag)

        self.assertEqual(halting_stack_cfg.hidden_dim, 41)
        self.assertEqual(halting_stack_cfg.num_layers, 4)
        self.assertEqual(
            halting_stack_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DEFAULT,
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
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertEqual(halting_stack_cfg.layer_config.dropout_probability, 0.21)
        self.assertFalse(halting_stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_stack_and_recurrent_controller_configs_are_independent(self):
        cfg = LinearConfigBuilder(
            recurrent_flag=True,
            stack_gate_flag=True,
            gate_hidden_dim=32,
            recurrent_gate_flag=True,
            recurrent_gate_hidden_dim=64,
            stack_halting_flag=True,
            halting_threshold=0.55,
            recurrent_halting_flag=True,
            recurrent_halting_threshold=0.75,
        ).build()

        recurrent_cfg = cfg.experiment_config.model_config
        block_cfg = recurrent_cfg.block_config

        self.assertEqual(block_cfg.layer_config.gate_config.model_config.hidden_dim, 32)
        self.assertEqual(recurrent_cfg.gate_config.model_config.hidden_dim, 64)
        self.assertEqual(block_cfg.layer_config.halting_config.threshold, 0.55)
        self.assertEqual(recurrent_cfg.halting_config.threshold, 0.75)

    def test_recurrent_presets_wire_optional_controllers(self):
        expected_controllers = {
            ExperimentOptions.RECURRENT: (False, False),
            ExperimentOptions.RECURRENT_GATING: (True, False),
            ExperimentOptions.RECURRENT_HALTING: (False, True),
            ExperimentOptions.RECURRENT_GATING_HALTING: (True, True),
            ExperimentOptions.RECURRENT_RESIDUAL: (False, False),
            ExperimentOptions.RECURRENT_POST_NORM: (False, False),
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

    def test_new_combination_presets_wire_config(self):
        presets = ExperimentPresets()

        cfg = presets.get_config(ExperimentOptions.RESIDUAL_POST_NORM)[0]
        layer_cfg = cfg.experiment_config.model_config.layer_config
        self.assertEqual(
            layer_cfg.residual_connection_option, ResidualConnectionOptions.RESIDUAL
        )
        self.assertEqual(layer_cfg.layer_norm_position, LayerNormPositionOptions.AFTER)

        cfg = presets.get_config(ExperimentOptions.RESIDUAL_GATING)[0]
        layer_cfg = cfg.experiment_config.model_config.layer_config
        self.assertEqual(
            layer_cfg.residual_connection_option, ResidualConnectionOptions.RESIDUAL
        )
        self.assertIsNotNone(layer_cfg.gate_config)

        cfg = presets.get_config(ExperimentOptions.RESIDUAL_HALTING)[0]
        layer_cfg = cfg.experiment_config.model_config.layer_config
        self.assertEqual(
            layer_cfg.residual_connection_option, ResidualConnectionOptions.RESIDUAL
        )
        self.assertIsNotNone(layer_cfg.halting_config)

        cfg = presets.get_config(ExperimentOptions.RECURRENT_RESIDUAL)[0]
        recurrent_cfg = cfg.experiment_config.model_config
        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            recurrent_cfg.block_config.layer_config.residual_connection_option,
            ResidualConnectionOptions.RESIDUAL,
        )

        cfg = presets.get_config(ExperimentOptions.RECURRENT_POST_NORM)[0]
        recurrent_cfg = cfg.experiment_config.model_config
        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            recurrent_cfg.block_config.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )


if __name__ == "__main__":
    unittest.main()

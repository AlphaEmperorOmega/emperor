import importlib
import runpy
import sys
import unittest
from unittest.mock import patch

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
from emperor.experiments.base import GridSearch, PresetLock, RandomSearch
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.config import (
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from emperor.memory.options import MemoryPositionOptions

import models.linears.linear.config as config
from models.linears.linear.config_builder import LinearConfigBuilder
from models.linears.linear.model import Model
from models.linears.linear.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)
from models.parser import get_experiment_parser, resolve_experiment_mode
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestLinearModel(unittest.TestCase):
    def test_model_preserves_full_model_config(self):
        cfg = LinearConfigBuilder().build()
        model = Model(cfg)

        self.assertIs(model.cfg, cfg)
        self.assertIs(model.model_cfg, cfg)
        self.assertIs(model.exp_cfg, cfg.experiment_config)

    def test_boundary_configs_are_separate_linear_layers(self):
        cfg = LinearConfigBuilder(
            stack_activation=ActivationOptions.MISH,
            stack_gate_flag=True,
            stack_halting_flag=True,
            memory_flag=True,
        ).build()
        exp_cfg = cfg.experiment_config

        self.assertIsNot(exp_cfg.input_model_config, exp_cfg.model_config)
        self.assertIsNot(exp_cfg.output_model_config, exp_cfg.model_config)
        self.assertIsNot(exp_cfg.input_model_config, exp_cfg.output_model_config)

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
                    LinearLayerConfig,
                )
                self.assertTrue(boundary_cfg.layer_model_config.bias_flag)

    def test_public_imports_remain_available(self):
        for module_name in (
            "models.linears.linear.config",
            "models.linears.linear.presets",
            "models.linears.linear.model",
            "models.linears.linear.config_builder",
            "models.linears.linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

    def test_experiment_public_model_id_remains_catalog_id(self):
        self.assertEqual(Experiment()._public_model_id(), "linears/linear")

    def test_module_entrypoint_resolves_cli_without_training(self):
        with (
            patch.object(sys, "argv", ["linear", "--preset", "baseline"]),
            patch(
                "models.linears.linear.presets.Experiment.train_model",
                autospec=True,
            ) as train_model,
        ):
            runpy.run_module("models.linears.linear.__main__", run_name="__main__")

        train_model.assert_called_once()
        experiment = train_model.call_args.args[0]
        kwargs = train_model.call_args.kwargs

        self.assertEqual(experiment.preset, ExperimentPreset.BASELINE)
        self.assertIsNone(kwargs["search_mode"])
        self.assertIsNone(kwargs["log_folder"])
        self.assertIsNone(kwargs["search_keys"])
        self.assertEqual(kwargs["config_overrides"], {})
        self.assertEqual(kwargs["search_overrides"], {})
        self.assertEqual(kwargs["selected_datasets"], config.DATASET_OPTIONS)
        self.assertIsNone(kwargs["selected_presets"])
        self.assertEqual(kwargs["callbacks"], [])

    def test_module_entrypoint_resolves_monitor_callbacks_without_training(self):
        with (
            patch.object(
                sys,
                "argv",
                [
                    "linear",
                    "--preset",
                    "baseline",
                    "--monitors",
                    "linear",
                    "halting",
                    "linear",
                ],
            ),
            patch(
                "models.linears.linear.presets.Experiment.train_model",
                autospec=True,
            ) as train_model,
        ):
            runpy.run_module("models.linears.linear.__main__", run_name="__main__")

        train_model.assert_called_once()
        kwargs = train_model.call_args.kwargs

        self.assertEqual(
            [type(callback) for callback in kwargs["callbacks"]],
            [
                config.LinearMonitorCallback,
                config.HaltingMonitorCallback,
            ],
        )

    def test_monitor_options_expose_callback_factories(self):
        self.assertTrue(config.MONITOR_OPTIONS)

        for option in config.MONITOR_OPTIONS:
            with self.subTest(option=option.name):
                self.assertTrue(option.name)
                self.assertTrue(option.label)
                self.assertTrue(option.kinds)
                self.assertTrue(callable(option.callback_factory))
                self.assertIsNotNone(option.callback_factory())

    def test_cli_legacy_and_stack_alias_flags_remain_available(self):
        parser = get_experiment_parser(
            ExperimentPreset.names(),
            "models.linears.linear",
        )
        cases = (
            ("--bias-flag", "false", "bias_flag", False),
            ("--stack-bias-flag", "false", "bias_flag", False),
            ("--hidden-dim", "64", "hidden_dim", 64),
            ("--stack-hidden-dim", "64", "hidden_dim", 64),
            (
                "--layer-norm-position",
                "AFTER",
                "layer_norm_position",
                LayerNormPositionOptions.AFTER,
            ),
            (
                "--stack-layer-norm-position",
                "AFTER",
                "layer_norm_position",
                LayerNormPositionOptions.AFTER,
            ),
        )

        for flag, value, override_key, expected_value in cases:
            with self.subTest(flag=flag):
                args = parser.parse_args(["--preset", "baseline", flag, value])

                mode = resolve_experiment_mode(
                    args,
                    ExperimentPreset,
                )

                self.assertEqual(mode.config_overrides[override_key], expected_value)
                self.assertEqual(mode.search_overrides, {})

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

    def test_non_baseline_options_accept_search_flags(self):
        presets = ExperimentPresets()
        searchable_options = [
            ExperimentPreset.GATING,
            ExperimentPreset.HALTING,
            ExperimentPreset.MEMORY,
            ExperimentPreset.GATING_HALTING,
        ]

        for preset in searchable_options:
            with self.subTest(preset=preset.name):
                configs = presets.get_config(
                    preset,
                    config.DATASET_OPTIONS[0],
                    RandomSearch(num_samples=2),
                )

                self.assertEqual(len(configs), 2)

    def test_controller_presets_wire_expected_layer_configs(self):
        presets = ExperimentPresets()
        cases = [
            (ExperimentPreset.GATING, True, False, False),
            (ExperimentPreset.HALTING, False, True, False),
            (ExperimentPreset.MEMORY, False, False, True),
            (ExperimentPreset.GATING_HALTING, True, True, False),
            (ExperimentPreset.GATING_MEMORY, True, False, True),
            (ExperimentPreset.HALTING_MEMORY, False, True, True),
            (ExperimentPreset.GATING_HALTING_MEMORY, True, True, True),
        ]

        for preset, expect_gate, expect_halting, expect_memory in cases:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset)[0]
                model_config = cfg.experiment_config.model_config
                layer_config = model_config.layer_config

                self.assertEqual(layer_config.gate_config is not None, expect_gate)
                self.assertEqual(
                    layer_config.halting_config is not None,
                    expect_halting,
                )
                self.assertEqual(
                    model_config.shared_memory_config is not None,
                    expect_memory,
                )

    def test_preset_locks_are_exposed_with_reasons(self):
        presets = ExperimentPresets()

        for preset, expected_locks in presets.PRESET_LOCKS.items():
            with self.subTest(preset=preset.name):
                locks = presets.locked_fields(preset)

                self.assertEqual(set(locks), set(expected_locks))
                for field, lock in locks.items():
                    expected = expected_locks[field]
                    expected_value = (
                        expected.value if isinstance(expected, PresetLock) else expected
                    )
                    self.assertEqual(lock.value, expected_value)
                    self.assertIn(preset.name, lock.reason)

    def test_gate_config_uses_builder_overrides(self):
        cfg = LinearConfigBuilder(
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

    def test_controller_stack_defaults_use_submodule_stack_options(self):
        cfg = LinearConfigBuilder(
            stack_gate_flag=True,
            stack_halting_flag=True,
            memory_flag=True,
        ).build()

        model_cfg = cfg.experiment_config.model_config
        gate_cfg = model_cfg.layer_config.gate_config.model_config
        halting_stack_cfg = model_cfg.layer_config.halting_config.halting_gate_config
        memory_stack_cfg = model_cfg.shared_memory_config.model_config

        self.assertEqual(gate_cfg.hidden_dim, config.STACK_HIDDEN_DIM)
        self.assertEqual(gate_cfg.num_layers, 2)
        self.assertEqual(gate_cfg.layer_config.activation, ActivationOptions.GELU)
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
        self.assertFalse(gate_cfg.apply_output_pipeline_flag)
        self.assertTrue(gate_cfg.layer_config.layer_model_config.bias_flag)

        self.assertEqual(halting_stack_cfg.hidden_dim, config.STACK_HIDDEN_DIM)
        self.assertEqual(halting_stack_cfg.num_layers, 2)
        self.assertEqual(
            halting_stack_cfg.layer_config.activation,
            ActivationOptions.GELU,
        )
        self.assertEqual(
            halting_stack_cfg.layer_config.layer_norm_position,
            config.STACK_LAYER_NORM_POSITION,
        )
        self.assertEqual(
            halting_stack_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertFalse(halting_stack_cfg.apply_output_pipeline_flag)
        self.assertTrue(halting_stack_cfg.layer_config.layer_model_config.bias_flag)

        self.assertEqual(memory_stack_cfg.hidden_dim, config.STACK_HIDDEN_DIM)
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

    def test_stack_defaults_match_submodule_defaults(self):
        self.assertEqual(config.SUBMODULE_STACK_HIDDEN_DIM, config.STACK_HIDDEN_DIM)
        self.assertEqual(
            config.SUBMODULE_STACK_LAYER_NORM_POSITION,
            config.STACK_LAYER_NORM_POSITION,
        )
        self.assertEqual(config.SUBMODULE_STACK_BIAS_FLAG, config.STACK_BIAS_FLAG)

        cfg = LinearConfigBuilder().build()
        model_cfg = cfg.experiment_config.model_config

        self.assertEqual(cfg.hidden_dim, config.STACK_HIDDEN_DIM)
        self.assertEqual(
            model_cfg.layer_config.layer_norm_position,
            config.STACK_LAYER_NORM_POSITION,
        )
        self.assertEqual(
            model_cfg.layer_config.layer_model_config.bias_flag,
            config.STACK_BIAS_FLAG,
        )

    def test_controller_stacks_inherit_submodule_defaults_when_overrides_are_none(
        self,
    ):
        cfg = LinearConfigBuilder(
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
            gate_stack_activation=ActivationOptions.TANH,
            gate_stack_apply_output_pipeline_flag=False,
            gate_stack_bias_flag=True,
            halting_stack_layer_norm_position=LayerNormPositionOptions.DISABLED,
            halting_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
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
                self.assertFalse(stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_controller_stack_overrides_require_independent_flags(self):
        cfg = LinearConfigBuilder(
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
        ).build()

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
        cfg = LinearConfigBuilder(
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
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            RandomSearch(num_samples=20),
            search_keys=["stack_hidden_dim"],
        )

        learning_rates = {cfg.learning_rate for cfg in configs}
        hidden_dims = {cfg.hidden_dim for cfg in configs}

        self.assertEqual(len(learning_rates), 1)
        self.assertEqual(hidden_dims, set(config.SEARCH_SPACE_STACK_HIDDEN_DIM))

    def test_search_keys_unknown_axis_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ExperimentPresets().get_config(
                ExperimentPreset.BASELINE,
                config.DATASET_OPTIONS[0],
                RandomSearch(num_samples=2),
                search_keys=["bogus_axis"],
            )

        self.assertIn("Unknown", str(ctx.exception))

    def test_gating_rejects_locked_config_override(self):
        with self.assertRaisesRegex(ValueError, "GATING.*stack_gate_flag.*True.*False"):
            ExperimentPresets().get_config(
                ExperimentPreset.GATING,
                config.DATASET_OPTIONS[0],
                config_overrides={"stack_gate_flag": False},
            )

    def test_gating_allows_unlocked_config_override(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.GATING,
            config.DATASET_OPTIONS[0],
            config_overrides={"learning_rate": 2e-3},
        )[0]

        self.assertEqual(cfg.learning_rate, 2e-3)
        self.assertIsNotNone(
            cfg.experiment_config.model_config.layer_config.gate_config
        )

    def test_post_norm_rejects_explicit_search_key_for_locked_alias(self):
        with self.assertRaisesRegex(
            ValueError,
            "POST_NORM.*layer_norm_position.*stack_layer_norm_position",
        ):
            ExperimentPresets().get_config(
                ExperimentPreset.POST_NORM,
                config.DATASET_OPTIONS[0],
                GridSearch(),
                search_keys=["stack_layer_norm_position"],
            )

    def test_post_norm_implicit_search_prunes_locked_layer_norm_axis(self):
        original_search_spaces = {
            "SEARCH_SPACE_LEARNING_RATE": config.SEARCH_SPACE_LEARNING_RATE,
            "SEARCH_SPACE_STACK_HIDDEN_DIM": config.SEARCH_SPACE_STACK_HIDDEN_DIM,
            "SEARCH_SPACE_STACK_NUM_LAYERS": config.SEARCH_SPACE_STACK_NUM_LAYERS,
            "SEARCH_SPACE_STACK_DROPOUT_PROBABILITY": (
                config.SEARCH_SPACE_STACK_DROPOUT_PROBABILITY
            ),
            "SEARCH_SPACE_STACK_LAYER_NORM_POSITION": (
                config.SEARCH_SPACE_STACK_LAYER_NORM_POSITION
            ),
            "SEARCH_SPACE_STACK_ACTIVATION": config.SEARCH_SPACE_STACK_ACTIVATION,
        }
        try:
            config.SEARCH_SPACE_LEARNING_RATE = [1e-4, 1e-3]
            config.SEARCH_SPACE_STACK_HIDDEN_DIM = [16, 32]
            config.SEARCH_SPACE_STACK_NUM_LAYERS = [2]
            config.SEARCH_SPACE_STACK_DROPOUT_PROBABILITY = [0.0]
            config.SEARCH_SPACE_STACK_LAYER_NORM_POSITION = [
                LayerNormPositionOptions.DISABLED,
                LayerNormPositionOptions.BEFORE,
            ]
            config.SEARCH_SPACE_STACK_ACTIVATION = [ActivationOptions.RELU]

            configs = ExperimentPresets().get_config(
                ExperimentPreset.POST_NORM,
                config.DATASET_OPTIONS[0],
                GridSearch(),
            )
        finally:
            for key, value in original_search_spaces.items():
                setattr(config, key, value)

        self.assertEqual(len(configs), 4)
        self.assertEqual({cfg.learning_rate for cfg in configs}, {1e-4, 1e-3})
        self.assertEqual({cfg.hidden_dim for cfg in configs}, {16, 32})
        self.assertEqual(
            {
                cfg.experiment_config.model_config.layer_config.layer_norm_position
                for cfg in configs
            },
            {LayerNormPositionOptions.AFTER},
        )

    def test_halting_hidden_dim_falls_back_to_output_dim(self):
        cfg = LinearConfigBuilder(
            output_dim=11,
            stack_halting_flag=True,
            halting_stack_independent_flag=True,
            halting_stack_hidden_dim=0,
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
            recurrent_gate_stack_independent_flag=True,
            recurrent_gate_stack_hidden_dim=64,
            recurrent_gate_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            recurrent_gate_stack_num_layers=4,
            recurrent_gate_stack_activation=ActivationOptions.SILU,
            recurrent_gate_stack_residual_connection_option=ResidualConnectionOptions.DISABLED,
            recurrent_gate_stack_dropout_probability=0.15,
            recurrent_gate_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            recurrent_gate_stack_apply_output_pipeline_flag=False,
            recurrent_gate_stack_bias_flag=False,
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
            recurrent_halting_stack_independent_flag=True,
            recurrent_halting_stack_hidden_dim=72,
            recurrent_halting_stack_layer_norm_position=LayerNormPositionOptions.BEFORE,
            recurrent_halting_stack_num_layers=4,
            recurrent_halting_stack_activation=ActivationOptions.MISH,
            recurrent_halting_stack_residual_connection_option=ResidualConnectionOptions.DISABLED,
            recurrent_halting_stack_dropout_probability=0.35,
            recurrent_halting_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            recurrent_halting_stack_apply_output_pipeline_flag=True,
            recurrent_halting_stack_bias_flag=False,
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
        self.assertEqual(halting_stack_cfg.output_dim, 2)
        self.assertEqual(halting_stack_cfg.num_layers, 4)
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
        self.assertEqual(halting_stack_cfg.layer_config.dropout_probability, 0.35)
        self.assertFalse(halting_stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_recurrent_controller_stacks_inherit_through_controller_defaults(self):
        cfg = LinearConfigBuilder(
            recurrent_flag=True,
            recurrent_gate_flag=True,
            recurrent_halting_flag=True,
            gate_stack_independent_flag=True,
            gate_stack_hidden_dim=31,
            gate_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            gate_stack_num_layers=3,
            gate_stack_activation=ActivationOptions.SILU,
            gate_stack_residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            gate_stack_dropout_probability=0.11,
            gate_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            gate_stack_apply_output_pipeline_flag=False,
            gate_stack_bias_flag=False,
            halting_stack_independent_flag=True,
            halting_stack_hidden_dim=41,
            halting_stack_layer_norm_position=LayerNormPositionOptions.BEFORE,
            halting_stack_num_layers=4,
            halting_stack_activation=ActivationOptions.MISH,
            halting_stack_residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            halting_stack_dropout_probability=0.21,
            halting_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            halting_stack_apply_output_pipeline_flag=True,
            halting_stack_bias_flag=False,
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
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertEqual(halting_stack_cfg.layer_config.dropout_probability, 0.21)
        self.assertFalse(halting_stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_stack_and_recurrent_controller_configs_are_independent(self):
        cfg = LinearConfigBuilder(
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
            recurrent_halting_flag=True,
            recurrent_halting_stack_independent_flag=True,
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

    def test_new_combination_presets_wire_config(self):
        presets = ExperimentPresets()

        cfg = presets.get_config(ExperimentPreset.RESIDUAL_POST_NORM)[0]
        layer_cfg = cfg.experiment_config.model_config.layer_config
        self.assertEqual(
            layer_cfg.residual_connection_option, ResidualConnectionOptions.RESIDUAL
        )
        self.assertEqual(layer_cfg.layer_norm_position, LayerNormPositionOptions.AFTER)

        cfg = presets.get_config(ExperimentPreset.RESIDUAL_GATING)[0]
        layer_cfg = cfg.experiment_config.model_config.layer_config
        self.assertEqual(
            layer_cfg.residual_connection_option, ResidualConnectionOptions.RESIDUAL
        )
        self.assertIsNotNone(layer_cfg.gate_config)

        cfg = presets.get_config(ExperimentPreset.RESIDUAL_HALTING)[0]
        layer_cfg = cfg.experiment_config.model_config.layer_config
        self.assertEqual(
            layer_cfg.residual_connection_option, ResidualConnectionOptions.RESIDUAL
        )
        self.assertIsNotNone(layer_cfg.halting_config)

        cfg = presets.get_config(ExperimentPreset.RESIDUAL_MEMORY)[0]
        model_cfg = cfg.experiment_config.model_config
        self.assertEqual(
            model_cfg.layer_config.residual_connection_option,
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertIsNotNone(model_cfg.shared_memory_config)

        cfg = presets.get_config(ExperimentPreset.RECURRENT_RESIDUAL)[0]
        recurrent_cfg = cfg.experiment_config.model_config
        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            recurrent_cfg.block_config.layer_config.residual_connection_option,
            ResidualConnectionOptions.RESIDUAL,
        )

        cfg = presets.get_config(ExperimentPreset.RECURRENT_POST_NORM)[0]
        recurrent_cfg = cfg.experiment_config.model_config
        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(
            recurrent_cfg.block_config.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )


if __name__ == "__main__":
    unittest.main()

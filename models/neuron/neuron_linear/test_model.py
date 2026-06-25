import importlib
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
from emperor.linears.core.config import LinearLayerConfig
from emperor.linears.core.layers import AdaptiveLinearLayer, LinearLayer
from emperor.experiments.base import GridSearch
from emperor.neuron.core.config import NeuronClusterConfig
from emperor.neuron.core import NeuronClusterOptimizerSyncCallback
from models.parser import get_experiment_parser, resolve_experiment_mode

import models.neuron.neuron_linear.config as config
from models.linears.linear.presets import (
    ExperimentPreset as SourceExperimentPreset,
    ExperimentPresets as SourceExperimentPresets,
)
from models.neuron.neuron_linear._control_config_factory import (
    NeuronControlConfigFactory,
)
from models.neuron.neuron_linear._controller_stack import HiddenBlockAdapter
from models.neuron.neuron_linear.config_builder import NeuronLinearConfigBuilder
from models.neuron.neuron_linear.experiment_config import (
    ExperimentConfig,
    HiddenBlockConfig,
)
from models.neuron.neuron_linear.model import Model
from models.neuron.neuron_linear.presets import ExperimentPreset, ExperimentPresets
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestNeuronLinearModel(unittest.TestCase):
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

    def test_presets_mirror_source_names(self):
        self.assertEqual(ExperimentPreset.names(), SourceExperimentPreset.names())

    def test_public_imports_and_cli_resolution_match_source(self):
        package = importlib.import_module("models.neuron.neuron_linear")
        parser = get_experiment_parser(
            ExperimentPreset.names(),
            "models.neuron.neuron_linear",
        )
        args = parser.parse_args(
            [
                "--preset",
                "recurrent-gating",
                "--grid-search",
                "--search-keys",
                "cluster-max-steps",
            ]
        )
        mode = resolve_experiment_mode(args, ExperimentPreset)

        self.assertIs(package.ExperimentPreset, ExperimentPreset)
        self.assertEqual(package.__all__, ["Experiment", "ExperimentPreset"])
        self.assertEqual(
            ExperimentPreset.cli_names(),
            SourceExperimentPreset.cli_names(),
        )
        self.assertEqual(mode.preset, ExperimentPreset.RECURRENT_GATING)
        self.assertEqual(mode.search_keys, ["cluster_max_steps"])

    def test_helper_modules_are_present_and_importable(self):
        control_module = importlib.import_module(
            "models.neuron.neuron_linear._control_config_factory"
        )
        controller_module = importlib.import_module(
            "models.neuron.neuron_linear._controller_stack"
        )

        self.assertIs(
            control_module.NeuronControlConfigFactory,
            NeuronControlConfigFactory,
        )
        self.assertIs(controller_module.HiddenBlockAdapter, HiddenBlockAdapter)
        self.assertIs(HiddenBlockConfig()._registry_owner(), HiddenBlockAdapter)

    def test_presets_mirror_source_overrides_and_locks(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                source_preset = SourceExperimentPreset[preset.name]
                self.assertEqual(
                    ExperimentPresets.PRESET_OVERRIDES[preset],
                    SourceExperimentPresets.PRESET_OVERRIDES[source_preset],
                )
                self.assertEqual(
                    {
                        key: lock.value
                        for key, lock in ExperimentPresets.PRESET_LOCKS.get(
                            preset,
                            {},
                        ).items()
                    },
                    {
                        key: lock.value
                        for key, lock in SourceExperimentPresets.PRESET_LOCKS.get(
                            source_preset, {}
                        ).items()
                    },
                )
        self.assertEqual(
            set(ExperimentPresets.PRESET_LOCKS),
            {
                ExperimentPreset[source_preset.name]
                for source_preset in SourceExperimentPresets.PRESET_LOCKS
            },
        )

    def test_preset_lock_reasons_are_mirrored_from_source(self):
        for preset, locks in ExperimentPresets.PRESET_LOCKS.items():
            source_preset = SourceExperimentPreset[preset.name]
            with self.subTest(preset=preset.name):
                self.assertEqual(
                    {key: lock.reason for key, lock in locks.items()},
                    {
                        key: lock.reason
                        for key, lock in SourceExperimentPresets.PRESET_LOCKS[
                            source_preset
                        ].items()
                    },
                )

    def test_locked_preset_rejects_conflicting_overrides(self):
        presets = ExperimentPresets()

        with self.assertRaises(ValueError):
            presets.get_config(
                ExperimentPreset.GATING,
                config.DATASET_OPTIONS[0],
                config_overrides={"stack_gate_flag": False},
            )

        with self.assertRaises(ValueError):
            presets.get_config(
                ExperimentPreset.GATING,
                config.DATASET_OPTIONS[0],
                search_overrides={"stack_gate_flag": [False, True]},
            )

    def test_builder_wraps_source_hidden_block_in_neuron_cluster(self):
        cfg = NeuronLinearConfigBuilder().build()
        experiment_cfg = cfg.experiment_config
        cluster_cfg = experiment_cfg.neuron_cluster_config

        self.assertIsNotNone(experiment_cfg.input_model_config)
        self.assertIsNotNone(experiment_cfg.output_model_config)
        self.assertIsInstance(cluster_cfg, NeuronClusterConfig)
        self.assertIsInstance(experiment_cfg, ExperimentConfig)
        self.assertIsNotNone(cluster_cfg.neuron_config.nucleus_config.model_config)
        self.assertIsNotNone(
            cluster_cfg.neuron_config.nucleus_config.model_config.model_config
        )
        self.assertIsInstance(
            cluster_cfg.neuron_config.nucleus_config.model_config,
            HiddenBlockConfig,
        )
        self.assertIsNot(
            cluster_cfg.neuron_config.nucleus_config.model_config.model_config,
            experiment_cfg.input_model_config,
        )
        self.assertIsNot(
            cluster_cfg.neuron_config.nucleus_config.model_config.model_config,
            experiment_cfg.output_model_config,
        )

    def test_control_factory_deep_copies_source_hidden_block(self):
        from models.linears.linear.config_builder import LinearConfigBuilder

        builder = NeuronLinearConfigBuilder()
        source_cfg = LinearConfigBuilder(**builder._source_linear_defaults()).build()
        cluster_cfg = NeuronControlConfigFactory(builder).build(
            source_cfg.experiment_config.model_config,
            source_cfg.hidden_dim,
        )
        hidden_block_cfg = cluster_cfg.neuron_config.nucleus_config.model_config

        self.assertIsInstance(hidden_block_cfg, HiddenBlockConfig)
        self.assertIsNot(
            hidden_block_cfg.model_config,
            source_cfg.experiment_config.model_config,
        )
        self.assertEqual(hidden_block_cfg.input_dim, source_cfg.hidden_dim)
        self.assertEqual(hidden_block_cfg.output_dim, source_cfg.hidden_dim)

    def test_builder_wires_neuron_cluster_options(self):
        cfg = NeuronLinearConfigBuilder(
            cluster_x_axis_total_neurons=6,
            cluster_y_axis_total_neurons=5,
            cluster_z_axis_total_neurons=2,
            cluster_initial_x_axis_total_neurons=2,
            cluster_initial_y_axis_total_neurons=2,
            cluster_initial_z_axis_total_neurons=1,
            cluster_max_steps=3,
            cluster_growth_threshold=99,
            cluster_terminal_top_k=2,
            cluster_halting_flag=True,
            cluster_halting_threshold=0.75,
            cluster_halting_stack_hidden_dim=33,
            cluster_halting_stack_layer_norm_position=LayerNormPositionOptions.BEFORE,
            cluster_halting_stack_bias_flag=False,
        ).build()
        cluster_cfg = cfg.experiment_config.neuron_cluster_config
        terminal_sampler = cluster_cfg.neuron_config.terminal_config.sampler_config
        router_cfg = terminal_sampler.router_config
        halting_stack = cluster_cfg.halting_config.halting_gate_config

        self.assertEqual(cluster_cfg.x_axis_total_neurons, 6)
        self.assertEqual(cluster_cfg.y_axis_total_neurons, 5)
        self.assertEqual(cluster_cfg.z_axis_total_neurons, 2)
        self.assertEqual(cluster_cfg.initial_x_axis_total_neurons, 2)
        self.assertEqual(cluster_cfg.initial_y_axis_total_neurons, 2)
        self.assertEqual(cluster_cfg.initial_z_axis_total_neurons, 1)
        self.assertEqual(cluster_cfg.max_steps, 3)
        self.assertEqual(cluster_cfg.growth_threshold, 99)
        self.assertIsNone(cluster_cfg.entry_sampler_config)
        self.assertIsNone(cluster_cfg.neuron_config.axons_config.memory_config)
        self.assertEqual(terminal_sampler.top_k, 2)
        self.assertEqual(terminal_sampler.num_experts, 18)
        self.assertEqual(router_cfg.input_dim, cfg.hidden_dim)
        self.assertEqual(router_cfg.num_experts, 18)
        self.assertEqual(router_cfg.model_config.input_dim, cfg.hidden_dim)
        self.assertEqual(router_cfg.model_config.output_dim, 18)
        self.assertIsNotNone(cluster_cfg.halting_config)
        self.assertEqual(cluster_cfg.halting_config.threshold, 0.75)
        self.assertEqual(halting_stack.hidden_dim, 33)
        self.assertEqual(
            halting_stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertFalse(halting_stack.layer_config.layer_model_config.bias_flag)

    def test_cluster_terminal_top_k_is_clamped_to_expert_count(self):
        cfg = NeuronLinearConfigBuilder(
            cluster_terminal_top_k=999,
            cluster_terminal_sampler_num_topk_samples=999,
        ).build()
        sampler_cfg = self._terminal_sampler_config(cfg)

        self.assertEqual(sampler_cfg.num_experts, 18)
        self.assertEqual(sampler_cfg.top_k, 18)
        self.assertEqual(sampler_cfg.num_topk_samples, 18)

    def test_cluster_terminal_top_k_is_clamped_to_at_least_one(self):
        cfg = NeuronLinearConfigBuilder(
            cluster_terminal_top_k=0,
            cluster_terminal_sampler_num_topk_samples=2,
        ).build()
        sampler_cfg = self._terminal_sampler_config(cfg)

        self.assertEqual(sampler_cfg.top_k, 1)
        self.assertEqual(sampler_cfg.num_topk_samples, 1)

    def test_cluster_halting_can_be_disabled(self):
        cfg = NeuronLinearConfigBuilder(cluster_halting_flag=False).build()

        self.assertIsNone(cfg.experiment_config.neuron_cluster_config.halting_config)

    def test_cluster_halting_builder_kwargs_are_canonical(self):
        parameters = inspect.signature(NeuronLinearConfigBuilder.__init__).parameters
        expected_names = {
            "cluster_halting_stack_hidden_dim",
            "cluster_halting_stack_layer_norm_position",
            "cluster_halting_stack_bias_flag",
        }
        legacy_names = {name.replace("_stack_", "_") for name in expected_names}

        for name in expected_names:
            with self.subTest(name=name):
                self.assertIn(name, parameters)

        for name in legacy_names:
            with self.subTest(name=name):
                self.assertNotIn(name, parameters)

        legacy_cluster_hidden_dim = "cluster_halting" + "_hidden_dim"
        with self.assertRaises(TypeError):
            NeuronLinearConfigBuilder(**{legacy_cluster_hidden_dim: 32}).build()

    def test_legacy_source_controller_stack_kwargs_are_normalized(self):
        cfg = NeuronLinearConfigBuilder(
            stack_gate_flag=True,
            gate_stack_independent_flag=True,
            gate_hidden_dim=32,
            gate_layer_norm_position=LayerNormPositionOptions.AFTER,
            gate_bias_flag=False,
            stack_halting_flag=True,
            halting_stack_independent_flag=True,
            halting_hidden_dim=48,
            halting_layer_norm_position=LayerNormPositionOptions.BEFORE,
            halting_bias_flag=False,
        ).build()
        hidden_block_cfg = (
            cfg.experiment_config.neuron_cluster_config.neuron_config.nucleus_config.model_config
        )
        stack_cfg = hidden_block_cfg.model_config
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

    def test_builder_uses_neuron_linear_source_defaults(self):
        cfg = NeuronLinearConfigBuilder().build()
        override_cfg = NeuronLinearConfigBuilder(hidden_dim=128).build()

        self.assertEqual(cfg.hidden_dim, config.HIDDEN_DIM)
        self.assertEqual(override_cfg.hidden_dim, 128)

    def test_shared_gate_config_flows_to_source_hidden_block(self):
        shared_gate_config = self.shared_gate_config()
        cfg = NeuronLinearConfigBuilder(shared_gate_config=shared_gate_config).build()
        hidden_block_cfg = (
            cfg.experiment_config.neuron_cluster_config.neuron_config.nucleus_config.model_config
        )
        stack_cfg = hidden_block_cfg.model_config

        self.assertIsInstance(stack_cfg.shared_gate_config, GateConfig)
        self.assertIsNone(stack_cfg.layer_config.gate_config)

    def test_gate_options_flow_to_source_hidden_block(self):
        cfg = NeuronLinearConfigBuilder(
            recurrent_flag=True,
            stack_gate_flag=True,
            recurrent_gate_flag=True,
            gate_option=LayerGateOptions.MULTIPLIER,
            recurrent_gate_option=LayerGateOptions.MULTIPLIER,
        ).build()
        hidden_block_cfg = (
            cfg.experiment_config.neuron_cluster_config.neuron_config.nucleus_config.model_config
        )
        recurrent_cfg = hidden_block_cfg.model_config

        self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
        self.assertEqual(recurrent_cfg.gate_config.option, LayerGateOptions.MULTIPLIER)
        self.assertEqual(
            recurrent_cfg.block_config.layer_config.gate_config.option,
            LayerGateOptions.MULTIPLIER,
        )

    def test_shared_gate_config_rejects_enabled_source_stack_gate(self):
        with self.assertRaises(ValueError):
            NeuronLinearConfigBuilder(
                stack_gate_flag=True,
                shared_gate_config=self.shared_gate_config(),
            ).build()

    def test_shared_gate_config_allows_absent_source_stack_gate(self):
        shared_gate_config = self.shared_gate_config()
        cfg = NeuronLinearConfigBuilder(
            stack_gate_flag=False,
            shared_gate_config=shared_gate_config,
        ).build()
        hidden_block_cfg = (
            cfg.experiment_config.neuron_cluster_config.neuron_config.nucleus_config.model_config
        )
        stack_cfg = hidden_block_cfg.model_config

        self.assertEqual(stack_cfg.shared_gate_config, shared_gate_config)
        self.assertIsNone(stack_cfg.layer_config.gate_config)

    def test_hidden_block_adapter_returns_hidden_and_overrides_dimensions(self):
        hidden_cfg = HiddenBlockConfig(
            input_dim=5,
            output_dim=7,
            model_config=LayerConfig(
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=True),
            ),
        )
        adapter = HiddenBlockAdapter(hidden_cfg)
        X = torch.randn(3, 5)

        output = adapter(X)

        self.assertEqual(output.shape, (3, 7))
        self.assertEqual(adapter.model.input_dim, 5)
        self.assertEqual(adapter.model.output_dim, 7)

    def test_runtime_uses_only_standard_emperor_linear_layers(self):
        cfg = NeuronLinearConfigBuilder().build()
        model = Model(cfg)

        linears = [
            module
            for module in model.modules()
            if type(module).__module__ == "emperor.linears.core.layers"
        ]

        self.assertGreater(len(linears), 0)
        self.assertTrue(all(type(module) is LinearLayer for module in linears))
        self.assertFalse(
            any(isinstance(module, AdaptiveLinearLayer) for module in model.modules())
        )
        self.assertFalse(
            any(
                type(module).__module__.startswith("emperor.experts")
                for module in model.modules()
            )
        )

    def test_config_auto_attaches_neuron_cluster_optimizer_sync_callback(self):
        self.assertIsInstance(
            config.CALLBACK_NEURON_CLUSTER_OPTIMIZER_SYNC,
            NeuronClusterOptimizerSyncCallback,
        )

    def test_all_presets_forward_one_mnist_batch(self):
        batch_size = 2
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                logits, auxiliary_loss = model(X)

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
                self.assertTrue(torch.isfinite(auxiliary_loss.detach()).item())

    def test_baseline_forwards_all_datasets(self):
        batch_size = 2
        presets = ExperimentPresets()

        for dataset in config.DATASET_OPTIONS:
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(ExperimentPreset.BASELINE, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                logits, auxiliary_loss = model(X)

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
                self.assertTrue(torch.isfinite(auxiliary_loss.detach()).item())

    def test_search_applies_neuron_cluster_axes(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            GridSearch(),
            search_keys=["cluster_max_steps", "cluster_terminal_top_k"],
        )

        self.assertEqual(
            len(configs),
            len(config.SEARCH_SPACE_CLUSTER_MAX_STEPS)
            * len(config.SEARCH_SPACE_CLUSTER_TERMINAL_TOP_K),
        )
        observed = {
            (
                cfg.experiment_config.neuron_cluster_config.max_steps,
                cfg.experiment_config.neuron_cluster_config.neuron_config.terminal_config.sampler_config.top_k,
            )
            for cfg in configs
        }
        self.assertEqual(
            observed,
            {
                (max_steps, top_k)
                for max_steps in config.SEARCH_SPACE_CLUSTER_MAX_STEPS
                for top_k in config.SEARCH_SPACE_CLUSTER_TERMINAL_TOP_K
            },
        )

    def test_search_keys_unknown_axis_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ExperimentPresets().get_config(
                ExperimentPreset.BASELINE,
                config.DATASET_OPTIONS[0],
                GridSearch(),
                search_keys=["bogus_axis"],
            )

        self.assertIn("Unknown", str(ctx.exception))

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

    def _terminal_sampler_config(self, cfg):
        cluster_cfg = cfg.experiment_config.neuron_cluster_config
        return cluster_cfg.neuron_config.terminal_config.sampler_config


if __name__ == "__main__":
    unittest.main()

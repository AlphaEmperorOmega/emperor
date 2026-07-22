# ruff: noqa: E501

import importlib
import unittest

import torch

import models.neuron.expert_linear.config as config
import models.neuron.expert_linear.dataset_options as dataset_options
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.neuron import NeuronClusterConfig, NeuronClusterOptimizerSyncCallback
from models.catalog import model_package
from models.cli_selection import resolve_cli_selection
from models.experiment_cli_parser import get_experiment_parser
from models.neuron.expert_linear._hidden_block import (
    HiddenBlockAdapter,
    HiddenBlockConfig,
)
from models.neuron.expert_linear.config_builder import NeuronExpertLinearConfigBuilder
from models.neuron.expert_linear.experiment_config import ExperimentConfig
from models.neuron.expert_linear.model import Model
from models.neuron.expert_linear.presets import (
    _PRESET_DEFINITIONS,
    Experiment,
    ExperimentPreset,
)
from models.neuron.expert_linear.runtime_options import NeuronClusterCapacityOptions


def _build_config(**runtime_defaults):
    runtime = model_package("neuron/expert_linear").bind_runtime_defaults(
        runtime_defaults
    )
    return NeuronExpertLinearConfigBuilder(runtime=runtime).build()


class TestNeuronExpertLinearModel(unittest.TestCase):
    package_module = "models.neuron.expert_linear"

    def test_public_imports_and_cli_resolution(self):
        package = importlib.import_module(self.package_module)
        registration = model_package("neuron/expert_linear")
        parser = get_experiment_parser(registration)
        args = parser.parse_args(
            [
                "--preset",
                ExperimentPreset.cli_names()[0],
                "--grid-search",
                "--search-keys",
                "cluster-max-steps",
            ]
        )
        mode = resolve_cli_selection(args, registration, ExperimentPreset)

        self.assertIs(package.MODEL_PACKAGE, registration)
        self.assertEqual(package.__all__, ["MODEL_PACKAGE"])
        self.assertIs(mode.preset, ExperimentPreset.BASELINE)
        self.assertEqual(mode.search_keys, ["cluster_max_steps"])
        experiment = Experiment(model_package=registration)
        self.assertEqual(
            experiment.model_package.identity.catalog_key,
            "neuron/expert_linear",
        )

    def test_local_preset_snapshot_is_complete(self):
        presets = list(ExperimentPreset)

        self.assertEqual(len(presets), 32)
        self.assertEqual(presets[0], ExperimentPreset.BASELINE)
        self.assertEqual(presets[-1], ExperimentPreset.RECURRENT_POST_NORM)
        self.assertEqual([preset.value for preset in presets], list(range(1, 33)))
        self.assertEqual(set(_PRESET_DEFINITIONS), set(presets))
        for preset in presets:
            with self.subTest(preset=preset.name):
                definition = _PRESET_DEFINITIONS[preset]
                self.assertIsInstance(definition.preset_values, dict)
                self.assertTrue(definition.description.strip())

    def test_all_presets_build_from_local_definitions(self):
        presets = model_package("neuron/expert_linear").presets
        dataset = self._datasets()[0]

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset, dataset)[0]

                self.assertIsInstance(cfg.experiment_config, ExperimentConfig)
                self.assertIsInstance(
                    cfg.experiment_config.neuron_cluster_config,
                    NeuronClusterConfig,
                )

    def test_builder_wraps_package_local_hidden_block(self):
        cfg = NeuronExpertLinearConfigBuilder().build()
        experiment_config = cfg.experiment_config
        cluster_config = experiment_config.neuron_cluster_config
        hidden_block = cluster_config.neuron_config.nucleus_config.model_config

        self.assertIsInstance(experiment_config, ExperimentConfig)
        self.assertIsInstance(cluster_config, NeuronClusterConfig)
        self.assertIsNotNone(experiment_config.input_model_config)
        self.assertIsNotNone(experiment_config.output_model_config)
        self.assertIsInstance(hidden_block, HiddenBlockConfig)
        self.assertIsNot(
            hidden_block.model_config, experiment_config.input_model_config
        )
        self.assertIsNot(
            hidden_block.model_config, experiment_config.output_model_config
        )

    def test_hidden_flat_overrides_build_locally(self):
        cfg = _build_config(
            batch_size=3,
            learning_rate=0.02,
            input_dim=8,
            hidden_dim=12,
            output_dim=4,
            stack_num_layers=1,
            stack_activation=ActivationOptions.MISH,
        )

        self.assertEqual(cfg.batch_size, 3)
        self.assertEqual(cfg.learning_rate, 0.02)
        self.assertEqual(cfg.input_dim, 8)
        self.assertEqual(cfg.hidden_dim, 12)
        self.assertEqual(cfg.output_dim, 4)

    def test_builder_wires_neuron_cluster_options(self):
        cfg = _build_config(
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
        )
        cluster_config = cfg.experiment_config.neuron_cluster_config
        terminal_sampler = cluster_config.neuron_config.terminal_config.sampler_config
        router_config = terminal_sampler.router_config
        halting_stack = cluster_config.halting_config.halting_gate_config

        self.assertEqual(cluster_config.x_axis_total_neurons, 6)
        self.assertEqual(cluster_config.y_axis_total_neurons, 5)
        self.assertEqual(cluster_config.z_axis_total_neurons, 2)
        self.assertEqual(cluster_config.initial_x_axis_total_neurons, 2)
        self.assertEqual(cluster_config.initial_y_axis_total_neurons, 2)
        self.assertEqual(cluster_config.initial_z_axis_total_neurons, 1)
        self.assertEqual(cluster_config.max_steps, 3)
        self.assertEqual(cluster_config.growth_threshold, 99)
        self.assertEqual(terminal_sampler.top_k, 2)
        self.assertEqual(terminal_sampler.num_experts, 18)
        self.assertEqual(router_config.input_dim, cfg.hidden_dim)
        self.assertEqual(router_config.num_experts, 18)
        self.assertEqual(halting_stack.hidden_dim, 33)
        self.assertEqual(
            halting_stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertFalse(halting_stack.layer_config.layer_model_config.bias_flag)

    def test_flat_cluster_capacity_binds_to_typed_runtime_options(self):
        capacity = NeuronClusterCapacityOptions(
            x_axis_total_neurons=6,
            y_axis_total_neurons=5,
            z_axis_total_neurons=2,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=2,
            initial_z_axis_total_neurons=1,
            max_steps=3,
            growth_threshold=99,
        )
        runtime = model_package("neuron/expert_linear").bind_runtime_defaults(
            {
                "hidden_dim": 12,
                "cluster_x_axis_total_neurons": 6,
                "cluster_y_axis_total_neurons": 5,
                "cluster_z_axis_total_neurons": 2,
                "cluster_initial_x_axis_total_neurons": 2,
                "cluster_initial_y_axis_total_neurons": 2,
                "cluster_initial_z_axis_total_neurons": 1,
                "cluster_max_steps": 3,
                "cluster_growth_threshold": 99,
            }
        )

        self.assertEqual(
            runtime._as_construction_kwargs()["cluster_capacity_options"],
            capacity,
        )
        NeuronExpertLinearConfigBuilder(runtime=runtime).build()

    def test_cluster_terminal_top_k_is_clamped(self):
        maximum = _build_config(
            cluster_terminal_top_k=999,
            cluster_terminal_sampler_num_topk_samples=999,
        )
        maximum_sampler = self._terminal_sampler(maximum)

        self.assertEqual(maximum_sampler.num_experts, 18)
        self.assertEqual(maximum_sampler.top_k, 18)
        self.assertEqual(maximum_sampler.num_topk_samples, 18)

        minimum = _build_config(
            cluster_terminal_top_k=0,
            cluster_terminal_sampler_num_topk_samples=2,
        )
        minimum_sampler = self._terminal_sampler(minimum)

        self.assertEqual(minimum_sampler.top_k, 1)
        self.assertEqual(minimum_sampler.num_topk_samples, 1)

    def test_cluster_halting_can_be_disabled(self):
        cfg = _build_config(cluster_halting_flag=False)

        self.assertIsNone(cfg.experiment_config.neuron_cluster_config.halting_config)

    def test_hidden_block_adapter_is_package_local(self):
        hidden_config = HiddenBlockConfig(
            input_dim=5,
            output_dim=7,
            model_config=LayerConfig(
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=True),
            ),
        )
        adapter = HiddenBlockAdapter(hidden_config)
        output = adapter(torch.randn(3, 5))

        self.assertEqual(output.shape, (3, 7))
        self.assertEqual(adapter.model.input_dim, 5)
        self.assertEqual(adapter.model.output_dim, 7)

    def test_config_attaches_neuron_optimizer_sync_callback(self):
        self.assertIsInstance(
            config.CALLBACK_NEURON_CLUSTER_OPTIMIZER_SYNC,
            NeuronClusterOptimizerSyncCallback,
        )

    def test_baseline_forwards_every_dataset(self):
        presets = model_package("neuron/expert_linear").presets
        for dataset in self._datasets():
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(ExperimentPreset.BASELINE, dataset)[0]
                model = Model(cfg)
                model.eval()
                with torch.no_grad():
                    logits, auxiliary_loss = model(self._fake_batch(dataset, 2))

                self.assertEqual(logits.shape, (2, dataset.num_classes))
                self.assertTrue(torch.isfinite(auxiliary_loss.detach()).item())

    def _datasets(self) -> list[type]:
        return list(
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ]
        )

    def _terminal_sampler(self, cfg):
        return cfg.experiment_config.neuron_cluster_config.neuron_config.terminal_config.sampler_config

    @staticmethod
    def _fake_batch(dataset, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )


if __name__ == "__main__":
    unittest.main()

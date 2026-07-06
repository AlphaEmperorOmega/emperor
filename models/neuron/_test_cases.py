import importlib

import torch
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.neuron.core import NeuronClusterOptimizerSyncCallback
from emperor.neuron.core.config import NeuronClusterConfig
from emperor.neuron.core.options import TerminalRangeOptions, TerminalZAxisOffsetOptions
from models.parser import get_experiment_parser, resolve_experiment_mode

from models.neuron._builder_options import (
    ClusterRouteHaltingOptions,
    NeuronClusterCapacityOptions,
    NeuronSubmoduleStackOptions,
    NeuronTerminalOptions,
    NeuronTerminalSamplerOptions,
)
from models.neuron._controller_stack import HiddenBlockAdapter
from models.neuron.experiment_config import ExperimentConfig, HiddenBlockConfig


class NeuronPackageTestMixin:
    package_module: str
    config_module = None
    builder_type = None
    model_type = None
    experiment_preset_type = None
    experiment_presets_type = None
    source_experiment_preset_type = None
    source_experiment_presets_type = None
    source_adapter = None
    expected_builder_type_name: str

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
        self.assertEqual(
            self.experiment_preset_type.names(),
            self.source_experiment_preset_type.names(),
        )
        self.assertEqual(
            self.experiment_preset_type.cli_names(),
            self.source_experiment_preset_type.cli_names(),
        )

    def test_public_imports_and_cli_resolution_match_source(self):
        package = importlib.import_module(self.package_module)
        parser = get_experiment_parser(
            self.experiment_preset_type.names(),
            self.package_module,
        )
        args = parser.parse_args(
            [
                "--preset",
                self.experiment_preset_type.cli_names()[0],
                "--grid-search",
                "--search-keys",
                "cluster-max-steps",
            ]
        )
        mode = resolve_experiment_mode(args, self.experiment_preset_type)

        self.assertIs(package.ExperimentPreset, self.experiment_preset_type)
        self.assertEqual(package.__all__, ["Experiment", "ExperimentPreset"])
        self.assertIs(mode.preset, self.experiment_preset_type.BASELINE)
        self.assertEqual(mode.search_keys, ["cluster_max_steps"])

    def test_presets_mirror_source_overrides_and_locks(self):
        presets = self.experiment_presets_type()
        source_presets = self.source_experiment_presets_type()
        for preset in self.experiment_preset_type:
            with self.subTest(preset=preset.name):
                source_preset = self.source_experiment_preset_type[preset.name]
                self.assertEqual(
                    presets.overrides_for_preset(preset),
                    source_presets.overrides_for_preset(source_preset),
                )
                self.assertEqual(
                    {
                        key: lock.value
                        for key, lock in presets.locks_for_preset(preset).items()
                    },
                    {
                        key: lock.value
                        for key, lock in source_presets.locks_for_preset(
                            source_preset
                        ).items()
                    },
                )

    def test_builder_wraps_source_hidden_block_in_neuron_cluster(self):
        cfg = self.builder_type().build()
        experiment_cfg = cfg.experiment_config
        cluster_cfg = experiment_cfg.neuron_cluster_config

        self.assertIsInstance(experiment_cfg, ExperimentConfig)
        self.assertIsInstance(cluster_cfg, NeuronClusterConfig)
        self.assertIsNotNone(experiment_cfg.input_model_config)
        self.assertIsNotNone(experiment_cfg.output_model_config)
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

    def test_builder_wires_neuron_cluster_options(self):
        cfg = self.builder_type(
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

    def test_option_group_build_matches_flat_kwargs(self):
        capacity_options = NeuronClusterCapacityOptions(
            x_axis_total_neurons=6,
            y_axis_total_neurons=5,
            z_axis_total_neurons=2,
            initial_x_axis_total_neurons=2,
            initial_y_axis_total_neurons=2,
            initial_z_axis_total_neurons=1,
            max_steps=3,
            growth_threshold=99,
        )
        terminal_options = NeuronTerminalOptions(
            xy_axis_range=TerminalRangeOptions.ONE,
            z_axis_range=TerminalRangeOptions.ONE,
            z_axis_offset=TerminalZAxisOffsetOptions.ZERO,
            top_k=2,
        )
        terminal_router_options = NeuronSubmoduleStackOptions(
            hidden_dim=18,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            activation=ActivationOptions.GELU,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.05,
            bias_flag=False,
        )
        terminal_sampler_options = NeuronTerminalSamplerOptions(
            threshold=0.17,
            filter_above_threshold=True,
            num_topk_samples=2,
            normalize_probabilities_flag=True,
            noisy_topk_flag=True,
            coefficient_of_variation_loss_weight=0.11,
            switch_loss_weight=0.12,
            zero_centred_loss_weight=0.13,
            mutual_information_loss_weight=0.14,
        )
        halting_stack_options = NeuronSubmoduleStackOptions(
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
        cluster_halting_options = ClusterRouteHaltingOptions(
            enabled=True,
            threshold=0.63,
            dropout=0.08,
            hidden_state_mode=HaltingHiddenStateModeOptions.ACCUMULATED,
            stack_options=halting_stack_options,
            output_dim=2,
        )
        source_kwargs = {
            "batch_size": 3,
            "learning_rate": 0.02,
            "input_dim": 8,
            "output_dim": 4,
            "stack_hidden_dim": 12,
            "stack_num_layers": 1,
            "stack_activation": ActivationOptions.MISH,
        }
        flat_kwargs = {
            **source_kwargs,
            "cluster_x_axis_total_neurons": (
                capacity_options.x_axis_total_neurons
            ),
            "cluster_y_axis_total_neurons": (
                capacity_options.y_axis_total_neurons
            ),
            "cluster_z_axis_total_neurons": (
                capacity_options.z_axis_total_neurons
            ),
            "cluster_initial_x_axis_total_neurons": (
                capacity_options.initial_x_axis_total_neurons
            ),
            "cluster_initial_y_axis_total_neurons": (
                capacity_options.initial_y_axis_total_neurons
            ),
            "cluster_initial_z_axis_total_neurons": (
                capacity_options.initial_z_axis_total_neurons
            ),
            "cluster_max_steps": capacity_options.max_steps,
            "cluster_growth_threshold": capacity_options.growth_threshold,
            "cluster_terminal_xy_axis_range": terminal_options.xy_axis_range,
            "cluster_terminal_z_axis_range": terminal_options.z_axis_range,
            "cluster_terminal_z_axis_offset": terminal_options.z_axis_offset,
            "cluster_terminal_top_k": terminal_options.top_k,
            "cluster_terminal_router_hidden_dim": (
                terminal_router_options.hidden_dim
            ),
            "cluster_terminal_router_num_layers": (
                terminal_router_options.num_layers
            ),
            "cluster_terminal_router_last_layer_bias_option": (
                terminal_router_options.last_layer_bias_option
            ),
            "cluster_terminal_router_apply_output_pipeline_flag": (
                terminal_router_options.apply_output_pipeline_flag
            ),
            "cluster_terminal_router_activation": (
                terminal_router_options.activation
            ),
            "cluster_terminal_router_layer_norm_position": (
                terminal_router_options.layer_norm_position
            ),
            "cluster_terminal_router_residual_connection_option": (
                terminal_router_options.residual_connection_option
            ),
            "cluster_terminal_router_dropout_probability": (
                terminal_router_options.dropout_probability
            ),
            "cluster_terminal_router_bias_flag": (
                terminal_router_options.bias_flag
            ),
            "cluster_terminal_sampler_threshold": (
                terminal_sampler_options.threshold
            ),
            "cluster_terminal_sampler_filter_above_threshold": (
                terminal_sampler_options.filter_above_threshold
            ),
            "cluster_terminal_sampler_num_topk_samples": (
                terminal_sampler_options.num_topk_samples
            ),
            "cluster_terminal_sampler_normalize_probabilities_flag": (
                terminal_sampler_options.normalize_probabilities_flag
            ),
            "cluster_terminal_sampler_noisy_topk_flag": (
                terminal_sampler_options.noisy_topk_flag
            ),
            "cluster_terminal_sampler_coefficient_of_variation_loss_weight": (
                terminal_sampler_options.coefficient_of_variation_loss_weight
            ),
            "cluster_terminal_sampler_switch_loss_weight": (
                terminal_sampler_options.switch_loss_weight
            ),
            "cluster_terminal_sampler_zero_centred_loss_weight": (
                terminal_sampler_options.zero_centred_loss_weight
            ),
            "cluster_terminal_sampler_mutual_information_loss_weight": (
                terminal_sampler_options.mutual_information_loss_weight
            ),
            "cluster_halting_flag": cluster_halting_options.enabled,
            "cluster_halting_threshold": cluster_halting_options.threshold,
            "cluster_halting_dropout": cluster_halting_options.dropout,
            "cluster_halting_hidden_state_mode": (
                cluster_halting_options.hidden_state_mode
            ),
            "cluster_halting_stack_hidden_dim": (
                halting_stack_options.hidden_dim
            ),
            "cluster_halting_output_dim": cluster_halting_options.output_dim,
            "cluster_halting_stack_layer_norm_position": (
                halting_stack_options.layer_norm_position
            ),
            "cluster_halting_stack_num_layers": halting_stack_options.num_layers,
            "cluster_halting_stack_activation": halting_stack_options.activation,
            "cluster_halting_stack_residual_connection_option": (
                halting_stack_options.residual_connection_option
            ),
            "cluster_halting_stack_dropout_probability": (
                halting_stack_options.dropout_probability
            ),
            "cluster_halting_stack_last_layer_bias_option": (
                halting_stack_options.last_layer_bias_option
            ),
            "cluster_halting_stack_apply_output_pipeline_flag": (
                halting_stack_options.apply_output_pipeline_flag
            ),
            "cluster_halting_stack_bias_flag": halting_stack_options.bias_flag,
        }

        flat_cfg = self.builder_type(**flat_kwargs).build()
        grouped_cfg = self.builder_type(
            **source_kwargs,
            cluster_capacity_options=capacity_options,
            terminal_options=terminal_options,
            terminal_router_options=terminal_router_options,
            terminal_sampler_options=terminal_sampler_options,
            cluster_halting_options=cluster_halting_options,
        ).build()

        self.assertEqual(flat_cfg, grouped_cfg)

    def test_cluster_terminal_top_k_is_clamped(self):
        cfg = self.builder_type(
            cluster_terminal_top_k=999,
            cluster_terminal_sampler_num_topk_samples=999,
        ).build()
        sampler_cfg = self._terminal_sampler_config(cfg)

        self.assertEqual(sampler_cfg.num_experts, 18)
        self.assertEqual(sampler_cfg.top_k, 18)
        self.assertEqual(sampler_cfg.num_topk_samples, 18)

        cfg = self.builder_type(
            cluster_terminal_top_k=0,
            cluster_terminal_sampler_num_topk_samples=2,
        ).build()
        sampler_cfg = self._terminal_sampler_config(cfg)

        self.assertEqual(sampler_cfg.top_k, 1)
        self.assertEqual(sampler_cfg.num_topk_samples, 1)

    def test_cluster_halting_can_be_disabled(self):
        cfg = self.builder_type(cluster_halting_flag=False).build()

        self.assertIsNone(cfg.experiment_config.neuron_cluster_config.halting_config)

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

    def test_config_auto_attaches_neuron_cluster_optimizer_sync_callback(self):
        self.assertIsInstance(
            self.config_module.CALLBACK_NEURON_CLUSTER_OPTIMIZER_SYNC,
            NeuronClusterOptimizerSyncCallback,
        )

    def test_baseline_forwards_one_batch(self):
        batch_size = 2
        dataset = self.config_module.DATASET_OPTIONS[0]
        cfg = self.experiment_presets_type().get_config(
            self.experiment_preset_type.BASELINE,
            dataset,
        )[0]
        model = self.model_type(cfg)
        X = self._fake_batch(dataset, batch_size)

        logits, auxiliary_loss = model(X)

        self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
        self.assertTrue(torch.isfinite(auxiliary_loss.detach()).item())

    def _terminal_sampler_config(self, cfg):
        return (
            cfg.experiment_config.neuron_cluster_config.neuron_config.terminal_config.sampler_config
        )

    def _fake_batch(self, dataset, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )

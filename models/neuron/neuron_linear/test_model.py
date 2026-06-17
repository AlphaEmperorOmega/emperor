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
from emperor.neuron.core.config import NeuronClusterConfig
from emperor.neuron.core import NeuronClusterOptimizerSyncCallback

import models.neuron.neuron_linear.config as config
from models.linears.linear.presets import ExperimentOptions as SourceExperimentOptions
from models.neuron.neuron_linear.config_builder import NeuronLinearConfigBuilder
from models.neuron.neuron_linear.model import Model
from models.neuron.neuron_linear.presets import ExperimentOptions, ExperimentPresets
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
        self.assertEqual(ExperimentOptions.names(), SourceExperimentOptions.names())

    def test_builder_wraps_source_hidden_block_in_neuron_cluster(self):
        cfg = NeuronLinearConfigBuilder().build()
        experiment_cfg = cfg.experiment_config
        cluster_cfg = experiment_cfg.neuron_cluster_config

        self.assertIsNotNone(experiment_cfg.input_model_config)
        self.assertIsNotNone(experiment_cfg.output_model_config)
        self.assertIsInstance(cluster_cfg, NeuronClusterConfig)
        self.assertIsNotNone(cluster_cfg.neuron_config.nucleus_config.model_config)
        self.assertIsNotNone(
            cluster_cfg.neuron_config.nucleus_config.model_config.model_config
        )

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

    def test_all_options_forward_one_mnist_batch(self):
        batch_size = 2
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for option in ExperimentOptions:
            with self.subTest(option=option.name):
                cfg = presets.get_config(option, dataset)[0]
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
                cfg = presets.get_config(ExperimentOptions.BASELINE, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                logits, auxiliary_loss = model(X)

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
                self.assertTrue(torch.isfinite(auxiliary_loss.detach()).item())

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


if __name__ == "__main__":
    unittest.main()

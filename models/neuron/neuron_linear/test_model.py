import unittest

import torch
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

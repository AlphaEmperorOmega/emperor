import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import torch

import models.parametric.parametric_matrix.config as config

from emperor.experiments.base import RandomSearch
from emperor.parametric import (
    AdaptiveRouterOptions,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
)
from models.parametric.parametric_matrix.model import Model
from models.parametric.parametric_matrix.presets import ExperimentOptions, ExperimentPresets
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestParametricMatrixModel(unittest.TestCase):
    def test_preset_builds_current_parametric_config(self):
        cfg = ExperimentPresets()._preset(input_dim=8, hidden_dim=5, output_dim=3)
        handler_config = cfg.experiment_config.model_config.layer_config
        layer_model_config = handler_config.layer_model_config

        self.assertIsInstance(handler_config, ParametricLayerHandlerConfig)
        self.assertIsInstance(layer_model_config, ParametricLayerConfig)
        self.assertIsInstance(
            layer_model_config.weight_mixture_config,
            MatrixWeightsMixtureConfig,
        )
        self.assertIsNone(layer_model_config.bias_mixture_config)
        self.assertEqual(
            layer_model_config.routing_initialization_mode,
            AdaptiveRouterOptions.SHARED_ROUTER,
        )

    def test_bias_option_maps_to_disabled_or_matrix_bias_config(self):
        disabled = ExperimentPresets()._preset(
            input_dim=8,
            hidden_dim=5,
            output_dim=3,
            adaptive_bias_option=None,
        )
        enabled = ExperimentPresets()._preset(
            input_dim=8,
            hidden_dim=5,
            output_dim=3,
            adaptive_bias_option=MatrixBiasMixtureConfig,
        )

        disabled_parametric_config = (
            disabled.experiment_config.model_config.layer_config.layer_model_config
        )
        enabled_parametric_config = (
            enabled.experiment_config.model_config.layer_config.layer_model_config
        )

        self.assertIsNone(disabled_parametric_config.bias_mixture_config)
        self.assertIsInstance(
            enabled_parametric_config.bias_mixture_config,
            MatrixBiasMixtureConfig,
        )

    def test_forward_one_batch_per_dataset(self):
        batch_size = 2
        presets = ExperimentPresets()

        for dataset in config.DATASET_OPTIONS:
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(ExperimentOptions.PRESET, dataset)[0]
                model = Model(cfg)
                model.eval()
                X = self._fake_batch(dataset, batch_size)

                with torch.no_grad():
                    logits, auxiliary_loss = model(X)

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
                self.assertEqual(auxiliary_loss.shape, torch.Size([]))

    def test_all_presets_train_one_epoch(self):
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for option in ExperimentOptions:
            with self.subTest(option=option.name):
                search_mode = (
                    RandomSearch(num_samples=1)
                    if option == ExperimentOptions.CONFIG
                    else None
                )
                cfg = presets.get_config(option, dataset, search_mode)[0]
                model = Model(cfg)
                datamodule = RandomImageClassificationDataModule(dataset)

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def test_config_search_space_builds_configs(self):
        configs = ExperimentPresets().get_config(
            ExperimentOptions.CONFIG,
            config.DATASET_OPTIONS[0],
            RandomSearch(num_samples=2),
        )

        self.assertEqual(len(configs), 2)
        for cfg in configs:
            with self.subTest(hidden_dim=cfg.hidden_dim):
                layer_model_config = (
                    cfg.experiment_config.model_config.layer_config.layer_model_config
                )
                self.assertIsInstance(layer_model_config, ParametricLayerConfig)
                self.assertIsInstance(
                    layer_model_config.weight_mixture_config,
                    MatrixWeightsMixtureConfig,
                )
                self.assertTrue(
                    layer_model_config.bias_mixture_config is None
                    or isinstance(
                        layer_model_config.bias_mixture_config,
                        MatrixBiasMixtureConfig,
                    )
                )

    def test_model_step_accepts_tuple_output(self):
        batch_size = 2
        cfg = ExperimentPresets()._preset(input_dim=8, hidden_dim=4, output_dim=3)
        model = Model(cfg)
        X = torch.randn(batch_size, 1, 2, 4)
        y = torch.tensor([0, 2])

        loss, logits, labels = model._model_step((X, y))

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(logits.shape, (batch_size, 3))
        self.assertEqual(labels.shape, (batch_size,))

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from copy import deepcopy

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import torch

import models.parametric_vector.config as config

from emperor.experiments.base import RandomSearch
from emperor.parametric import (
    AdaptiveRouterOptions,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
    VectorWeightsMixtureConfig,
)
from models.parametric_vector.model import Model
from models.parametric_vector.presets import ExperimentOptions, ExperimentPresets


class TestParametricVectorModel(unittest.TestCase):
    def test_preset_builds_current_parametric_config(self):
        hidden_dim = 5
        cfg = ExperimentPresets()._preset(
            input_dim=8,
            hidden_dim=hidden_dim,
            output_dim=3,
        )
        stack_config = cfg.experiment_config.model_config
        handler_config = stack_config.layer_config
        layer_model_config = handler_config.layer_model_config

        self.assertIsInstance(handler_config, ParametricLayerHandlerConfig)
        self.assertIsInstance(layer_model_config, ParametricLayerConfig)
        self.assertIsInstance(
            layer_model_config.weight_mixture_config,
            VectorWeightsMixtureConfig,
        )
        self.assertIsNone(layer_model_config.bias_mixture_config)
        self.assertEqual(
            layer_model_config.routing_initialization_mode,
            AdaptiveRouterOptions.INDEPENDENT_ROUTER,
        )
        self.assertEqual(stack_config.input_dim, hidden_dim)
        self.assertEqual(stack_config.hidden_dim, hidden_dim)
        self.assertEqual(stack_config.output_dim, hidden_dim)
        self.assertEqual(layer_model_config.input_dim, hidden_dim)
        self.assertEqual(layer_model_config.output_dim, hidden_dim)
        self.assertEqual(layer_model_config.weight_mixture_config.input_dim, hidden_dim)
        self.assertEqual(
            layer_model_config.weight_mixture_config.output_dim,
            hidden_dim,
        )

    def test_vector_shared_router_is_rejected(self):
        cfg = ExperimentPresets()._preset(input_dim=8, hidden_dim=4, output_dim=3)
        parametric_config = deepcopy(
            cfg.experiment_config.model_config.layer_config.layer_model_config
        )
        parametric_config.routing_initialization_mode = (
            AdaptiveRouterOptions.SHARED_ROUTER
        )

        with self.assertRaises(ValueError):
            parametric_config.build()

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
                    VectorWeightsMixtureConfig,
                )
                self.assertIsNone(layer_model_config.bias_mixture_config)

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

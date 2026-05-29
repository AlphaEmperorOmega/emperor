import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import torch

import models.parametric_generator.config as config

from emperor.experiments.base import RandomSearch
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core.options import RoutingInitializationMode
from emperor.parametric import (
    AdaptiveRouterOptions,
    GeneratorBiasMixtureConfig,
    GeneratorWeightsMixtureConfig,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
)
from models.parametric_generator.model import Model
from models.parametric_generator.presets import ExperimentOptions, ExperimentPresets


class TestParametricGeneratorModel(unittest.TestCase):
    def test_preset_builds_current_parametric_config(self):
        cfg = ExperimentPresets()._preset(input_dim=8, hidden_dim=5, output_dim=3)
        handler_config = cfg.experiment_config.model_config.layer_config
        layer_model_config = handler_config.layer_model_config

        self.assertIsInstance(handler_config, ParametricLayerHandlerConfig)
        self.assertIsInstance(layer_model_config, ParametricLayerConfig)
        self.assertIsInstance(
            layer_model_config.weight_mixture_config,
            GeneratorWeightsMixtureConfig,
        )
        self.assertIsNone(layer_model_config.bias_mixture_config)
        self.assertEqual(
            layer_model_config.routing_initialization_mode,
            AdaptiveRouterOptions.SHARED_ROUTER,
        )

    def test_generator_weight_and_bias_configs_use_matching_moe_configs(self):
        cfg = ExperimentPresets()._preset(
            input_dim=8,
            hidden_dim=5,
            output_dim=3,
            adaptive_mixture_top_k=1,
            adaptive_mixture_num_experts=2,
            adaptive_bias_option=GeneratorBiasMixtureConfig,
        )
        parametric_config = (
            cfg.experiment_config.model_config.layer_config.layer_model_config
        )
        mixture_configs = [
            parametric_config.weight_mixture_config,
            parametric_config.bias_mixture_config,
        ]

        for mixture_config in mixture_configs:
            with self.subTest(mixture=type(mixture_config).__name__):
                generator_config = mixture_config.generator_config
                self.assertIsInstance(generator_config, MixtureOfExpertsConfig)
                self.assertEqual(generator_config.top_k, mixture_config.top_k)
                self.assertEqual(
                    generator_config.num_experts,
                    mixture_config.num_experts,
                )
                self.assertEqual(
                    generator_config.routing_initialization_mode,
                    RoutingInitializationMode.DISABLED,
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

    def test_config_search_space_builds_configs(self):
        configs = ExperimentPresets().get_config(
            ExperimentOptions.CONFIG,
            config.DATASET_OPTIONS[0],
            RandomSearch(num_samples=2),
        )

        self.assertEqual(len(configs), 2)
        for cfg in configs:
            with self.subTest(output_dim=cfg.output_dim):
                layer_model_config = (
                    cfg.experiment_config.model_config.layer_config.layer_model_config
                )
                self.assertIsInstance(layer_model_config, ParametricLayerConfig)
                self.assertIsInstance(
                    layer_model_config.weight_mixture_config,
                    GeneratorWeightsMixtureConfig,
                )
                self.assertTrue(
                    layer_model_config.bias_mixture_config is None
                    or isinstance(
                        layer_model_config.bias_mixture_config,
                        GeneratorBiasMixtureConfig,
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

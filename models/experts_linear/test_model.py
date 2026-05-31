import unittest

import torch

import models.experts_linear.config as config

from emperor.base.layer import RecurrentLayerConfig
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.experiments.base import RandomSearch
from models.experts_linear.model import Model
from models.experts_linear.presets import ExperimentOptions, ExperimentPresets


class TestExpertsLinearModel(unittest.TestCase):
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

                output = model(X)
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )

    def test_preset_accepts_search_flags(self):
        configs = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
            config.DATASET_OPTIONS[0],
            RandomSearch(num_samples=2),
        )

        self.assertEqual(len(configs), 2)

    def test_recurrent_presets_wrap_full_moe_model(self):
        expected_controllers = {
            ExperimentOptions.RECURRENT: (False, False),
            ExperimentOptions.RECURRENT_GATING: (True, False),
            ExperimentOptions.RECURRENT_HALTING: (False, True),
            ExperimentOptions.RECURRENT_GATING_HALTING: (True, True),
        }

        for option, (expected_gate, expected_halting) in expected_controllers.items():
            with self.subTest(option=option.name):
                cfg = ExperimentPresets().get_config(option)[0]
                recurrent_cfg = cfg.experiment_config.model_config

                self.assertIsInstance(recurrent_cfg, RecurrentLayerConfig)
                self.assertEqual(recurrent_cfg.max_steps, config.RECURRENT_MAX_STEPS)
                self.assertIsInstance(
                    recurrent_cfg.block_config,
                    MixtureOfExpertsModelConfig,
                )
                self.assertEqual(recurrent_cfg.gate_config is not None, expected_gate)
                self.assertEqual(
                    recurrent_cfg.halting_config is not None,
                    expected_halting,
                )
                inner_layer_config = (
                    recurrent_cfg.block_config.stack_config.layer_config
                )
                self.assertIsNone(inner_layer_config.gate_config)
                self.assertIsNone(inner_layer_config.halting_config)

    def test_new_moe_combination_presets_wire_config(self):
        presets = ExperimentPresets()

        cfg = presets.get_config(ExperimentOptions.SHARED_ROUTER_AFTER_WEIGHT)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        moe_layer_cfg = moe_model_cfg.stack_config.layer_config.layer_model_config
        self.assertEqual(
            moe_model_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self.assertEqual(
            moe_layer_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self.assertEqual(
            moe_layer_cfg.weighting_position_option,
            ExpertWeightingPositionOptions.AFTER_EXPERTS,
        )

        cfg = presets.get_config(ExperimentOptions.TOP1_SWITCH_AUX)[0]
        moe_layer_cfg = self._moe_layer_config(cfg)
        self.assertEqual(moe_layer_cfg.top_k, 1)
        self.assertFalse(moe_layer_cfg.sampler_config.normalize_probabilities_flag)
        self.assertEqual(moe_layer_cfg.sampler_config.switch_loss_weight, 0.1)

        cfg = presets.get_config(ExperimentOptions.TOP2_BALANCED_AUX)[0]
        moe_layer_cfg = self._moe_layer_config(cfg)
        self.assertEqual(moe_layer_cfg.top_k, 2)
        self.assertEqual(
            moe_layer_cfg.sampler_config.coefficient_of_variation_loss_weight,
            0.1,
        )

        expected_capacity = {
            ExperimentOptions.CAPACITY_TOP1_ZERO: DroppedTokenOptions.ZEROS,
            ExperimentOptions.CAPACITY_TOP1_IDENTITY: DroppedTokenOptions.IDENTITY,
        }
        for option, expected_dropped_behavior in expected_capacity.items():
            with self.subTest(option=option.name):
                cfg = presets.get_config(option)[0]
                moe_layer_cfg = self._moe_layer_config(cfg)

                self.assertEqual(moe_layer_cfg.top_k, 1)
                self.assertEqual(moe_layer_cfg.capacity_factor, 1.0)
                self.assertFalse(
                    moe_layer_cfg.sampler_config.normalize_probabilities_flag
                )
                self.assertEqual(
                    moe_layer_cfg.dropped_token_behavior,
                    expected_dropped_behavior,
                )

        cfg = presets.get_config(ExperimentOptions.NOISY_SHARED_ROUTER)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        moe_layer_cfg = self._moe_layer_config(cfg)
        self.assertEqual(
            moe_model_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self.assertTrue(moe_model_cfg.sampler_config.noisy_topk_flag)
        self.assertTrue(moe_model_cfg.sampler_config.router_config.noisy_topk_flag)
        self.assertTrue(moe_layer_cfg.sampler_config.noisy_topk_flag)
        self.assertTrue(moe_layer_cfg.sampler_config.router_config.noisy_topk_flag)

        cfg = presets.get_config(ExperimentOptions.RESIDUAL_SHARED_ROUTER)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        layer_cfg = moe_model_cfg.stack_config.layer_config
        self.assertEqual(
            moe_model_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self.assertTrue(layer_cfg.residual_flag)

        cfg = presets.get_config(ExperimentOptions.POST_NORM_AFTER_WEIGHT)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        layer_cfg = moe_model_cfg.stack_config.layer_config
        moe_layer_cfg = layer_cfg.layer_model_config
        self.assertEqual(
            layer_cfg.layer_norm_position,
            config.LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(
            moe_layer_cfg.weighting_position_option,
            ExpertWeightingPositionOptions.AFTER_EXPERTS,
        )

    def test_auxiliary_loss_presets_return_finite_loss(self):
        batch_size = 4
        dataset = config.DATASET_OPTIONS[0]
        presets = ExperimentPresets()

        for option in (
            ExperimentOptions.TOP1_SWITCH_AUX,
            ExperimentOptions.TOP2_BALANCED_AUX,
        ):
            with self.subTest(option=option.name):
                cfg = presets.get_config(option, dataset)[0]
                model = Model(cfg)
                X = self._fake_batch(dataset, batch_size)

                output = model(X)

                self.assertIsInstance(output, tuple)
                logits, auxiliary_loss = output
                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
                self.assertTrue(torch.isfinite(auxiliary_loss).item())

    def _moe_layer_config(self, cfg):
        model_config = cfg.experiment_config.model_config
        if isinstance(model_config, RecurrentLayerConfig):
            model_config = model_config.block_config
        return model_config.stack_config.layer_config.layer_model_config


if __name__ == "__main__":
    unittest.main()

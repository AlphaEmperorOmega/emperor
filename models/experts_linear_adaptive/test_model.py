import unittest

import torch

import models.experts_linear_adaptive.config as config

from emperor.augmentations.adaptive_parameters.core.bias import (
    AdditiveDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    CombinedDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
)
from emperor.base.layer import RecurrentLayerConfig
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.experiments.base import RandomSearch
from models.config_overrides import parse_config_value
from models.experts_linear_adaptive.model import Model
from models.experts_linear_adaptive.presets import ExperimentOptions, ExperimentPresets


class TestExpertsLinearAdaptiveModel(unittest.TestCase):
    def test_all_options_forward_one_mnist_batch(self):
        batch_size = 2
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
        batch_size = 2
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

    def test_new_adaptive_moe_presets_wire_config(self):
        presets = ExperimentPresets()

        cfg = presets.get_config(ExperimentOptions.ADAPTIVE_SHARED_ROUTER)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        self.assertEqual(
            moe_model_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self.assertIsInstance(
            self._expert_augmentation_config(cfg).weight_config,
            DualModelDynamicWeightConfig,
        )

        cfg = presets.get_config(ExperimentOptions.ADAPTIVE_AFTER_WEIGHT)[0]
        moe_layer_cfg = self._moe_layer_config(cfg)
        self.assertEqual(
            moe_layer_cfg.weighting_position_option,
            ExpertWeightingPositionOptions.AFTER_EXPERTS,
        )
        self.assertIsInstance(
            self._expert_augmentation_config(cfg).weight_config,
            DualModelDynamicWeightConfig,
        )

        cfg = presets.get_config(ExperimentOptions.ADAPTIVE_TOP1_SWITCH)[0]
        moe_layer_cfg = self._moe_layer_config(cfg)
        self.assertEqual(moe_layer_cfg.top_k, 1)
        self.assertFalse(moe_layer_cfg.sampler_config.normalize_probabilities_flag)
        self.assertEqual(moe_layer_cfg.sampler_config.switch_loss_weight, 0.1)
        self.assertIsInstance(
            self._expert_augmentation_config(cfg).weight_config,
            DualModelDynamicWeightConfig,
        )

        cfg = presets.get_config(ExperimentOptions.ADAPTIVE_FULL_SHARED)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        self.assertEqual(
            moe_model_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self._assert_full_adaptive_augmentation(
            self._expert_augmentation_config(cfg)
        )

        cfg = presets.get_config(ExperimentOptions.ADAPTIVE_FULL_CAPACITY)[0]
        moe_layer_cfg = self._moe_layer_config(cfg)
        self.assertEqual(moe_layer_cfg.top_k, 1)
        self.assertEqual(moe_layer_cfg.capacity_factor, 1.0)
        self.assertFalse(moe_layer_cfg.sampler_config.normalize_probabilities_flag)
        self.assertEqual(
            moe_layer_cfg.dropped_token_behavior,
            DroppedTokenOptions.ZEROS,
        )
        self._assert_full_adaptive_augmentation(
            self._expert_augmentation_config(cfg)
        )

        cfg = presets.get_config(ExperimentOptions.ADAPTIVE_BANK_ROUTER)[0]
        moe_model_cfg = cfg.experiment_config.model_config
        router_augmentation_config = (
            moe_model_cfg.sampler_config.router_config.model_config.layer_config
            .layer_model_config.adaptive_augmentation_config
        )
        self.assertEqual(
            moe_model_cfg.routing_initialization_mode,
            RoutingInitializationMode.SHARED,
        )
        self.assertIsInstance(
            router_augmentation_config.weight_config,
            LayeredWeightedBankDynamicWeightConfig,
        )

    def test_auxiliary_loss_presets_return_finite_loss(self):
        batch_size = 2
        dataset = config.DATASET_OPTIONS[0]
        cfg = ExperimentPresets().get_config(
            ExperimentOptions.ADAPTIVE_TOP1_SWITCH,
            dataset,
        )[0]
        model = Model(cfg)
        X = self._fake_batch(dataset, batch_size)

        output = model(X)

        self.assertIsInstance(output, tuple)
        logits, auxiliary_loss = output
        self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
        self.assertTrue(torch.isfinite(auxiliary_loss).item())

    def test_cli_config_overrides_resolve_adaptive_class_names(self):
        self.assertIs(
            parse_config_value(
                config,
                "WEIGHT_OPTION",
                "DualModelDynamicWeightConfig",
            ),
            DualModelDynamicWeightConfig,
        )
        self.assertIs(
            parse_config_value(
                config,
                "BIAS_OPTION",
                "AdditiveDynamicBiasConfig",
            ),
            AdditiveDynamicBiasConfig,
        )
        self.assertIs(
            parse_config_value(
                config,
                "DIAGONAL_OPTION",
                "CombinedDynamicDiagonalConfig",
            ),
            CombinedDynamicDiagonalConfig,
        )
        self.assertIs(
            parse_config_value(
                config,
                "ROW_MASK_OPTION",
                "WeightInformedScoreAxisMaskConfig",
            ),
            WeightInformedScoreAxisMaskConfig,
        )

    def _moe_layer_config(self, cfg):
        model_config = cfg.experiment_config.model_config
        if isinstance(model_config, RecurrentLayerConfig):
            model_config = model_config.block_config
        return model_config.stack_config.layer_config.layer_model_config

    def _expert_augmentation_config(self, cfg):
        return (
            self._moe_layer_config(cfg).expert_model_config.layer_config
            .layer_model_config.adaptive_augmentation_config
        )

    def _assert_full_adaptive_augmentation(self, augmentation_config) -> None:
        self.assertIsInstance(
            augmentation_config.weight_config,
            DualModelDynamicWeightConfig,
        )
        self.assertIsInstance(
            augmentation_config.bias_config,
            AdditiveDynamicBiasConfig,
        )
        self.assertIsInstance(
            augmentation_config.diagonal_config,
            CombinedDynamicDiagonalConfig,
        )
        self.assertIsInstance(
            augmentation_config.mask_config,
            WeightInformedScoreAxisMaskConfig,
        )


if __name__ == "__main__":
    unittest.main()

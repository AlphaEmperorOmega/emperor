from __future__ import annotations

import unittest

import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdditiveDynamicBiasConfig,
    DualModelDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    PerAxisScoreMaskConfig,
    StandardDynamicDiagonalConfig,
)
from emperor.linears import LinearLayerConfig
from models.mlp_mixer.linear_adaptive.config_builder import (
    MlpMixerLinearAdaptiveConfigBuilder,
)
from models.mlp_mixer.linear_adaptive.model import Model
from models.mlp_mixer.linear_adaptive.presets import ExperimentPreset
from models.mlp_mixer.linear_adaptive.runtime_defaults import runtime_from_flat
from models.mlp_mixer.linear_adaptive.runtime_options import RuntimeOptions
from support.mlp_mixer_package import MlpMixerPackageContractMixin


class MlpMixerLinearAdaptivePackageTests(
    MlpMixerPackageContractMixin,
    unittest.TestCase,
):
    MODEL_ID = "mlp_mixer/linear_adaptive"
    ADAPTIVE = True
    FIT_PRESETS = ("adaptive",)

    def test_adaptive_layers_are_confined_to_token_and_channel_mixers(self) -> None:
        runtime = runtime_from_flat(self._small_overrides())
        config = MlpMixerLinearAdaptiveConfigBuilder(runtime=runtime).build()
        model = Model(config)
        block_config = config.experiment_config.encoder_config.layer_config
        transformer_config = block_config.layer_model_config
        token_config = transformer_config.attention_config.mixing_model_config
        channel_config = transformer_config.feed_forward_config.stack_config

        self.assertIsInstance(runtime, RuntimeOptions)
        self.assertIsInstance(
            token_config.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )
        self.assertIsInstance(
            channel_config.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )
        self.assertIsInstance(
            config.experiment_config.patch_config.embedding_stack_config.layer_config.layer_model_config,
            LinearLayerConfig,
        )
        self.assertIsInstance(
            config.experiment_config.output_config.layer_model_config,
            LinearLayerConfig,
        )
        adaptive_names = [
            name for name, _ in model.named_parameters() if "adaptive_behaviour" in name
        ]
        self.assertTrue(adaptive_names)
        self.assertTrue(all(name.startswith("transformer.") for name in adaptive_names))
        self.assertIn(ExperimentPreset.ADAPTIVE, tuple(ExperimentPreset))

    def test_complete_adaptive_parameter_surface_is_wired(self) -> None:
        runtime = runtime_from_flat(
            self._small_overrides(
                weight_option_flag=True,
                weight_option=DualModelDynamicWeightConfig,
                bias_option_flag=True,
                bias_option=AdditiveDynamicBiasConfig,
                mask_option_flag=True,
                row_mask_option=PerAxisScoreMaskConfig,
                weight_generator_stack_independent_flag=True,
                weight_generator_stack_hidden_dim=5,
                bias_generator_stack_independent_flag=True,
                bias_generator_stack_hidden_dim=6,
                diagonal_generator_stack_independent_flag=True,
                diagonal_generator_stack_hidden_dim=7,
                mask_generator_stack_independent_flag=True,
                mask_generator_stack_hidden_dim=8,
            )
        )
        config = MlpMixerLinearAdaptiveConfigBuilder(runtime=runtime).build()
        token_stack = config.experiment_config.encoder_config.layer_config.layer_model_config.attention_config.mixing_model_config
        augmentation = (
            token_stack.layer_config.layer_model_config.adaptive_augmentation_config
        )

        self.assertIsInstance(augmentation.weight_config, DualModelDynamicWeightConfig)
        self.assertIsInstance(augmentation.bias_config, AdditiveDynamicBiasConfig)
        self.assertIsInstance(
            augmentation.diagonal_config,
            StandardDynamicDiagonalConfig,
        )
        self.assertIsInstance(augmentation.mask_config, PerAxisScoreMaskConfig)
        self.assertEqual(augmentation.model_config.hidden_dim, 4)
        self.assertEqual(augmentation.weight_config.model_config.hidden_dim, 5)
        self.assertEqual(augmentation.bias_config.model_config.hidden_dim, 6)
        self.assertEqual(augmentation.diagonal_config.model_config.hidden_dim, 7)
        self.assertEqual(augmentation.mask_config.model_config.hidden_dim, 8)

        model = Model(config)
        inputs = torch.randn(2, 1, 8, 8, requires_grad=True)
        logits = model(inputs)
        logits.square().mean().backward()
        adaptive_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "adaptive_behaviour" in name and parameter.grad is not None
        ]
        self.assertTrue(adaptive_gradients)
        self.assertTrue(
            any(gradient.abs().sum().item() > 0.0 for gradient in adaptive_gradients)
        )

        bank_model = self._model(
            diagonal_option_flag=False,
            weight_option_flag=True,
            weight_option=LayeredWeightedBankDynamicWeightConfig,
        )
        bank_logits = bank_model(torch.randn(2, 1, 8, 8))
        bank_logits.square().mean().backward()
        bank_modules = [
            module
            for module in bank_model.modules()
            if type(module).__name__ == "LayeredWeightedBankDynamicWeight"
        ]
        self.assertTrue(bank_modules)
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
                for module in bank_modules
                for parameter in module.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()

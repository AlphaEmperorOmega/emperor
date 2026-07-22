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
from emperor.experts import MixtureOfExpertsModelConfig, RoutingInitializationMode
from emperor.linears import LinearLayerConfig
from models.mlp_mixer.expert_linear_adaptive.config_builder import (
    MlpMixerExpertLinearAdaptiveConfigBuilder,
)
from models.mlp_mixer.expert_linear_adaptive.model import Model
from models.mlp_mixer.expert_linear_adaptive.presets import ExperimentPreset
from models.mlp_mixer.expert_linear_adaptive.runtime_defaults import runtime_from_flat
from models.mlp_mixer.expert_linear_adaptive.runtime_options import RuntimeOptions
from support.mlp_mixer_package import MlpMixerPackageContractMixin


class MlpMixerExpertLinearAdaptivePackageTests(
    MlpMixerPackageContractMixin,
    unittest.TestCase,
):
    MODEL_ID = "mlp_mixer/expert_linear_adaptive"
    ADAPTIVE = True
    EXPERT = True
    FIT_PRESETS = ("expert-auxiliary-loss",)

    def test_real_moe_uses_adaptive_experts_but_plain_boundaries(self) -> None:
        runtime = runtime_from_flat(
            self._small_overrides(
                capacity_factor=0.5,
                token_mixer_stack_hidden_dim=4,
                channel_mixer_stack_hidden_dim=8,
            )
        )
        config = MlpMixerExpertLinearAdaptiveConfigBuilder(runtime=runtime).build()
        block_config = config.experiment_config.encoder_config.layer_config
        transformer_config = block_config.layer_model_config
        token_config = transformer_config.attention_config.mixing_model_config
        channel_config = transformer_config.feed_forward_config.stack_config

        self.assertIsInstance(runtime, RuntimeOptions)
        self.assertIsInstance(token_config, MixtureOfExpertsModelConfig)
        self.assertIsInstance(channel_config, MixtureOfExpertsModelConfig)
        self.assertIsInstance(
            token_config.stack_config.layer_config.layer_model_config.expert_model_config.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )
        self.assertIsInstance(
            channel_config.stack_config.layer_config.layer_model_config.expert_model_config.layer_config.layer_model_config,
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

        model = Model(config)
        mixtures = [
            module
            for module in model.modules()
            if type(module).__name__ == "MixtureOfExperts"
        ]
        self.assertEqual(len(mixtures), 4)
        self.assertTrue(all(mixture.capacity_factor == 0.5 for mixture in mixtures))
        self.assertTrue(
            all(mixture.compute_expert_mixture_flag for mixture in mixtures)
        )
        self.assertTrue(all(mixture.sampler is not None for mixture in mixtures))
        self.assertTrue(all(len(mixture.expert_modules) == 3 for mixture in mixtures))
        adaptive_names = [
            name for name, _ in model.named_parameters() if "adaptive_behaviour" in name
        ]
        self.assertTrue(adaptive_names)
        self.assertTrue(
            all(".expert_modules." in name for name in adaptive_names),
        )

        auxiliary_model = self._model(ExperimentPreset.EXPERT_AUXILIARY_LOSS)
        output = auxiliary_model(torch.randn(2, 1, 8, 8))
        self.assertIsInstance(output, tuple)
        self.assertTrue(torch.isfinite(output[-1]).item())
        self.assertGreater(output[-1].item(), 0.0)

        shared_model = self._model(
            routing_initialization_mode=RoutingInitializationMode.SHARED
        )
        shared_mixture_models = [
            module
            for module in shared_model.modules()
            if type(module).__name__ == "MixtureOfExpertsModel"
        ]
        shared_mixtures = [
            module
            for module in shared_model.modules()
            if type(module).__name__ == "MixtureOfExperts"
        ]
        self.assertEqual(len(shared_mixture_models), 2)
        self.assertTrue(
            all(model.shared_sampler is not None for model in shared_mixture_models)
        )
        self.assertTrue(all(mixture.sampler is None for mixture in shared_mixtures))
        shared_logits, _ = self._logits_and_loss(shared_model(torch.randn(2, 1, 8, 8)))
        self.assertEqual(shared_logits.shape, (2, 3))

    def test_complete_adaptive_parameter_surface_is_wired_inside_experts(
        self,
    ) -> None:
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
        config = MlpMixerExpertLinearAdaptiveConfigBuilder(runtime=runtime).build()
        token_mixture = config.experiment_config.encoder_config.layer_config.layer_model_config.attention_config.mixing_model_config
        adaptive_linear = token_mixture.stack_config.layer_config.layer_model_config.expert_model_config.layer_config.layer_model_config
        augmentation = adaptive_linear.adaptive_augmentation_config

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
        logits, auxiliary_loss = self._logits_and_loss(model(inputs))
        (logits.square().mean() + auxiliary_loss).backward()
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
        bank_logits, bank_loss = self._logits_and_loss(
            bank_model(torch.randn(2, 1, 8, 8))
        )
        (bank_logits.square().mean() + bank_loss).backward()
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

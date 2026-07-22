from __future__ import annotations

import unittest

import torch

from emperor.experts import MixtureOfExpertsModelConfig, RoutingInitializationMode
from emperor.linears import LinearLayerConfig
from models.mlp_mixer.expert_linear.config_builder import (
    MlpMixerExpertLinearConfigBuilder,
)
from models.mlp_mixer.expert_linear.model import Model
from models.mlp_mixer.expert_linear.presets import ExperimentPreset
from models.mlp_mixer.expert_linear.runtime_defaults import runtime_from_flat
from models.mlp_mixer.expert_linear.runtime_options import RuntimeOptions
from support.mlp_mixer_package import MlpMixerPackageContractMixin


class MlpMixerExpertLinearPackageTests(
    MlpMixerPackageContractMixin,
    unittest.TestCase,
):
    MODEL_ID = "mlp_mixer/expert_linear"
    EXPERT = True
    FIT_PRESETS = ("top-1-expert",)

    def test_real_moe_owns_both_mixers_and_propagates_auxiliary_loss(self) -> None:
        runtime = runtime_from_flat(
            self._small_overrides(
                capacity_factor=0.5,
                token_mixer_stack_hidden_dim=4,
                channel_mixer_stack_hidden_dim=8,
            )
        )
        config = MlpMixerExpertLinearConfigBuilder(runtime=runtime).build()
        model = Model(config)
        block_config = config.experiment_config.encoder_config.layer_config
        transformer_config = block_config.layer_model_config

        self.assertIsInstance(runtime, RuntimeOptions)
        self.assertIsInstance(
            transformer_config.attention_config.mixing_model_config,
            MixtureOfExpertsModelConfig,
        )
        self.assertIsInstance(
            transformer_config.feed_forward_config.stack_config,
            MixtureOfExpertsModelConfig,
        )
        self.assertIsInstance(
            config.experiment_config.patch_config.embedding_stack_config.layer_config.layer_model_config,
            LinearLayerConfig,
        )
        self.assertIsInstance(
            config.experiment_config.output_config.layer_model_config,
            LinearLayerConfig,
        )
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

        auxiliary_model = self._model(ExperimentPreset.EXPERT_AUXILIARY_LOSS)
        output = auxiliary_model(torch.randn(2, 1, 8, 8))
        self.assertIsInstance(output, tuple)
        self.assertTrue(torch.isfinite(output[-1]).item())
        self.assertGreater(output[-1].item(), 0.0)
        module_names = {type(module).__name__ for module in auxiliary_model.modules()}
        self.assertIn("MixtureOfExperts", module_names)
        self.assertIn("SamplerTopk", module_names)

        top_one_model = self._model(ExperimentPreset.TOP_1_EXPERT)
        self.assertIn(
            "SamplerSparse",
            {type(module).__name__ for module in top_one_model.modules()},
        )

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

        with self.assertRaisesRegex(ValueError, "must be LAYER or SHARED"):
            self._model(routing_initialization_mode=RoutingInitializationMode.DISABLED)


if __name__ == "__main__":
    unittest.main()

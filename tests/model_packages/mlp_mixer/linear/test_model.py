from __future__ import annotations

import unittest

from emperor.linears import LinearLayerConfig
from models.mlp_mixer.linear.config_builder import MlpMixerLinearConfigBuilder
from models.mlp_mixer.linear.model import Model
from models.mlp_mixer.linear.presets import ExperimentPreset
from models.mlp_mixer.linear.runtime_defaults import runtime_from_flat
from models.mlp_mixer.linear.runtime_options import RuntimeOptions
from support.mlp_mixer_package import MlpMixerPackageContractMixin


class MlpMixerLinearPackageTests(
    MlpMixerPackageContractMixin,
    unittest.TestCase,
):
    MODEL_ID = "mlp_mixer/linear"
    FIT_PRESETS = ("baseline", "recurrent")

    def test_package_owned_builder_uses_only_plain_boundary_and_mixer_linears(
        self,
    ) -> None:
        runtime = runtime_from_flat(self._small_overrides())
        config = MlpMixerLinearConfigBuilder(runtime=runtime).build()
        model = Model(config)

        self.assertIsInstance(runtime, RuntimeOptions)
        self.assertIsInstance(
            config.experiment_config.patch_config.embedding_stack_config.layer_config.layer_model_config,
            LinearLayerConfig,
        )
        self.assertIsInstance(
            config.experiment_config.output_config.layer_model_config,
            LinearLayerConfig,
        )
        self.assertFalse(
            any(
                "adaptive_behaviour" in name or ".sampler." in name
                for name, _ in model.named_parameters()
            )
        )
        self.assertIs(ExperimentPreset.BASELINE, next(iter(ExperimentPreset)))


if __name__ == "__main__":
    unittest.main()

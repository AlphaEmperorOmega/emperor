import unittest

import models.neuron.linear.config as config
from models.linears.linear.presets import (
    ExperimentPreset as SourceExperimentPreset,
    ExperimentPresets as SourceExperimentPresets,
)
from models.neuron._test_cases import NeuronPackageTestMixin
from models.neuron.linear._source_linear_adapter import (
    SOURCE_ADAPTER,
    canonical_source_kwarg_aliases,
    source_builder_kwargs_from_flat,
    source_linear_default_kwargs,
)
from models.neuron.linear.config_builder import NeuronLinearConfigBuilder
from models.neuron.linear.model import Model
from models.neuron.linear.presets import ExperimentPreset, ExperimentPresets


class TestNeuronLinearModel(NeuronPackageTestMixin, unittest.TestCase):
    package_module = "models.neuron.linear"
    config_module = config
    builder_type = NeuronLinearConfigBuilder
    model_type = Model
    experiment_preset_type = ExperimentPreset
    experiment_presets_type = ExperimentPresets
    source_experiment_preset_type = SourceExperimentPreset
    source_experiment_presets_type = SourceExperimentPresets
    source_adapter = SOURCE_ADAPTER

    def test_source_linear_adapter_defaults_adapt_to_source_builder(self):
        from models.linears.linear.config_builder import LinearConfigBuilder

        source_defaults = source_linear_default_kwargs()
        source_kwargs = source_builder_kwargs_from_flat(source_defaults)
        default_cfg = LinearConfigBuilder().build()
        adapted_cfg = LinearConfigBuilder(**source_kwargs).build()

        self.assertIn("hidden_dim", source_defaults)
        self.assertNotIn("stack_options", source_defaults)
        with self.assertRaises(TypeError):
            LinearConfigBuilder(**source_defaults).build()
        self.assertEqual(adapted_cfg, default_cfg)

    def test_source_linear_adapter_exposes_legacy_aliases(self):
        self.assertEqual(
            canonical_source_kwarg_aliases(),
            {
                "gate_hidden_dim": "gate_stack_hidden_dim",
                "gate_layer_norm_position": "gate_stack_layer_norm_position",
                "gate_bias_flag": "gate_stack_bias_flag",
                "halting_hidden_dim": "halting_stack_hidden_dim",
                "halting_layer_norm_position": "halting_stack_layer_norm_position",
                "halting_bias_flag": "halting_stack_bias_flag",
            },
        )

    def test_builder_uses_source_linear_adapter_defaults(self):
        source_defaults = source_linear_default_kwargs()
        cfg = NeuronLinearConfigBuilder().build()
        override_cfg = NeuronLinearConfigBuilder(hidden_dim=128).build()

        self.assertEqual(cfg.hidden_dim, source_defaults["hidden_dim"])
        self.assertEqual(override_cfg.hidden_dim, 128)


if __name__ == "__main__":
    unittest.main()

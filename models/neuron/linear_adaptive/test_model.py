import unittest

import models.neuron.linear_adaptive.config as config
from models.linears.linear_adaptive.presets import (
    ExperimentPreset as SourceExperimentPreset,
    ExperimentPresets as SourceExperimentPresets,
)
from models.neuron._test_cases import NeuronPackageTestMixin
from models.neuron.linear_adaptive._source_linear_adaptive_adapter import (
    SOURCE_ADAPTER,
)
from models.neuron.linear_adaptive.config_builder import (
    NeuronLinearAdaptiveConfigBuilder,
)
from models.neuron.linear_adaptive.model import Model
from models.neuron.linear_adaptive.presets import ExperimentPreset, ExperimentPresets


class TestNeuronLinearAdaptiveModel(NeuronPackageTestMixin, unittest.TestCase):
    package_module = "models.neuron.linear_adaptive"
    config_module = config
    builder_type = NeuronLinearAdaptiveConfigBuilder
    model_type = Model
    experiment_preset_type = ExperimentPreset
    experiment_presets_type = ExperimentPresets
    source_experiment_preset_type = SourceExperimentPreset
    source_experiment_presets_type = SourceExperimentPresets
    source_adapter = SOURCE_ADAPTER


if __name__ == "__main__":
    unittest.main()

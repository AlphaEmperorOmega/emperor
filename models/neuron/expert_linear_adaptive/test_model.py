import unittest

import models.neuron.expert_linear_adaptive.config as config
from models.experts.linear_adaptive.presets import (
    ExperimentPreset as SourceExperimentPreset,
    ExperimentPresets as SourceExperimentPresets,
)
from models.neuron._test_cases import NeuronPackageTestMixin
from models.neuron.expert_linear_adaptive._source_expert_linear_adaptive_adapter import (
    SOURCE_ADAPTER,
)
from models.neuron.expert_linear_adaptive.config_builder import (
    NeuronExpertLinearAdaptiveConfigBuilder,
)
from models.neuron.expert_linear_adaptive.model import Model
from models.neuron.expert_linear_adaptive.presets import (
    ExperimentPreset,
    ExperimentPresets,
)


class TestNeuronExpertLinearAdaptiveModel(
    NeuronPackageTestMixin,
    unittest.TestCase,
):
    package_module = "models.neuron.expert_linear_adaptive"
    config_module = config
    builder_type = NeuronExpertLinearAdaptiveConfigBuilder
    model_type = Model
    experiment_preset_type = ExperimentPreset
    experiment_presets_type = ExperimentPresets
    source_experiment_preset_type = SourceExperimentPreset
    source_experiment_presets_type = SourceExperimentPresets
    source_adapter = SOURCE_ADAPTER


if __name__ == "__main__":
    unittest.main()

import unittest

import models.neuron.expert_linear.config as config
from models.experts.linear.presets import (
    ExperimentPreset as SourceExperimentPreset,
    ExperimentPresets as SourceExperimentPresets,
)
from models.neuron._test_cases import NeuronPackageTestMixin
from models.neuron.expert_linear._source_expert_linear_adapter import SOURCE_ADAPTER
from models.neuron.expert_linear.config_builder import (
    NeuronExpertLinearConfigBuilder,
)
from models.neuron.expert_linear.model import Model
from models.neuron.expert_linear.presets import ExperimentPreset, ExperimentPresets


class TestNeuronExpertLinearModel(NeuronPackageTestMixin, unittest.TestCase):
    package_module = "models.neuron.expert_linear"
    config_module = config
    builder_type = NeuronExpertLinearConfigBuilder
    model_type = Model
    experiment_preset_type = ExperimentPreset
    experiment_presets_type = ExperimentPresets
    source_experiment_preset_type = SourceExperimentPreset
    source_experiment_presets_type = SourceExperimentPresets
    source_adapter = SOURCE_ADAPTER


if __name__ == "__main__":
    unittest.main()

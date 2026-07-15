import unittest
from types import SimpleNamespace

import torch
from emperor.neuron.core._validator import (
    NeuronClusterValidator,
    NeuronValidator,
)
from emperor.neuron.core.config import NeuronClusterConfig, NeuronConfig
from emperor.neuron.core.model import NeuronCluster


class TestNeuronClusterValidatorAdapter(unittest.TestCase):
    def test_module_declares_its_validator_adapter(self):
        self.assertIs(NeuronCluster.VALIDATOR, NeuronClusterValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(NeuronClusterValidator):
            @classmethod
            def validate_required_fields(cls, cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingCluster(NeuronCluster):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingCluster(NeuronClusterConfig())

    def test_nested_neuron_validation_uses_replaceable_collaborator(self):
        class TrackingNeuronValidator(NeuronValidator):
            @classmethod
            def validate(cls, cfg):
                raise RuntimeError("substituted neuron validator was called")

        class TrackingClusterValidator(NeuronClusterValidator):
            NEURON_VALIDATOR = TrackingNeuronValidator

            @classmethod
            def validate_required_fields(cls, cfg):
                return None

            @classmethod
            def validate_field_types(cls, cfg):
                return None

        model = SimpleNamespace(
            cfg=NeuronClusterConfig(neuron_config=NeuronConfig())
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted neuron validator was called",
        ):
            TrackingClusterValidator.validate(model)

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(NeuronClusterValidator):
            @staticmethod
            def validate_forward_input(input):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingCluster(NeuronCluster):
            VALIDATOR = RejectingValidator

        model = RejectingCluster.__new__(RejectingCluster)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()

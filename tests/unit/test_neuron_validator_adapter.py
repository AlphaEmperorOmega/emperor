import unittest

import torch

from emperor.neuron import Neuron, NeuronConfig
from emperor.neuron._validation import NeuronValidator
from unit.test_neuron import NeuronTestCase


class TestNeuronValidatorAdapter(NeuronTestCase):
    def test_module_declares_its_validator_adapter(self):
        self.assertIs(Neuron.VALIDATOR, NeuronValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(NeuronValidator):
            @classmethod
            def validate_required_fields(cls, cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingNeuron(Neuron):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingNeuron(NeuronConfig())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(NeuronValidator):
            @classmethod
            def validate_forward_input(cls, input):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingNeuron(Neuron):
            VALIDATOR = RejectingValidator

        model = RejectingNeuron.__new__(RejectingNeuron)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model.process_signal(torch.ones(1, 3))

    def test_axons_validation_uses_replaceable_collaborator(self):
        class TrackingAxonsValidator:
            @classmethod
            def validate_config(cls, cfg):
                raise RuntimeError("substituted Axons validator was called")

        class TrackingValidator(NeuronValidator):
            AXONS_VALIDATOR = TrackingAxonsValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted Axons validator was called",
        ):
            TrackingValidator.validate_axons_memory_dimensions(self.neuron_config())

    def test_terminal_validation_uses_replaceable_collaborator(self):
        class TrackingTerminalValidator:
            @classmethod
            def validate_config_composition(cls, cfg):
                raise RuntimeError("substituted Terminal validator was called")

        class TrackingValidator(NeuronValidator):
            TERMINAL_VALIDATOR = TrackingTerminalValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted Terminal validator was called",
        ):
            TrackingValidator.validate_terminal_composition(self.neuron_config())


if __name__ == "__main__":
    unittest.main()

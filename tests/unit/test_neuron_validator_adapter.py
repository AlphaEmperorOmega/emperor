import unittest

import torch
from emperor.neuron.core._validator import NeuronValidator
from emperor.neuron.core.config import NeuronConfig
from emperor.neuron.core.layers import Neuron


class TestNeuronValidatorAdapter(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

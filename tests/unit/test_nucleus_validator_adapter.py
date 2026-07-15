import unittest

import torch
from emperor.neuron.core._validator import NucleusValidator
from emperor.neuron.core.config import NucleusConfig
from emperor.neuron.core.layers import Nucleus


class TestNucleusValidatorAdapter(unittest.TestCase):
    def test_module_declares_its_validator_adapter(self):
        self.assertIs(Nucleus.VALIDATOR, NucleusValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(NucleusValidator):
            @staticmethod
            def validate_required_fields(cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingNucleus(Nucleus):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingNucleus(NucleusConfig())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(NucleusValidator):
            @classmethod
            def validate_forward_input(cls, input):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingNucleus(Nucleus):
            VALIDATOR = RejectingValidator

        model = RejectingNucleus.__new__(RejectingNucleus)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()

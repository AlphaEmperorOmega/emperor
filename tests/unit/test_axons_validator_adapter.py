import unittest

import torch
from emperor.neuron.core._validator import AxonsValidator
from emperor.neuron.core.config import AxonsConfig
from emperor.neuron.core.layers import Axons


class TestAxonsValidatorAdapter(unittest.TestCase):
    def test_module_declares_its_validator_adapter(self):
        self.assertIs(Axons.VALIDATOR, AxonsValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(AxonsValidator):
            @staticmethod
            def validate_memory_config(memory_config):
                raise RuntimeError("substituted construction validator was called")

        class TrackingAxons(Axons):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingAxons(AxonsConfig())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(AxonsValidator):
            @classmethod
            def validate_forward_input(cls, input):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingAxons(Axons):
            VALIDATOR = RejectingValidator

        model = RejectingAxons.__new__(RejectingAxons)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()

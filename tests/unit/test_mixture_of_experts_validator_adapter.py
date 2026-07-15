import unittest

import torch
from emperor.experts.core._validator import MixtureOfExpertsValidator
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core.layers import (
    MixtureOfExperts,
    MixtureOfExpertsMap,
    MixtureOfExpertsReduce,
)


class TestMixtureOfExpertsValidatorAdapter(unittest.TestCase):
    def test_expert_modules_share_the_base_owner_adapter(self):
        module_types = (
            MixtureOfExperts,
            MixtureOfExpertsMap,
            MixtureOfExpertsReduce,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, MixtureOfExpertsValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(MixtureOfExpertsValidator):
            @staticmethod
            def validate_required_fields(cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingExperts(MixtureOfExperts):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingExperts(MixtureOfExpertsConfig())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(MixtureOfExpertsValidator):
            @classmethod
            def validate_forward_inputs(
                cls,
                model,
                input_batch,
                probabilities,
                indices,
            ):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingExperts(MixtureOfExperts):
            VALIDATOR = RejectingValidator

        model = RejectingExperts.__new__(RejectingExperts)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()

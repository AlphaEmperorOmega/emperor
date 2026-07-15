import unittest

import torch
from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicBiasValidator,
)
from emperor.augmentations.adaptive_parameters.core.bias import (
    AdditiveDynamicBias,
    AffineTransformDynamicBias,
    DynamicBiasAbstract,
    DynamicBiasConfig,
    GeneratorDynamicBias,
    MultiplicativeDynamicBias,
    SigmoidGatedDynamicBias,
    TanhGatedDynamicBias,
    WeightedBankDynamicBias,
)


class TestDynamicBiasValidatorAdapter(unittest.TestCase):
    def test_bias_modules_share_the_base_owner_adapter(self):
        module_types = (
            DynamicBiasAbstract,
            AdditiveDynamicBias,
            AffineTransformDynamicBias,
            MultiplicativeDynamicBias,
            SigmoidGatedDynamicBias,
            TanhGatedDynamicBias,
            GeneratorDynamicBias,
            WeightedBankDynamicBias,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, DynamicBiasValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(DynamicBiasValidator):
            @staticmethod
            def validate_required_fields(cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingBias(DynamicBiasAbstract):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingBias(DynamicBiasConfig())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(DynamicBiasValidator):
            @staticmethod
            def ensure_parameters_exist(bias_params):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingBias(AdditiveDynamicBias):
            VALIDATOR = RejectingValidator

        model = RejectingBias.__new__(RejectingBias)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(None, torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()

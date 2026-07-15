import unittest

from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicWeightValidator,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeight,
    DynamicWeightAbstract,
    DynamicWeightConfig,
    HypernetworkDynamicWeight,
    LayeredWeightedBankDynamicWeight,
    LowRankDynamicWeight,
    SingleModelDynamicWeight,
    SoftWeightedBankDynamicWeight,
)


class TestDynamicWeightValidatorAdapter(unittest.TestCase):
    def test_weight_modules_share_the_base_owner_adapter(self):
        module_types = (
            DynamicWeightAbstract,
            SingleModelDynamicWeight,
            DualModelDynamicWeight,
            LowRankDynamicWeight,
            HypernetworkDynamicWeight,
            LayeredWeightedBankDynamicWeight,
            SoftWeightedBankDynamicWeight,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, DynamicWeightValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(DynamicWeightValidator):
            @staticmethod
            def validate_required_fields(cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingWeight(DynamicWeightAbstract):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingWeight(DynamicWeightConfig())


if __name__ == "__main__":
    unittest.main()

import unittest

from emperor.augmentations.adaptive_parameters import DynamicWeightConfig
from emperor.augmentations.adaptive_parameters._weights.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters._weights.validation import (
    DynamicWeightValidator,
)
from emperor.augmentations.adaptive_parameters._weights.variants.dual_model import (
    DualModelDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.hypernetwork import (
    HypernetworkDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.layered_weighted_bank import (
    LayeredWeightedBankDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.low_rank import (
    LowRankDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.single_model import (
    SingleModelDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.soft_weighted_bank import (
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

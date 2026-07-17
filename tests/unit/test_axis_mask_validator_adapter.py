import unittest

from emperor.augmentations.adaptive_parameters import AxisMaskConfig
from emperor.augmentations.adaptive_parameters._masks.base import AxisMaskAbstract
from emperor.augmentations.adaptive_parameters._masks.validation import (
    AxisMaskValidator,
)
from emperor.augmentations.adaptive_parameters._masks.variants.diagonal import (
    DiagonalAxisMask,
)
from emperor.augmentations.adaptive_parameters._masks.variants.outer_product import (
    OuterProductMask,
)
from emperor.augmentations.adaptive_parameters._masks.variants.per_axis import (
    PerAxisScoreMask,
)
from emperor.augmentations.adaptive_parameters._masks.variants.top_slice import (
    TopSliceAxisMask,
)
from emperor.augmentations.adaptive_parameters._masks.variants.weight_informed import (
    WeightInformedScoreAxisMask,
)


class TestAxisMaskValidatorAdapter(unittest.TestCase):
    def test_mask_modules_share_the_base_owner_adapter(self):
        module_types = (
            AxisMaskAbstract,
            WeightInformedScoreAxisMask,
            PerAxisScoreMask,
            TopSliceAxisMask,
            OuterProductMask,
            DiagonalAxisMask,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, AxisMaskValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(AxisMaskValidator):
            @staticmethod
            def validate_required_fields(cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingMask(AxisMaskAbstract):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingMask(AxisMaskConfig())


if __name__ == "__main__":
    unittest.main()

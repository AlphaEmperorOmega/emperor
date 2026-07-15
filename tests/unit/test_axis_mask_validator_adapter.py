import unittest

from emperor.augmentations.adaptive_parameters.core._validator import AxisMaskValidator
from emperor.augmentations.adaptive_parameters.core.mask import (
    AxisMaskAbstract,
    AxisMaskConfig,
    DiagonalAxisMask,
    OuterProductMask,
    PerAxisScoreMask,
    TopSliceAxisMask,
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

import unittest

from emperor.augmentations.adaptive_parameters import DynamicDiagonalConfig
from emperor.augmentations.adaptive_parameters._diagonals.base import (
    DynamicDiagonalAbstract,
)
from emperor.augmentations.adaptive_parameters._diagonals.validation import (
    DynamicDiagonalValidator,
)
from emperor.augmentations.adaptive_parameters._diagonals.variants.anti import (
    AntiDynamicDiagonal,
)
from emperor.augmentations.adaptive_parameters._diagonals.variants.combined import (
    CombinedDynamicDiagonal,
)
from emperor.augmentations.adaptive_parameters._diagonals.variants.standard import (
    StandardDynamicDiagonal,
)


class TestDynamicDiagonalValidatorAdapter(unittest.TestCase):
    def test_diagonal_modules_share_the_base_owner_adapter(self):
        module_types = (
            DynamicDiagonalAbstract,
            StandardDynamicDiagonal,
            AntiDynamicDiagonal,
            CombinedDynamicDiagonal,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, DynamicDiagonalValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(DynamicDiagonalValidator):
            @staticmethod
            def validate_required_fields(cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingDiagonal(DynamicDiagonalAbstract):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingDiagonal(DynamicDiagonalConfig())


if __name__ == "__main__":
    unittest.main()

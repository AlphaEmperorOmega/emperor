import unittest

from emperor.layers import RowLayout
from emperor.layers._row_layout.validation import RowLayoutValidator


class TestRowLayoutValidatorAdapter(unittest.TestCase):
    def test_value_exposes_validator_adapter(self):
        self.assertIs(RowLayout.VALIDATOR, RowLayoutValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(RowLayoutValidator):
            @staticmethod
            def _validate_leading_shape(leading_shape):
                raise RuntimeError("substituted row-layout validator was called")

        class TrackingRowLayout(RowLayout):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted row-layout validator was called",
        ):
            TrackingRowLayout(
                leading_shape=(3,),
                context_sharing_restricted=False,
            )


if __name__ == "__main__":
    unittest.main()

import unittest

from emperor.augmentations.adaptive_parameters import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)


class TestAdaptiveParameterOptions(unittest.TestCase):
    def test_public_option_names_and_serialized_values_are_stable(self):
        expected_members = {
            WeightNormalizationOptions: (
                ("DISABLED", 0),
                ("CLAMP", 1),
                ("L2_SCALE", 2),
                ("SOFT_CLAMP", 3),
                ("RMS", 4),
                ("SIGMOID_SCALE", 5),
            ),
            DynamicDepthOptions: (
                ("DISABLED", 0),
                ("DEPTH_OF_ONE", 1),
                ("DEPTH_OF_TWO", 2),
                ("DEPTH_OF_THREE", 3),
                ("DEPTH_OF_FOUR", 4),
                ("DEPTH_OF_FIVE", 5),
                ("DEPTH_OF_SIX", 6),
                ("DEPTH_OF_SEVEN", 7),
                ("DEPTH_OF_EIGHT", 8),
                ("DEPTH_OF_NINE", 9),
                ("DEPTH_OF_TEN", 10),
            ),
            BankExpansionFactorOptions: (
                ("DISABLED", 0),
                ("FACTOR_OF_ONE", 1),
                ("FACTOR_OF_TWO", 2),
                ("FACTOR_OF_THREE", 3),
                ("FACTOR_OF_FOUR", 4),
            ),
            MaskDimensionOptions: (
                ("ROW", 0),
                ("COLUMN", 1),
            ),
            WeightNormalizationPositionOptions: (
                ("DISABLED", 0),
                ("BEFORE_OUTER_PRODUCT", 1),
                ("AFTER_OUTER_PRODUCT", 2),
            ),
            WeightDecayScheduleOptions: (
                ("DISABLED", 0),
                ("EXPONENTIAL", 1),
                ("LINEAR", 2),
                ("MULTIPLICATIVE", 3),
            ),
        }

        for option_type, expected in expected_members.items():
            with self.subTest(option_type=option_type.__name__):
                actual = tuple((option.name, option.value) for option in option_type)
                self.assertTupleEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

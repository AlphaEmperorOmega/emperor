import unittest

import torch

from emperor.neuron._validation import NeuronValidationMixin


class TestNeuronValidationMixin(unittest.TestCase):
    def test_positive_integer_orchestration_dispatches_through_subclass(self):
        validated_values = []

        class TrackingMixin(NeuronValidationMixin):
            @staticmethod
            def validate_integer(name, value):
                validated_values.append((name, value))

        TrackingMixin.validate_positive_integer("count", 2)

        self.assertEqual(validated_values, [("count", 2)])

    def test_non_positive_integer_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            r"count must be a positive integer, received 0\.",
        ):
            NeuronValidationMixin.validate_positive_integer("count", 0)

    def test_config_integer_and_tensor_type_errors_are_precise(self) -> None:
        invalid_cases = (
            (
                lambda: NeuronValidationMixin.validate_integer("count", True),
                TypeError,
                "count must be an integer, received bool",
            ),
            (
                lambda: NeuronValidationMixin.validate_tensor_rank("signal", [1.0], 2),
                TypeError,
                "signal must be a Tensor, received list",
            ),
        )
        for invoke, error_type, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    invoke()

        NeuronValidationMixin.validate_integer("count", 2)
        NeuronValidationMixin.validate_tensor_rank("signal", torch.ones(1, 2), 2)


if __name__ == "__main__":
    unittest.main()

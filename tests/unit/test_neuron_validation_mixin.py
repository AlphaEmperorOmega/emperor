import unittest

from emperor.neuron.core._validator import NeuronValidationMixin


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


if __name__ == "__main__":
    unittest.main()

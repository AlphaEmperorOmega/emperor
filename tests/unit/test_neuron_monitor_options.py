import unittest

from emperor.neuron import NeuronClusterMonitorCallback


class TestNeuronMonitorOptions(unittest.TestCase):
    def test_cadence_and_history_require_integers(self) -> None:
        for option_name in ("log_every_n_steps", "history_size"):
            for invalid_value in (True, False, 1.0, "1"):
                with self.subTest(
                    option_name=option_name,
                    invalid_value=invalid_value,
                ):
                    with self.assertRaisesRegex(
                        TypeError,
                        rf"^{option_name} must be a positive integer",
                    ):
                        NeuronClusterMonitorCallback(
                            **{option_name: invalid_value}
                        )


if __name__ == "__main__":
    unittest.main()

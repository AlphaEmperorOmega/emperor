import unittest
from types import SimpleNamespace

from emperor.neuron._terminal.connection_topology import (
    TargetCoordinateBuilder,
)


class TestNeuronTerminalTopologyErrors(unittest.TestCase):
    def invalid_shape_builder(self) -> TargetCoordinateBuilder:
        return TargetCoordinateBuilder(
            SimpleNamespace(
                connection_shape="invalid",
                x_axis_position=0,
                y_axis_position=0,
                z_axis_position=0,
                xy_axis_range=SimpleNamespace(value=1),
                z_axis_range=SimpleNamespace(value=1),
            )
        )

    def test_unsupported_connection_shape_names_terminal_contract(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "^Unsupported terminal connection shape: 'invalid'$",
        ):
            self.invalid_shape_builder().build()

    def test_unsupported_connection_shape_cannot_report_a_count(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "^Unsupported terminal connection shape: 'invalid'$",
        ):
            self.invalid_shape_builder().get_total_neuron_connections()


if __name__ == "__main__":
    unittest.main()

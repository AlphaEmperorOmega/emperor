import unittest

from emperor.neuron import (
    TerminalConnectionShapeOptions,
    TerminalRangeOptions,
    TerminalZAxisOffsetOptions,
)
from unit.test_neuron import NeuronTestCase


class TestNeuronTerminalTopology(NeuronTestCase):
    def test_shifted_ellipsoid_matches_exact_integer_cross_sections(self) -> None:
        terminal = self.shaped_terminal(
            TerminalConnectionShapeOptions.SPHERE,
            num_experts=33,
            xy_axis_range=TerminalRangeOptions.TWO,
            z_axis_range=TerminalRangeOptions.FOUR,
            z_axis_offset=TerminalZAxisOffsetOptions.ONE,
        )
        three_by_three_plane = {
            (x_coordinate, y_coordinate)
            for x_coordinate in range(3)
            for y_coordinate in range(3)
        }
        radius_two_integer_disc = {
            (1, 1),
            (0, 1),
            (2, 1),
            (1, 0),
            (1, 2),
            (-1, 1),
            (3, 1),
            (1, -1),
            (1, 3),
            (0, 0),
            (0, 2),
            (2, 0),
            (2, 2),
        }
        expected_cross_sections = {
            0: {(1, 1)},
            1: three_by_three_plane,
            2: radius_two_integer_disc,
            3: three_by_three_plane,
            4: {(1, 1)},
        }
        expected_connections = {
            (x_coordinate, y_coordinate, z_coordinate)
            for z_coordinate, cross_section in expected_cross_sections.items()
            for x_coordinate, y_coordinate in cross_section
        }

        actual_connections = {
            tuple(connection) for connection in terminal.neuron_connections.tolist()
        }

        self.assertEqual(terminal.total_neuron_connections, 33)
        self.assertEqual(actual_connections, expected_connections)


if __name__ == "__main__":
    unittest.main()

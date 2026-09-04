import unittest
from math import gcd

from emperor.neuron import (
    TerminalConnectionShapeOptions,
    TerminalRangeOptions,
)
from unit.test_neuron import NeuronTestCase


class TestNeuronTerminalTopology(NeuronTestCase):
    def test_diagonal_shape_counts_match_every_supported_range_pair(self) -> None:
        for xy_axis_range in TerminalRangeOptions:
            for z_axis_range in TerminalRangeOptions:
                shared_axis_range_divisor = gcd(
                    xy_axis_range.value,
                    z_axis_range.value,
                )
                shape_connection_counts = (
                    (
                        TerminalConnectionShapeOptions.DIAGONAL,
                        4 * xy_axis_range.value + 4 * shared_axis_range_divisor + 1,
                    ),
                    (
                        TerminalConnectionShapeOptions.CROSS_DIAGONAL,
                        8 * xy_axis_range.value
                        + 2 * z_axis_range.value
                        + 4 * shared_axis_range_divisor
                        + 1,
                    ),
                )
                for connection_shape, expected_connection_count in (
                    shape_connection_counts
                ):
                    with self.subTest(
                        connection_shape=connection_shape,
                        xy_axis_range=xy_axis_range,
                        z_axis_range=z_axis_range,
                    ):
                        terminal = self.shaped_terminal(
                            connection_shape,
                            num_experts=expected_connection_count,
                            xy_axis_range=xy_axis_range,
                            z_axis_range=z_axis_range,
                        )
                        unique_connections = {
                            tuple(connection)
                            for connection in terminal.neuron_connections.tolist()
                        }

                        self.assertEqual(
                            terminal.total_neuron_connections,
                            expected_connection_count,
                        )
                        self.assertEqual(
                            terminal.total_neuron_connections,
                            len(unique_connections),
                        )

    def test_centered_ellipsoid_matches_exact_integer_cross_sections(self) -> None:
        terminal = self.shaped_terminal(
            TerminalConnectionShapeOptions.SPHERE,
            num_experts=33,
            xy_axis_range=TerminalRangeOptions.TWO,
            z_axis_range=TerminalRangeOptions.TWO,
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
            -1: {(1, 1)},
            0: three_by_three_plane,
            1: radius_two_integer_disc,
            2: three_by_three_plane,
            3: {(1, 1)},
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
        self.assertEqual(terminal.neuron_connections.tolist().count([1, 1, 1]), 1)
        for x_coordinate, y_coordinate, z_coordinate in actual_connections:
            mirrored_z_coordinate = 2 - z_coordinate
            self.assertIn(
                (x_coordinate, y_coordinate, mirrored_z_coordinate),
                actual_connections,
            )


if __name__ == "__main__":
    unittest.main()

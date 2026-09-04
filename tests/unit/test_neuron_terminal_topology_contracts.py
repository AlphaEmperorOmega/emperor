import unittest
from math import gcd

import torch

from emperor.neuron import (
    TerminalConnectionShapeOptions,
    TerminalRangeOptions,
)
from emperor.neuron._terminal.connection_topology import (
    TargetCoordinateBuilder,
)
from unit.test_neuron import NeuronTestCase


class TestNeuronTerminalTopology(NeuronTestCase):
    def test_builder_snapshots_topology_values_from_config(self) -> None:
        terminal_config = self.terminal_config()
        builder = TargetCoordinateBuilder(terminal_config)
        total_neuron_connections = builder.get_total_neuron_connections()
        expected_connections = builder.build()

        self.assertIs(builder.connection_shape, terminal_config.connection_shape)
        self.assertEqual(builder.x_axis_position, terminal_config.x_axis_position)
        self.assertEqual(builder.y_axis_position, terminal_config.y_axis_position)
        self.assertEqual(builder.z_axis_position, terminal_config.z_axis_position)
        self.assertEqual(builder.xy_axis_range, terminal_config.xy_axis_range.value)
        self.assertEqual(builder.z_axis_range, terminal_config.z_axis_range.value)
        self.assertEqual(total_neuron_connections, 27)

        terminal_config.connection_shape = TerminalConnectionShapeOptions.CROSS
        terminal_config.x_axis_position = 10
        terminal_config.y_axis_position = 20
        terminal_config.z_axis_position = 30
        terminal_config.xy_axis_range = TerminalRangeOptions.TWO
        terminal_config.z_axis_range = TerminalRangeOptions.FOUR

        torch.testing.assert_close(builder.build(), expected_connections)

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
                for (
                    connection_shape,
                    expected_connection_count,
                ) in shape_connection_counts:
                    with self.subTest(
                        connection_shape=connection_shape,
                        xy_axis_range=xy_axis_range,
                        z_axis_range=z_axis_range,
                    ):
                        terminal_config = self.terminal_config(
                            connection_shape=connection_shape,
                            xy_axis_range=xy_axis_range,
                            z_axis_range=z_axis_range,
                        )
                        connections = TargetCoordinateBuilder(terminal_config).build()
                        unique_connections = {
                            tuple(connection) for connection in connections.tolist()
                        }

                        self.assertEqual(
                            len(connections),
                            expected_connection_count,
                        )
                        self.assertEqual(len(connections), len(unique_connections))

    def test_connection_count_is_computed_without_building_coordinates(self) -> None:
        shape_connection_counts = (
            (TerminalConnectionShapeOptions.BOX, 27),
            (TerminalConnectionShapeOptions.CROSS, 7),
            (TerminalConnectionShapeOptions.SPHERE, 7),
            (TerminalConnectionShapeOptions.DIAGONAL, 9),
            (TerminalConnectionShapeOptions.CROSS_DIAGONAL, 15),
        )

        for connection_shape, expected_connection_count in shape_connection_counts:
            with self.subTest(connection_shape=connection_shape):
                terminal_config = self.terminal_config(
                    connection_shape=connection_shape,
                )
                builder = TargetCoordinateBuilder(terminal_config)

                self.assertEqual(
                    builder.get_total_neuron_connections(),
                    expected_connection_count,
                )
                self.assertEqual(len(builder.build()), expected_connection_count)

    def test_centered_ellipsoid_matches_exact_integer_cross_sections(self) -> None:
        terminal_config = self.terminal_config(
            connection_shape=TerminalConnectionShapeOptions.SPHERE,
            sampler_config=self.sampler_config(num_experts=33),
            xy_axis_range=TerminalRangeOptions.TWO,
            z_axis_range=TerminalRangeOptions.TWO,
        )
        builder = TargetCoordinateBuilder(terminal_config)
        connections = builder.build()
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

        actual_connections = {tuple(connection) for connection in connections.tolist()}

        self.assertEqual(builder.get_total_neuron_connections(), 33)
        self.assertEqual(connections.shape, (33, 3))
        self.assertEqual(actual_connections, expected_connections)
        self.assertEqual(connections.tolist().count([1, 1, 1]), 1)
        for x_coordinate, y_coordinate, z_coordinate in actual_connections:
            mirrored_z_coordinate = 2 - z_coordinate
            self.assertIn(
                (x_coordinate, y_coordinate, mirrored_z_coordinate),
                actual_connections,
            )


if __name__ == "__main__":
    unittest.main()

from collections.abc import Iterator
from math import gcd
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.neuron._options import TerminalConnectionShapeOptions

if TYPE_CHECKING:
    from emperor.neuron._config import TerminalConfig


class TargetCoordinateBuilder:
    """Build the ordered absolute target coordinates for one Terminal."""

    def __init__(self, cfg: "TerminalConfig") -> None:
        self.connection_shape: TerminalConnectionShapeOptions = cfg.connection_shape
        self.x_axis_position: int = cfg.x_axis_position
        self.y_axis_position: int = cfg.y_axis_position
        self.z_axis_position: int = cfg.z_axis_position
        self.xy_axis_range: int = cfg.xy_axis_range.value
        self.z_axis_range: int = cfg.z_axis_range.value

    def build(self) -> Tensor:
        if self.connection_shape is TerminalConnectionShapeOptions.BOX:
            return torch.cartesian_prod(
                self.__x_axis_coordinates(),
                self.__y_axis_coordinates(),
                self.__z_axis_coordinates(),
            )

        terminal_position = self.__terminal_position()
        deduplicated_connection_offset_matrix = (
            self.__deduplicated_connection_offset_matrix()
        )
        return terminal_position + deduplicated_connection_offset_matrix

    def get_total_neuron_connections(self) -> int:
        """Return the topology size without building target coordinates."""
        if self.connection_shape is TerminalConnectionShapeOptions.BOX:
            xy_axis_width = 2 * self.xy_axis_range + 1
            z_axis_width = 2 * self.z_axis_range + 1
            return xy_axis_width**2 * z_axis_width
        if self.connection_shape is TerminalConnectionShapeOptions.CROSS:
            return 4 * self.xy_axis_range + 2 * self.z_axis_range + 1
        if self.connection_shape is TerminalConnectionShapeOptions.SPHERE:
            return self.__ellipsoid_connection_count()

        shared_yz_axis_steps = gcd(self.xy_axis_range, self.z_axis_range)
        if self.connection_shape is TerminalConnectionShapeOptions.DIAGONAL:
            return 4 * self.xy_axis_range + 4 * shared_yz_axis_steps + 1
        if self.connection_shape is TerminalConnectionShapeOptions.CROSS_DIAGONAL:
            return (
                8 * self.xy_axis_range
                + 2 * self.z_axis_range
                + 4 * shared_yz_axis_steps
                + 1
            )
        raise ValueError(
            f"Unsupported terminal connection shape: {self.connection_shape!r}"
        )

    def __x_axis_coordinates(self) -> Tensor:
        return torch.arange(
            self.x_axis_position - self.xy_axis_range,
            self.x_axis_position + self.xy_axis_range + 1,
        )

    def __y_axis_coordinates(self) -> Tensor:
        return torch.arange(
            self.y_axis_position - self.xy_axis_range,
            self.y_axis_position + self.xy_axis_range + 1,
        )

    def __z_axis_coordinates(self) -> Tensor:
        return torch.arange(
            self.z_axis_position - self.z_axis_range,
            self.z_axis_position + self.z_axis_range + 1,
        )

    def __terminal_position(self) -> Tensor:
        terminal_position_coordinates = [
            self.x_axis_position,
            self.y_axis_position,
            self.z_axis_position,
        ]
        return torch.tensor(
            terminal_position_coordinates,
            dtype=torch.long,
        )

    def __deduplicated_connection_offset_matrix(self) -> Tensor:
        connection_offsets = self.__connection_offsets()
        unique_connection_offsets_in_order = dict.fromkeys(connection_offsets)
        deduplicated_connection_offsets = list(unique_connection_offsets_in_order)
        return torch.tensor(
            deduplicated_connection_offsets,
            dtype=torch.long,
        )

    def __connection_offsets(self) -> list[tuple[int, int, int]]:
        if self.connection_shape is TerminalConnectionShapeOptions.CROSS:
            return self.__cross_offsets()
        if self.connection_shape is TerminalConnectionShapeOptions.SPHERE:
            return self.__ellipsoid_offsets()
        if self.connection_shape is TerminalConnectionShapeOptions.DIAGONAL:
            return self.__diagonal_offsets()
        if self.connection_shape is TerminalConnectionShapeOptions.CROSS_DIAGONAL:
            return self.__cross_diagonal_offsets()
        raise ValueError(
            f"Unsupported terminal connection shape: {self.connection_shape!r}"
        )

    def __cross_offsets(self) -> list[tuple[int, int, int]]:
        x_axis_line_offsets = self.__x_axis_line_offsets()
        y_axis_line_offsets = self.__y_axis_line_offsets()
        z_axis_line_offsets = self.__z_axis_line_offsets()
        return x_axis_line_offsets + y_axis_line_offsets + z_axis_line_offsets

    def __x_axis_line_offsets(self) -> list[tuple[int, int, int]]:
        first_x_axis_offset = -self.xy_axis_range
        x_axis_offset_after_last = self.xy_axis_range + 1
        return [
            (delta, 0, 0)
            for delta in range(
                first_x_axis_offset,
                x_axis_offset_after_last,
            )
        ]

    def __y_axis_line_offsets(self) -> list[tuple[int, int, int]]:
        first_y_axis_offset = -self.xy_axis_range
        y_axis_offset_after_last = self.xy_axis_range + 1
        return [
            (0, delta, 0)
            for delta in range(
                first_y_axis_offset,
                y_axis_offset_after_last,
            )
        ]

    def __z_axis_line_offsets(self) -> list[tuple[int, int, int]]:
        first_z_axis_offset = -self.z_axis_range
        z_axis_offset_after_last = self.z_axis_range + 1
        return [
            (0, 0, delta)
            for delta in range(
                first_z_axis_offset,
                z_axis_offset_after_last,
            )
        ]

    def __ellipsoid_offsets(self) -> list[tuple[int, int, int]]:
        return list(self.__iter_ellipsoid_offsets())

    def __ellipsoid_connection_count(self) -> int:
        return sum(1 for _ in self.__iter_ellipsoid_offsets())

    def __iter_ellipsoid_offsets(self) -> Iterator[tuple[int, int, int]]:
        for x_delta in range(-self.xy_axis_range, self.xy_axis_range + 1):
            for y_delta in range(-self.xy_axis_range, self.xy_axis_range + 1):
                for z_delta in range(
                    -self.z_axis_range,
                    self.z_axis_range + 1,
                ):
                    normalized_squared_distance = self.__normalized_squared_distance(
                        x_delta=x_delta,
                        y_delta=y_delta,
                        z_delta=z_delta,
                    )
                    if normalized_squared_distance <= 1.0 + 1e-9:
                        yield x_delta, y_delta, z_delta

    def __normalized_squared_distance(
        self,
        *,
        x_delta: int,
        y_delta: int,
        z_delta: int,
    ) -> float:
        normalized_x_axis_distance_squared = (x_delta / self.xy_axis_range) ** 2
        normalized_y_axis_distance_squared = (y_delta / self.xy_axis_range) ** 2
        normalized_z_axis_distance_squared = (z_delta / self.z_axis_range) ** 2
        normalized_squared_distance = (
            normalized_x_axis_distance_squared
            + normalized_y_axis_distance_squared
            + normalized_z_axis_distance_squared
        )
        return normalized_squared_distance

    def __diagonal_offsets(self) -> list[tuple[int, int, int]]:
        xy_diagonal_offsets = self.__xy_diagonal_offsets()
        yz_diagonal_offsets = self.__yz_diagonal_offsets()
        return xy_diagonal_offsets + yz_diagonal_offsets

    def __xy_diagonal_offsets(self) -> list[tuple[int, int, int]]:
        first_xy_diagonal_delta = -self.xy_axis_range
        xy_diagonal_delta_after_last = self.xy_axis_range + 1
        diagonal_offsets = []
        for delta in range(
            first_xy_diagonal_delta,
            xy_diagonal_delta_after_last,
        ):
            diagonal_offsets.append((delta, delta, 0))
            diagonal_offsets.append((delta, -delta, 0))
        return diagonal_offsets

    def __yz_diagonal_offsets(self) -> list[tuple[int, int, int]]:
        steps_from_center_to_yz_extent = gcd(
            self.xy_axis_range,
            self.z_axis_range,
        )
        y_axis_delta_per_step = self.xy_axis_range // steps_from_center_to_yz_extent
        z_axis_delta_per_step = self.z_axis_range // steps_from_center_to_yz_extent
        first_yz_diagonal_step = -steps_from_center_to_yz_extent
        yz_diagonal_step_after_last = steps_from_center_to_yz_extent + 1

        diagonal_offsets = []
        for step_multiplier in range(
            first_yz_diagonal_step,
            yz_diagonal_step_after_last,
        ):
            y_delta = step_multiplier * y_axis_delta_per_step
            z_delta = step_multiplier * z_axis_delta_per_step
            diagonal_offsets.append((0, y_delta, z_delta))
            diagonal_offsets.append((0, y_delta, -z_delta))
        return diagonal_offsets

    def __cross_diagonal_offsets(self) -> list[tuple[int, int, int]]:
        cross_offsets = self.__cross_offsets()
        diagonal_offsets = self.__diagonal_offsets()
        return cross_offsets + diagonal_offsets

import torch
from torch import Tensor

from emperor.neuron._options import TerminalConnectionShapeOptions


def initialize_terminal_connections(cfg) -> Tensor:
    connection_shape = (
        TerminalConnectionShapeOptions.BOX
        if cfg.connection_shape is None
        else cfg.connection_shape
    )
    if connection_shape is TerminalConnectionShapeOptions.BOX:
        return torch.cartesian_prod(
            _axis_range(cfg.x_axis_position, cfg.xy_axis_range.value),
            _axis_range(cfg.y_axis_position, cfg.xy_axis_range.value),
            _z_axis_range(cfg),
        )

    deduplicated_offsets = list(
        dict.fromkeys(_connection_offsets(cfg, connection_shape))
    )
    terminal_position = torch.tensor(
        [cfg.x_axis_position, cfg.y_axis_position, cfg.z_axis_position],
        dtype=torch.long,
    )
    return terminal_position + torch.tensor(deduplicated_offsets, dtype=torch.long)


def _axis_range(position: int, axis_range: int) -> Tensor:
    return torch.arange(position - axis_range, position + axis_range + 1)


def _z_axis_range(cfg) -> Tensor:
    return torch.arange(
        cfg.z_axis_position - cfg.z_axis_offset.value,
        cfg.z_axis_position + cfg.z_axis_range.value - cfg.z_axis_offset.value + 1,
    )


def _connection_offsets(
    cfg,
    connection_shape: TerminalConnectionShapeOptions,
) -> list[tuple[int, int, int]]:
    if connection_shape is TerminalConnectionShapeOptions.CROSS:
        return (
            _x_axis_line_offsets(cfg)
            + _y_axis_line_offsets(cfg)
            + _z_axis_line_offsets(cfg)
        )
    if connection_shape is TerminalConnectionShapeOptions.SPHERE:
        return _ellipsoid_offsets(cfg)
    if connection_shape is TerminalConnectionShapeOptions.DIAGONAL_X:
        return _xy_diagonal_offsets(cfg)
    if connection_shape is TerminalConnectionShapeOptions.LINE_LEFT_RIGHT:
        return _x_axis_line_offsets(cfg)
    if connection_shape is TerminalConnectionShapeOptions.LINE_UP_DOWN:
        return _y_axis_line_offsets(cfg)
    if connection_shape is TerminalConnectionShapeOptions.LINE_FRONT_BACK:
        return _z_axis_line_offsets(cfg)
    raise ValueError(f"Unsupported connection_shape {connection_shape!r} for Terminal.")


def _x_axis_line_offsets(cfg) -> list[tuple[int, int, int]]:
    return [
        (delta, 0, 0)
        for delta in range(-cfg.xy_axis_range.value, cfg.xy_axis_range.value + 1)
    ]


def _y_axis_line_offsets(cfg) -> list[tuple[int, int, int]]:
    return [
        (0, delta, 0)
        for delta in range(-cfg.xy_axis_range.value, cfg.xy_axis_range.value + 1)
    ]


def _z_axis_line_offsets(cfg) -> list[tuple[int, int, int]]:
    return [
        (0, 0, delta)
        for delta in range(
            -cfg.z_axis_offset.value,
            cfg.z_axis_range.value - cfg.z_axis_offset.value + 1,
        )
    ]


def _xy_diagonal_offsets(cfg) -> list[tuple[int, int, int]]:
    diagonal_offsets = []
    for delta in range(-cfg.xy_axis_range.value, cfg.xy_axis_range.value + 1):
        diagonal_offsets.append((delta, delta, 0))
        diagonal_offsets.append((delta, -delta, 0))
    return diagonal_offsets


def _ellipsoid_offsets(cfg) -> list[tuple[int, int, int]]:
    xy_axis_range = cfg.xy_axis_range.value
    z_axis_range = cfg.z_axis_range.value
    z_axis_offset = cfg.z_axis_offset.value
    z_window_center = (z_axis_range - 2 * z_axis_offset) / 2
    z_half_extent = z_axis_range / 2
    ellipsoid_offsets = []
    for x_delta in range(-xy_axis_range, xy_axis_range + 1):
        for y_delta in range(-xy_axis_range, xy_axis_range + 1):
            for z_delta in range(-z_axis_offset, z_axis_range - z_axis_offset + 1):
                normalized_squared_distance = (
                    (x_delta / xy_axis_range) ** 2
                    + (y_delta / xy_axis_range) ** 2
                    + ((z_delta - z_window_center) / z_half_extent) ** 2
                )
                if normalized_squared_distance <= 1.0 + 1e-9:
                    ellipsoid_offsets.append((x_delta, y_delta, z_delta))
    return ellipsoid_offsets

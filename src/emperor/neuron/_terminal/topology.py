from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import torch
from torch import Tensor

from emperor.neuron._options import TerminalConnectionShapeOptions

AxisBounds = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
AxisSubdivision = tuple[int, int, int]


@dataclass(frozen=True)
class TerminalRoutingTreeNodePlan:
    """Immutable spatial assignment for one routing-tree node."""

    path: tuple[int, ...]
    level: int
    bounds: AxisBounds
    connection_indices: tuple[int, ...]
    subdivision: AxisSubdivision
    children: tuple["TerminalRoutingTreeNodePlan", ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class TerminalRoutingTreePlan:
    """Compiled, translation-invariant partition of terminal connections."""

    depth: int
    direction_top_k: tuple[int, ...]
    leaf_top_k: int
    root: TerminalRoutingTreeNodePlan

    @property
    def output_width(self) -> int:
        width = self.leaf_top_k
        for level_top_k in self.direction_top_k:
            width *= level_top_k
        return width

    def walk(self):
        yield from self.root.walk()


def _compile_terminal_routing_tree(
    neuron_connections: Tensor,
    routing_tree_config,
    leaf_top_k: int,
) -> TerminalRoutingTreePlan:
    """Compile coordinates into a deterministic hierarchy of nonempty cuboids."""

    coordinate_rows = tuple(
        tuple(int(component) for component in row)
        for row in neuron_connections.detach().cpu().tolist()
    )
    if not coordinate_rows:
        raise ValueError("Terminal routing tree requires at least one connection.")

    bounds: AxisBounds = tuple(
        (
            min(row[axis] for row in coordinate_rows),
            max(row[axis] for row in coordinate_rows),
        )
        for axis in range(3)
    )  # type: ignore[assignment]
    depth = int(routing_tree_config.depth.value)
    root = _compile_routing_node(
        coordinate_rows=coordinate_rows,
        connection_indices=tuple(range(len(coordinate_rows))),
        bounds=bounds,
        path=(),
        level=0,
        depth=depth,
        direction_branch_counts=routing_tree_config.direction_branch_counts,
    )
    return TerminalRoutingTreePlan(
        depth=depth,
        direction_top_k=tuple(routing_tree_config.direction_top_k),
        leaf_top_k=leaf_top_k,
        root=root,
    )


def _compile_routing_node(
    *,
    coordinate_rows: tuple[tuple[int, int, int], ...],
    connection_indices: tuple[int, ...],
    bounds: AxisBounds,
    path: tuple[int, ...],
    level: int,
    depth: int,
    direction_branch_counts: tuple[int, ...],
) -> TerminalRoutingTreeNodePlan:
    if level == depth - 1:
        return TerminalRoutingTreeNodePlan(
            path=path,
            level=level,
            bounds=bounds,
            connection_indices=connection_indices,
            subdivision=(1, 1, 1),
        )

    subdivision = _balanced_axis_subdivision(
        bounds,
        direction_branch_counts[level],
    )
    child_bounds = _subdivide_bounds(bounds, subdivision)
    children = []
    for candidate_bounds in child_bounds:
        child_connection_indices = tuple(
            index
            for index in connection_indices
            if _coordinate_is_within(coordinate_rows[index], candidate_bounds)
        )
        if not child_connection_indices:
            continue
        child_index = len(children)
        children.append(
            _compile_routing_node(
                coordinate_rows=coordinate_rows,
                connection_indices=child_connection_indices,
                bounds=candidate_bounds,
                path=(*path, child_index),
                level=level + 1,
                depth=depth,
                direction_branch_counts=direction_branch_counts,
            )
        )

    return TerminalRoutingTreeNodePlan(
        path=path,
        level=level,
        bounds=bounds,
        connection_indices=connection_indices,
        subdivision=subdivision,
        children=tuple(children),
    )


def _balanced_axis_subdivision(
    bounds: AxisBounds,
    maximum_regions: int,
) -> AxisSubdivision:
    axis_spans = tuple(end - start + 1 for start, end in bounds)
    candidates = (
        subdivision
        for subdivision in product(
            range(1, axis_spans[0] + 1),
            range(1, axis_spans[1] + 1),
            range(1, axis_spans[2] + 1),
        )
        if subdivision[0] * subdivision[1] * subdivision[2] <= maximum_regions
    )

    def score(subdivision: AxisSubdivision):
        region_count = subdivision[0] * subdivision[1] * subdivision[2]
        cell_edges = tuple(
            Fraction(span, split)
            for span, split in zip(axis_spans, subdivision, strict=True)
        )
        aspect_ratio = max(cell_edges) / min(cell_edges)
        edge_spread = sum(
            (edge - sum(cell_edges, Fraction()) / 3) ** 2 for edge in cell_edges
        )
        return (
            -region_count,
            aspect_ratio,
            edge_spread,
            -subdivision[0],
            -subdivision[1],
            -subdivision[2],
        )

    return min(candidates, key=score)


def _subdivide_bounds(
    bounds: AxisBounds,
    subdivision: AxisSubdivision,
) -> tuple[AxisBounds, ...]:
    axis_intervals = tuple(
        _split_integer_interval(axis_bounds, parts)
        for axis_bounds, parts in zip(bounds, subdivision, strict=True)
    )
    return tuple(
        (x_interval, y_interval, z_interval)
        for x_interval in axis_intervals[0]
        for y_interval in axis_intervals[1]
        for z_interval in axis_intervals[2]
    )


def _split_integer_interval(
    bounds: tuple[int, int],
    parts: int,
) -> tuple[tuple[int, int], ...]:
    start, end = bounds
    interval_length = end - start + 1
    base_length, remainder = divmod(interval_length, parts)
    intervals = []
    next_start = start
    for part_index in range(parts):
        part_length = base_length + int(part_index < remainder)
        part_end = next_start + part_length - 1
        intervals.append((next_start, part_end))
        next_start = part_end + 1
    return tuple(intervals)


def _coordinate_is_within(
    coordinate: tuple[int, int, int],
    bounds: AxisBounds,
) -> bool:
    return all(
        start <= value <= end
        for value, (start, end) in zip(coordinate, bounds, strict=True)
    )


def initialize_terminal_connections(cfg) -> Tensor:
    connection_shape = cfg.connection_shape
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
        cfg.z_axis_position - cfg.z_axis_range.value,
        cfg.z_axis_position + cfg.z_axis_range.value + 1,
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
    raise ValueError(f"Unsupported terminal connection shape: {connection_shape!r}")


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
            -cfg.z_axis_range.value,
            cfg.z_axis_range.value + 1,
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
    ellipsoid_offsets = []
    for x_delta in range(-xy_axis_range, xy_axis_range + 1):
        for y_delta in range(-xy_axis_range, xy_axis_range + 1):
            for z_delta in range(-z_axis_range, z_axis_range + 1):
                normalized_squared_distance = (
                    (x_delta / xy_axis_range) ** 2
                    + (y_delta / xy_axis_range) ** 2
                    + (z_delta / z_axis_range) ** 2
                )
                if normalized_squared_distance <= 1.0 + 1e-9:
                    ellipsoid_offsets.append((x_delta, y_delta, z_delta))
    return ellipsoid_offsets

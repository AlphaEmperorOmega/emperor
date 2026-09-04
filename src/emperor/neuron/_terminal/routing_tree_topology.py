from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from itertools import product
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from emperor.neuron._config import TerminalRoutingTreeConfig

AxisBounds = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
AxisSubdivision = tuple[int, int, int]


@dataclass(frozen=True)
class RoutingTreeNodePlan:
    """Immutable spatial assignment for one routing-tree node."""

    path: tuple[int, ...]
    level: int
    bounds: AxisBounds
    connection_indices: tuple[int, ...]
    subdivision: AxisSubdivision
    children: tuple[RoutingTreeNodePlan, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class RoutingTreePlan:
    """Compiled, translation-invariant partition of terminal connections."""

    depth: int
    direction_top_k: tuple[int, ...]
    leaf_top_k: int
    root: RoutingTreeNodePlan

    @property
    def output_width(self) -> int:
        width = self.leaf_top_k
        for level_top_k in self.direction_top_k:
            width *= level_top_k
        return width

    def walk(self):
        yield from self.root.walk()


class RoutingTreeCompiler:
    """Compile Terminal coordinates into a deterministic spatial routing plan."""

    def __init__(
        self,
        neuron_connections: Tensor,
        routing_tree_config: TerminalRoutingTreeConfig,
        leaf_top_k: int,
    ) -> None:
        self.coordinate_rows: tuple[tuple[int, int, int], ...] = (
            self.__snapshot_coordinate_rows(neuron_connections)
        )
        self.depth: int = int(routing_tree_config.depth.value)
        self.direction_branch_counts: tuple[int, ...] = tuple(
            routing_tree_config.direction_branch_counts
        )
        self.direction_top_k: tuple[int, ...] = tuple(
            routing_tree_config.direction_top_k
        )
        self.leaf_top_k: int = leaf_top_k

    @staticmethod
    def __snapshot_coordinate_rows(
        neuron_connections: Tensor,
    ) -> tuple[tuple[int, int, int], ...]:
        detached_neuron_connections = neuron_connections.detach()
        cpu_neuron_connections = detached_neuron_connections.cpu()
        connection_coordinate_rows = cpu_neuron_connections.tolist()
        integer_coordinate_rows = (
            tuple(int(component) for component in row)
            for row in connection_coordinate_rows
        )
        return tuple(integer_coordinate_rows)

    def compile(self) -> RoutingTreePlan:
        if not self.coordinate_rows:
            raise ValueError("Terminal routing tree requires at least one connection.")

        root_node_plan = self.__compile_root_node()
        routing_tree_plan = RoutingTreePlan(
            depth=self.depth,
            direction_top_k=self.direction_top_k,
            leaf_top_k=self.leaf_top_k,
            root=root_node_plan,
        )
        return routing_tree_plan

    def __compile_root_node(self) -> RoutingTreeNodePlan:
        root_bounds = self.__root_bounds()
        total_connections = len(self.coordinate_rows)
        connection_index_range = range(total_connections)
        all_connection_indices = tuple(connection_index_range)
        root_node_plan = self.__compile_node(
            connection_indices=all_connection_indices,
            bounds=root_bounds,
            path=(),
            level=0,
        )
        return root_node_plan

    def __root_bounds(self) -> AxisBounds:
        x_axis_bounds = self.__axis_bounds(0)
        y_axis_bounds = self.__axis_bounds(1)
        z_axis_bounds = self.__axis_bounds(2)
        return (x_axis_bounds, y_axis_bounds, z_axis_bounds)

    def __axis_bounds(self, axis: int) -> tuple[int, int]:
        minimum_axis_coordinate = min(row[axis] for row in self.coordinate_rows)
        maximum_axis_coordinate = max(row[axis] for row in self.coordinate_rows)
        return (minimum_axis_coordinate, maximum_axis_coordinate)

    def __compile_node(
        self,
        *,
        connection_indices: tuple[int, ...],
        bounds: AxisBounds,
        path: tuple[int, ...],
        level: int,
    ) -> RoutingTreeNodePlan:
        leaf_level = self.depth - 1
        if level == leaf_level:
            return self.__compile_leaf_node(
                connection_indices=connection_indices,
                bounds=bounds,
                path=path,
                level=level,
            )

        return self.__compile_direction_node(
            connection_indices=connection_indices,
            bounds=bounds,
            path=path,
            level=level,
        )

    @staticmethod
    def __compile_leaf_node(
        *,
        connection_indices: tuple[int, ...],
        bounds: AxisBounds,
        path: tuple[int, ...],
        level: int,
    ) -> RoutingTreeNodePlan:
        leaf_node_plan = RoutingTreeNodePlan(
            path=path,
            level=level,
            bounds=bounds,
            connection_indices=connection_indices,
            subdivision=(1, 1, 1),
        )
        return leaf_node_plan

    def __compile_direction_node(
        self,
        *,
        connection_indices: tuple[int, ...],
        bounds: AxisBounds,
        path: tuple[int, ...],
        level: int,
    ) -> RoutingTreeNodePlan:
        maximum_regions = self.direction_branch_counts[level]
        axis_subdivision = self.__balanced_axis_subdivision(bounds, maximum_regions)
        child_node_plans = self.__compile_child_nodes(
            connection_indices=connection_indices,
            bounds=bounds,
            subdivision=axis_subdivision,
            path=path,
            level=level,
        )
        routing_tree_node_plan = RoutingTreeNodePlan(
            path=path,
            level=level,
            bounds=bounds,
            connection_indices=connection_indices,
            subdivision=axis_subdivision,
            children=child_node_plans,
        )
        return routing_tree_node_plan

    @classmethod
    def __balanced_axis_subdivision(
        cls,
        bounds: AxisBounds,
        maximum_regions: int,
    ) -> AxisSubdivision:
        axis_spans = cls.__axis_spans(bounds)
        bounded_subdivision_candidates = cls.__bounded_subdivision_candidates(
            axis_spans,
            maximum_regions,
        )
        subdivision_score = partial(cls.__subdivision_score, axis_spans=axis_spans)
        best_subdivision = min(bounded_subdivision_candidates, key=subdivision_score)
        return best_subdivision

    @staticmethod
    def __axis_spans(bounds: AxisBounds) -> tuple[int, int, int]:
        x_axis_bounds, y_axis_bounds, z_axis_bounds = bounds
        x_axis_span = x_axis_bounds[1] - x_axis_bounds[0] + 1
        y_axis_span = y_axis_bounds[1] - y_axis_bounds[0] + 1
        z_axis_span = z_axis_bounds[1] - z_axis_bounds[0] + 1
        axis_spans = (x_axis_span, y_axis_span, z_axis_span)
        return axis_spans

    @classmethod
    def __bounded_subdivision_candidates(
        cls,
        axis_spans: tuple[int, int, int],
        maximum_regions: int,
    ) -> Iterator[AxisSubdivision]:
        x_axis_split_count_after_last = axis_spans[0] + 1
        y_axis_split_count_after_last = axis_spans[1] + 1
        z_axis_split_count_after_last = axis_spans[2] + 1
        x_axis_split_counts = range(1, x_axis_split_count_after_last)
        y_axis_split_counts = range(1, y_axis_split_count_after_last)
        z_axis_split_counts = range(1, z_axis_split_count_after_last)
        all_subdivision_candidates = product(
            x_axis_split_counts,
            y_axis_split_counts,
            z_axis_split_counts,
        )
        bounded_subdivision_candidates = (
            subdivision
            for subdivision in all_subdivision_candidates
            if cls.__subdivision_region_count(subdivision) <= maximum_regions
        )
        return bounded_subdivision_candidates

    @staticmethod
    def __subdivision_region_count(subdivision: AxisSubdivision) -> int:
        x_axis_split_count, y_axis_split_count, z_axis_split_count = subdivision
        return x_axis_split_count * y_axis_split_count * z_axis_split_count

    @classmethod
    def __subdivision_score(
        cls,
        subdivision: AxisSubdivision,
        *,
        axis_spans: tuple[int, int, int],
    ) -> tuple[int, Fraction, Fraction, int, int, int]:
        region_count = cls.__subdivision_region_count(subdivision)
        cell_edges = cls.__subdivision_cell_edges(axis_spans, subdivision)
        aspect_ratio = max(cell_edges) / min(cell_edges)
        edge_spread = cls.__cell_edge_spread(cell_edges)
        region_count_sort_key = -region_count
        x_axis_split_count, y_axis_split_count, z_axis_split_count = subdivision
        x_axis_split_count_sort_key = -x_axis_split_count
        y_axis_split_count_sort_key = -y_axis_split_count
        z_axis_split_count_sort_key = -z_axis_split_count
        return (
            region_count_sort_key,
            aspect_ratio,
            edge_spread,
            x_axis_split_count_sort_key,
            y_axis_split_count_sort_key,
            z_axis_split_count_sort_key,
        )

    @staticmethod
    def __subdivision_cell_edges(
        axis_spans: tuple[int, int, int],
        subdivision: AxisSubdivision,
    ) -> tuple[Fraction, ...]:
        axis_spans_with_split_counts = zip(
            axis_spans,
            subdivision,
            strict=True,
        )
        cell_edges = tuple(
            Fraction(span, split) for span, split in axis_spans_with_split_counts
        )
        return cell_edges

    @staticmethod
    def __cell_edge_spread(cell_edges: tuple[Fraction, ...]) -> Fraction:
        average_cell_edge = sum(cell_edges, Fraction()) / 3
        edge_spread = sum((edge - average_cell_edge) ** 2 for edge in cell_edges)
        return edge_spread

    def __compile_child_nodes(
        self,
        *,
        connection_indices: tuple[int, ...],
        bounds: AxisBounds,
        subdivision: AxisSubdivision,
        path: tuple[int, ...],
        level: int,
    ) -> tuple[RoutingTreeNodePlan, ...]:
        candidate_child_bounds = self.__subdivide_bounds(bounds, subdivision)
        child_node_plans = []
        for candidate_bounds in candidate_child_bounds:
            child_connection_indices = self.__connection_indices_within_bounds(
                connection_indices,
                candidate_bounds,
            )
            if not child_connection_indices:
                continue

            child_index = len(child_node_plans)
            child_path = (*path, child_index)
            child_level = level + 1
            child_node_plan = self.__compile_node(
                connection_indices=child_connection_indices,
                bounds=candidate_bounds,
                path=child_path,
                level=child_level,
            )
            child_node_plans.append(child_node_plan)

        compiled_child_node_plans = tuple(child_node_plans)
        return compiled_child_node_plans

    @classmethod
    def __subdivide_bounds(
        cls,
        bounds: AxisBounds,
        subdivision: AxisSubdivision,
    ) -> tuple[AxisBounds, ...]:
        x_axis_bounds, y_axis_bounds, z_axis_bounds = bounds
        x_axis_split_count, y_axis_split_count, z_axis_split_count = subdivision
        x_axis_intervals = cls.__split_integer_interval(
            x_axis_bounds,
            x_axis_split_count,
        )
        y_axis_intervals = cls.__split_integer_interval(
            y_axis_bounds,
            y_axis_split_count,
        )
        z_axis_intervals = cls.__split_integer_interval(
            z_axis_bounds,
            z_axis_split_count,
        )
        subdivided_bounds = tuple(
            (x_interval, y_interval, z_interval)
            for x_interval in x_axis_intervals
            for y_interval in y_axis_intervals
            for z_interval in z_axis_intervals
        )
        return subdivided_bounds

    @staticmethod
    def __split_integer_interval(
        bounds: tuple[int, int],
        parts: int,
    ) -> tuple[tuple[int, int], ...]:
        interval_start, interval_end = bounds
        interval_length = interval_end - interval_start + 1
        base_subinterval_length, longer_subinterval_count = divmod(
            interval_length,
            parts,
        )
        subintervals = []
        next_subinterval_start = interval_start
        for subinterval_index in range(parts):
            additional_coordinate = int(subinterval_index < longer_subinterval_count)
            subinterval_length = base_subinterval_length + additional_coordinate
            subinterval_end = next_subinterval_start + subinterval_length - 1
            subintervals.append((next_subinterval_start, subinterval_end))
            next_subinterval_start = subinterval_end + 1
        compiled_subintervals = tuple(subintervals)
        return compiled_subintervals

    def __connection_indices_within_bounds(
        self,
        connection_indices: tuple[int, ...],
        bounds: AxisBounds,
    ) -> tuple[int, ...]:
        matching_connection_indices = []
        for connection_index in connection_indices:
            connection_coordinate = self.coordinate_rows[connection_index]
            coordinate_is_within_bounds = self.__coordinate_is_within(
                connection_coordinate,
                bounds,
            )
            if coordinate_is_within_bounds:
                matching_connection_indices.append(connection_index)
        compiled_matching_connection_indices = tuple(matching_connection_indices)
        return compiled_matching_connection_indices

    @staticmethod
    def __coordinate_is_within(
        coordinate: tuple[int, int, int],
        bounds: AxisBounds,
    ) -> bool:
        coordinate_components_with_bounds = zip(coordinate, bounds, strict=True)
        axis_membership_checks = (
            start <= value <= end
            for value, (start, end) in coordinate_components_with_bounds
        )
        coordinate_is_within_all_axis_bounds = all(axis_membership_checks)
        return coordinate_is_within_all_axis_bounds

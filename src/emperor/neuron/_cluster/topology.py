from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.neuron._cluster.model import NeuronCluster


class ClusterTopologyDelegate:
    """Resolve coordinates against the owner's live neuron grid."""

    def __init__(self, owner: NeuronCluster) -> None:
        self.__owner = owner

    def neuron_name(self, x: int, y: int, z: int) -> str:
        return f"neuron_{x}_{y}_{z}"

    def parse_neuron_name(self, neuron_name: str) -> tuple[int, int, int]:
        _, x, y, z = neuron_name.split("_")
        return int(x), int(y), int(z)

    def is_neuron_name(self, name: str) -> bool:
        return re.fullmatch(r"neuron_\d+_\d+_\d+", name) is not None

    def coordinate_from_row(self, row: list[int]) -> tuple[int, int, int]:
        x, y, z = row
        return int(x), int(y), int(z)

    def is_within_grid_capacity(self, coordinate: tuple[int, int, int]) -> bool:
        x, y, z = coordinate
        return (
            1 <= x <= self.__owner.x_axis_total_neurons
            and 1 <= y <= self.__owner.y_axis_total_neurons
            and 1 <= z <= self.__owner.z_axis_total_neurons
        )

    def is_valid_coordinate(self, coordinate: tuple[int, int, int]) -> bool:
        if not self.is_within_grid_capacity(coordinate):
            return False
        return self.neuron_name(*coordinate) in self.__owner.cluster

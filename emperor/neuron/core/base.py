import re

from torch import Tensor
from torch.nn import ModuleDict

from emperor.base.utils import Module


class NeuronClusterModuleBase(Module):
    def _add_neuron(self, cluster: ModuleDict, name: str, instance: Module) -> None:
        cluster[name] = instance

    def _neuron_name(self, x: int, y: int, z: int) -> str:
        return f"neuron_{x}_{y}_{z}"

    def _parse_neuron_name(self, neuron_name: str) -> tuple[int, int, int]:
        _, x, y, z = neuron_name.split("_")
        return int(x), int(y), int(z)

    def _is_neuron_name(self, name: str) -> bool:
        return re.fullmatch(r"neuron_\d+_\d+_\d+", name) is not None

    def _coordinate_from_row(self, row: list[int]) -> tuple[int, int, int]:
        x, y, z = row
        return int(x), int(y), int(z)

    def _is_within_grid_capacity(self, coordinate: tuple[int, int, int]) -> bool:
        x, y, z = coordinate
        return (
            1 <= x <= self.x_axis_total_neurons
            and 1 <= y <= self.y_axis_total_neurons
            and 1 <= z <= self.z_axis_total_neurons
        )

    def _accumulate_auxiliary_loss(
        self,
        loss: Tensor,
        auxiliary_loss: Tensor,
    ) -> Tensor:
        return loss + auxiliary_loss

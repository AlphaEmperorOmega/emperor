from dataclasses import dataclass, field

from torch._prims_common import Tensor
from torch.nn import ModuleDict
from Emperor.base.utils import DataClassBase, Module
from Emperor.config import ModelConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class NeuronClusterConfig(DataClassBase):
    x_axis_total_neurons: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    y_axis_total_neurons: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    z_axis_total_neurons: int | None = field(
        default=None,
        metadata={"help": ""},
    )


class NeuronCluster(Module):
    def __init__(
        self,
        cfg: "NeuronClusterConfig | ModelConfig",
        overrides: "NeuronClusterConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "neuron_cluster_config", cfg)
        self.cfg: "NeuronClusterConfig" = self._overwrite_config(config, overrides)
        self.x_axis_total_neurons = self.cfg.x_axis_total_neurons
        self.y_axis_total_neurons = self.cfg.y_axis_total_neurons
        self.z_axis_total_neurons = self.cfg.z_axis_total_neurons

        self.cluster = self.__initialize_cluster()

    def __initialize_cluster(self) -> ModuleDict:
        cluster = ModuleDict()
        for x_coordinate in range(1, self.x_axis_total_neurons + 1):
            for y_coordinate in range(1, self.y_axis_total_neurons + 1):
                for z_coordinate in range(1, self.z_axis_total_neurons + 1):
                    name = self.__neuron_name(x_coordinate, y_coordinate, z_coordinate)
                    instance = self.__initialize_neuron()
                    self.__add_neuron(cluster, name, instance)
        return cluster

    def __add_neuron(self, cluster: ModuleDict, name: str, instance: Neuron) -> None:
        cluster[name] = instance

    def __neuron_name(self, x: int, y: int, z: int) -> str:
        return f"neuron_{x}_{y}_{z}"

    def __initialize_neuron(self) -> Neuron:
        return Neuron(self.main_config)

    def forward(self, input: Tensor) -> Tensor:
        should_process_neurons = True

        while should_process_neurons:
            selected_neuron = self.__select_enuron(data)
            output, loss = selected_neuron(input)

        return output

    def __select_enuron(self, data) -> Neuron:
        selected_x_coordinate = 1
        selected_y_coordinate = 1
        selected_z_coordinate = 1
        neuron_name = self.__neuron_name(
            selected_x_coordinate,
            selected_y_coordinate,
            selected_z_coordinate,
        )
        return self.cluster[neuron_name]

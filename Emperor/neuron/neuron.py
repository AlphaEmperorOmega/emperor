from dataclasses import dataclass, field

from torch._prims_common import Tensor
from torch.nn import ModuleDict
from Emperor.attention.attention import MultiHeadAttention
from Emperor.base.utils import DataClassBase, Module
from Emperor.config import ModelConfig

from typing import TYPE_CHECKING

from Emperor.layers.utils.enums import LinearLayerTypes, ParameterGeneratorTypes
from Emperor.transformer.layer import (
    Transformer,
    TransformerDecoder,
    TransformerEncoder,
)

if TYPE_CHECKING:
    from Emperor.config import ModelConfig

# TODO: An idea is to create a residual router with the new
# dynamic linear layer where you loop until only one
# route remains. Just to make this clear:
# - first iteration only the first top-k will pass
# - second iteration from the remaining neurons you need to repeat the process from 1
# until only a single neuron remains


class Nucleus(Module):
    def __init__(
        self,
        processing_unit: Transformer
        | TransformerEncoder
        | TransformerDecoder
        | MultiHeadAttention
        | ParameterGeneratorTypes
        | LinearLayerTypes,
    ):
        super().__init__()
        self.processor_unit = processing_unit

    def forward(self, input: Tensor) -> Tensor:
        output = self.processing_unit(input)
        return output


class Axons(Module):
    def __init__(
        self,
        processing_unit,
    ):
        super().__init__()
        self.processor_unit = processing_unit

    def forward(self, input: Tensor) -> Tensor:
        output = self.processing_unit(input)
        return output


@dataclass
class TerminalConfig(DataClassBase):
    x_axis_position: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    y_axis_position: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    z_axis_position: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    x_axis_range: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    y_axis_range: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    z_axis_range: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    use_multi_level_routing_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    router_type: int | None = field(
        default=None,
        metadata={"help": ""},
    )


class Terminal(Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.x_axis_position = self.cfg.x_axis_position
        self.y_axis_position = self.cfg.y_axis_position
        self.z_axis_position = self.cfg.z_axis_position
        self.x_axis_range = self.cfg.x_axis_range
        self.y_axis_range = self.cfg.y_axis_range
        self.z_axis_range = self.cfg.z_axis_range
        self.use_multi_level_routing_flag = self.cfg.use_multi_level_routing_flag
        self.router_type = self.cfg.router_type

        self.num_split_signals = torch.tensor(num_split_signals)
        self.row_range = row_range
        self.col_range = col_range
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.total_root_routers = 4
        self.terminals_per_axon = col_range * 2 + 1
        self.total_col_range = col_range * 2 + 1

        self.terminal_root_router = terminal_root_router
        self.terminal_branch_routers = terminal_branch_routers

        self.connections = self.__initialize_connections()

    def __initialize_connections(self, row_idx, col_idx):
        # Range of all layers the current neuron has access to, when
        # row_range = 4 and row_idx = 0
        # Ex: [-1, 0, 1, 2, 3]
        # Negative numbers indicate connections do not exist and positive otherwise
        row_range_indexes = torch.arange((row_idx - 1), (row_idx + self.row_range - 1))

        # Range of all neuron connections the current neuron has access to when
        # col_range = 3 and col_idx = 0
        # Ex: [-3, -2, -1,  0,  1,  2,  3]
        # Negative numbers indicate connections do not exist and positive otherwise
        col_range_indexes = torch.arange(
            (col_idx - self.col_range), (col_idx + self.col_range + 1)
        )

        # Get all coordinates accessible by the current neuron, accessible or not
        # [[-1, -3], [-1, -2], [-1, -1], [-1,  0], [-1,  1], [-1,  2], [-1,  3],
        #  [ 0, -3], [ 0, -2], [ 0, -1], [ 0,  0], [ 0,  1], [ 0,  2], [ 0,  3],
        #  [ 1, -3], [ 1, -2], [ 1, -1], [ 1,  0], [ 1,  1], [ 1,  2], [ 1,  3],
        #  [ 2, -3], [ 2, -2], [ 2, -1], [ 2,  0], [ 2,  1], [ 2,  2], [ 2,  3]]
        curr_neuron_connections = torch.cartesian_prod(
            row_range_indexes, col_range_indexes
        )

        # Generate terminal connections for each axon, Connections available for each 4 axons:
        # - axon 1: [[-1, -3], [-1, -2], [-1, -1], [-1,  0], [ 0, -3], [ 0, -2], [ 0, -1]]
        # - axon 2: [[-1,  1], [-1,  2], [-1,  3], [ 0,  0], [ 0,  1], [ 0,  2], [ 0,  3]]
        # - axon 3: [[ 1, -3], [ 1, -2], [ 1, -1], [ 1,  0], [ 2, -3], [ 2, -2], [ 2, -1]]
        # - axon 4: [[ 1,  1], [ 1,  2], [ 1,  3], [ 2,  0], [ 2,  1], [ 2,  2], [ 2,  3]]
        terminal_connections = []

        for axon in range(0, self.row_range, 2):
            temp = torch.zeros(self.row_range, self.terminals_per_axon)
            temp[axon : axon + 2, : self.col_range] = 1.0
            temp[axon : axon + 1, self.col_range : (self.col_range + 1)] = 1.0
            temp = temp.view(-1).bool()
            temp = curr_neuron_connections[temp]
            terminal_connections.append(temp)

            temp = torch.zeros(self.row_range, self.terminals_per_axon)
            temp[axon : axon + 2, (self.col_range + 1) :] = 1.0
            temp[axon + 1 : axon + 2, self.col_range : self.col_range + 1] = 1.0
            temp = temp.view(-1).bool()
            temp = curr_neuron_connections[temp]
            terminal_connections.append(temp)

        setattr(self, "row_idx", row_idx)
        setattr(self, "col_idx", col_idx)
        setattr(self, "row_range_indexes", row_range_indexes)
        setattr(self, "col_range_indexes", col_range_indexes)
        setattr(self, "terminal_connections", terminal_connections)

    def forward(self, input: Tensor) -> Tensor:
        output = self.processing_unit(input)
        return output


@dataclass
class NeuronConfig(DataClassBase):
    pass


class Neuron(Module):
    def __init__(
        self,
        cfg: "NeuronConfig | ModelConfig",
        overrides: "NeuronConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "neuron_cluster_config", cfg)
        self.cfg: "NeuronClusterConfig" = self._overwrite_config(config, overrides)
        self.nucleus = Nucleus(cfg)
        self.axons = Axons(cfg)
        self.router = Terminal()

    def forward(self, input: Tensor) -> tuple[Tensor, float]:
        processed_signal = self.nucleus(input)
        transmitted_signal = self.axons(processed_signal)
        output, loss = self.router(transmitted_signal)
        return output, loss


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

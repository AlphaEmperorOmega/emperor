import torch

from enum import Enum
from torch.nn import ModuleDict
from dataclasses import dataclass, field

from torch.types import Tensor
from Emperor.base.utils import DataClassBase, Module
from Emperor.layers.utils.enums import LayerTypes
from Emperor.layers.utils.routers import RouterModel
from Emperor.layers.utils.samplers import SamplerModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig

# TODO: An idea is to create a residual router with the new
# dynamic linear layer where you loop until only one
# route remains. Just to make this clear:
# - first iteration only the first top-k will pass
# - second iteration from the remaining neurons you need to repeat the process from 1
# until only a single neuron remains


@dataclass
class NucleusConfig(DataClassBase):
    model_type: LayerTypes | None = field(
        default=None,
        metadata={"help": "Type of layer used for the experts."},
    )


class Nucleus(Module):
    def __init__(
        self,
        cfg: "NucleusConfig | ModelConfig",
        overrides: "NucleusConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "neuron_nucleus_config", cfg)
        self.cfg: "NucleusConfig" = self._overwrite_config(config, overrides)
        self.model_type = self.cfg.model_type
        self.processing_unit = self.__create_model(cfg)

    def __create_model(self, cfg: "ModelConfig"):
        return self.model_type.value(cfg)

    def forward(self, input: Tensor) -> Tensor:
        output = self.processing_unit(input)
        return output


@dataclass
class AxonsConfig(DataClassBase):
    memory_type: LayerTypes | None = field(
        default=None,
        metadata={
            "help": "Memory module used as axons, or additional modules that modify the output of nucleus"
        },
    )


class Axons(Module):
    def __init__(
        self,
        cfg: "AxonsConfig | ModelConfig",
        overrides: "AxonsConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "neuron_axon_config", cfg)
        self.cfg: "AxonsConfig" = self._overwrite_config(config, overrides)
        self.memory_type = self.cfg.memory_type

    def forward(self, input: Tensor) -> Tensor:
        return input


class TerminalRangeOptions(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8


class TerminalZAxisOffsetOptions(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


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
    xy_axis_range: TerminalRangeOptions | None = field(
        default=None,
        metadata={
            "help": "Defines the range for the xy axis surrounding the current neuron, with the neuron's position as the center. The total span is calculated as (xy_axis_range * 2 + 1), excluding the current neuron itself."
        },
    )
    z_axis_range: TerminalRangeOptions | None = field(
        default=None,
        metadata={
            "help": "Defines the range for the z axis neighboring the current neuron. The actual span is (z_axis_range + 1), excluding the position of the current neuron itself."
        },
    )
    z_axis_offset: TerminalZAxisOffsetOptions | None = field(
        default=None,
        metadata={
            "help": "Specifies the offset along the z axis for the current neuron. This determines how many rows of neurons behind the current z-axis position can be accessed, effectively shifting the accessible range and allowing the model to connect to neurons in previous rows."
        },
    )


class Terminal(Module):
    # TODO: Later add ability to split connections into branches
    # by that i mean:
    # - think of the connections to a neuron as a tree structure, where
    # each branch connects to smaller branches until reaching the final neuron
    # - for each branch split you neeed to have a router and sampler model
    # For example:
    # - first router chooses the main branches where each branch has the same number
    # of connections
    # - the chosen branch has it's own router meaning with its connections to neuron
    # this needs to be similar ot `RouterDecisionSquash` in the old version
    def __init__(
        self,
        cfg: "TerminalConfig | ModelConfig",
        overrides: "TerminalConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "neuron_terminal_config", cfg)
        self.cfg: "TerminalConfig" = self._overwrite_config(config, overrides)
        self.x_axis_position = self.cfg.x_axis_position
        self.y_axis_position = self.cfg.y_axis_position
        self.z_axis_position = self.cfg.z_axis_position
        self.xy_axis_range = self.cfg.xy_axis_range.value
        self.z_axis_range = self.cfg.z_axis_range.value
        self.z_axis_offset = self.cfg.z_axis_offset.value

        self.router = RouterModel(cfg)
        self.sampler = SamplerModel(cfg)

        self.total_neuron_connections = self.__compute_total_neuron_connections()
        self.neuron_connections = self.__initialize_connections()

    def __compute_total_neuron_connections(self) -> int:
        positions_left_right = 2
        current_neuron_offset = 1
        single_axis_range = (
            self.xy_axis_range * positions_left_right + current_neuron_offset
        )
        total_xy_axis_connections = single_axis_range**2
        return total_xy_axis_connections * (self.z_axis_range + current_neuron_offset)

    def __initialize_connections(self) -> Tensor:
        is_y_axis_flag = True
        x_axis_range_indexes = self.__compute_xy_axis_range()
        y_axis_range_indexes = self.__compute_xy_axis_range(is_y_axis_flag)
        z_axis_range_indexes = self.__compute_z_axis_range()
        return torch.cartesian_prod(
            x_axis_range_indexes,
            y_axis_range_indexes,
            z_axis_range_indexes,
        )

    def __compute_xy_axis_range(self, is_y_axis_flag: bool = False) -> Tensor:
        position = self.y_axis_position if is_y_axis_flag else self.x_axis_position
        range_start = position - self.xy_axis_range
        range_end = position + self.xy_axis_range + 1
        return torch.arange(range_start, range_end)

    def __compute_z_axis_range(self) -> Tensor:
        self.__validate_z_axis_range_for_given_offset()
        self.__validate_z_axis_neuron_connections()
        current_neuron_offset = 1
        range_start = self.z_axis_position - self.z_axis_offset
        range_end = (
            self.z_axis_position
            + self.z_axis_range
            - self.z_axis_offset
            + current_neuron_offset
        )
        return torch.arange(range_start, range_end)

    def __validate_z_axis_range_for_given_offset(self) -> None:
        if self.z_axis_range <= 2 and self.z_axis_offset > 0:
            raise ValueError(
                f"Invalid configuration: z_axis_range ({self.z_axis_range}) must be greater than 2 when z_axis_offset ({self.z_axis_offset}) is positive."
            )

    def __validate_z_axis_neuron_connections(self) -> None:
        if (self.z_axis_range - self.z_axis_offset) <= 0:
            raise ValueError(
                f"Invalid configuration: z_axis_range ({self.z_axis_range}) is too small relative to z_axis_offset ({self.z_axis_offset}). When z_axis_offset is positive, z_axis_range must be greater than z_axis_offset, this ensures the current neuron has access to forward neurons."
            )

    def forward(self, input: Tensor) -> Tensor:
        logits = self.router.compute_logit_scores(input)
        probabilities, indices, _, _ = self.sampler.sample_probabilities_and_indices(
            logits
        )
        selected_neurons = self.neuron_connections[indices]

        return input, probabilities, selected_neurons


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
        self.terminal = Terminal(cfg)

    def forward(self, input: Tensor) -> tuple[Tensor, float]:
        processed_signal = self.nucleus(input)
        augmented_signal = self.axons(processed_signal)
        output, probabilities, selected_neurons = self.terminal(augmented_signal)

        return output, probabilities, selected_neurons


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
        self.main_config = cfg
        self.x_axis_total_neurons = self.cfg.x_axis_total_neurons
        self.y_axis_total_neurons = self.cfg.y_axis_total_neurons
        self.z_axis_total_neurons = self.cfg.z_axis_total_neurons

        self.cluster = self.__initialize_cluster()
        print(self.cluster)

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

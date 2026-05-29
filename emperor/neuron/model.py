import copy

from torch import Tensor
from torch.nn import ModuleDict

from emperor.base.utils import Module
from emperor.neuron.config import NeuronClusterConfig
from emperor.neuron.core.config import TerminalConfig
from emperor.neuron.core._validator import NeuronClusterValidator


class NeuronCluster(Module):
    def __init__(
        self,
        cfg: NeuronClusterConfig,
        overrides: NeuronClusterConfig | None = None,
    ):
        super().__init__()
        self.cfg: NeuronClusterConfig = self._override_config(cfg, overrides)
        NeuronClusterValidator.validate(self)

        self.x_axis_total_neurons: int = self.cfg.x_axis_total_neurons
        self.y_axis_total_neurons: int = self.cfg.y_axis_total_neurons
        self.z_axis_total_neurons: int = self.cfg.z_axis_total_neurons
        self.growth_threshold: int | None = self.cfg.growth_threshold

        self.cluster = self.__initialize_cluster()

    def __initialize_cluster(self) -> ModuleDict:
        cluster = ModuleDict()
        for x_coordinate in range(1, self.x_axis_total_neurons + 1):
            for y_coordinate in range(1, self.y_axis_total_neurons + 1):
                for z_coordinate in range(1, self.z_axis_total_neurons + 1):
                    name = self.__neuron_name(
                        x_coordinate,
                        y_coordinate,
                        z_coordinate,
                    )
                    self.__add_neuron(
                        cluster,
                        name,
                        self.__initialize_neuron(
                            x_coordinate,
                            y_coordinate,
                            z_coordinate,
                        ),
                    )
        return cluster

    def __add_neuron(self, cluster: ModuleDict, name: str, instance: Module) -> None:
        cluster[name] = instance

    def __neuron_name(self, x: int, y: int, z: int) -> str:
        return f"neuron_{x}_{y}_{z}"

    def __initialize_neuron(self, x: int, y: int, z: int) -> Module:
        neuron_config = copy.deepcopy(self.cfg.neuron_config)
        terminal_config = neuron_config.terminal_config
        terminal_overrides = TerminalConfig(
            x_axis_position=x,
            y_axis_position=y,
            z_axis_position=z,
        )
        neuron_config.terminal_config = self._override_config(
            terminal_config,
            terminal_overrides,
        )
        return neuron_config.build()

    def __check_neuron_growth(self) -> None:
        if self.growth_threshold is None:
            return

        for name, neuron in list(self.cluster.items()):
            if int(neuron.batch_counter.item()) < self.growth_threshold:
                continue

            neuron.batch_counter.zero_()
            position = self.__find_closest_empty_connection(name)
            if position is None:
                return

            x, y, z = position
            new_name = self.__neuron_name(x, y, z)
            self.__add_neuron(
                self.cluster,
                new_name,
                self.__initialize_neuron(x, y, z),
            )
            return

    def __find_closest_empty_connection(
        self, neuron_name: str
    ) -> tuple[int, int, int] | None:
        neuron = self.cluster[neuron_name]
        connections = neuron.terminal.neuron_connections
        origin_x, origin_y, origin_z = self.__parse_neuron_name(neuron_name)

        empty_positions = []
        for connection in connections:
            x, y, z = (int(value.item()) for value in connection)
            candidate_name = self.__neuron_name(x, y, z)
            if candidate_name not in self.cluster:
                empty_positions.append((x, y, z))

        if not empty_positions:
            return None

        return min(
            empty_positions,
            key=lambda pos: abs(pos[0] - origin_x)
            + abs(pos[1] - origin_y)
            + abs(pos[2] - origin_z),
        )

    def __parse_neuron_name(self, neuron_name: str) -> tuple[int, int, int]:
        _, x, y, z = neuron_name.split("_")
        return int(x), int(y), int(z)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        NeuronClusterValidator.validate_forward_input(input)
        selected_neuron = self.__select_neuron()
        output, probabilities, selected_neurons = selected_neuron(input)
        self.__check_neuron_growth()
        return output, probabilities, selected_neurons

    def __select_neuron(self) -> Module:
        first_neuron_name = self.__neuron_name(1, 1, 1)
        return self.cluster[first_neuron_name]

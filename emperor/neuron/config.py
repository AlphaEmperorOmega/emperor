from dataclasses import dataclass

from emperor.base.utils import ConfigBase, optional_field
from emperor.neuron.core.config import AxonsConfig, NucleusConfig, TerminalConfig


@dataclass
class NeuronConfig(ConfigBase):
    nucleus_config: NucleusConfig | None = optional_field(
        "Nucleus stage configuration."
    )
    axons_config: AxonsConfig | None = optional_field(
        "Axons stage configuration."
    )
    terminal_config: TerminalConfig | None = optional_field(
        "Terminal routing stage configuration."
    )

    def _registry_owner(self) -> type:
        from emperor.neuron.core.layers import Neuron

        return Neuron


@dataclass
class NeuronClusterConfig(ConfigBase):
    x_axis_total_neurons: int | None = optional_field(
        "Initial cluster size along the x axis."
    )
    y_axis_total_neurons: int | None = optional_field(
        "Initial cluster size along the y axis."
    )
    z_axis_total_neurons: int | None = optional_field(
        "Initial cluster size along the z axis."
    )
    growth_threshold: int | None = optional_field(
        "Forward-pass count that triggers growth. Set to None to disable."
    )
    neuron_config: NeuronConfig | None = optional_field(
        "Base neuron configuration used for each cluster coordinate."
    )

    def _registry_owner(self) -> type:
        from emperor.neuron.model import NeuronCluster

        return NeuronCluster

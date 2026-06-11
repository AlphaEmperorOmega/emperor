from dataclasses import dataclass

from emperor.base.utils import ConfigBase, optional_field
from emperor.halting.config import HaltingConfig
from emperor.neuron.core.config import AxonsConfig, NucleusConfig, TerminalConfig
from emperor.sampler.core.config import SamplerConfig


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
        "Maximum cluster capacity along the x axis."
    )
    y_axis_total_neurons: int | None = optional_field(
        "Maximum cluster capacity along the y axis."
    )
    z_axis_total_neurons: int | None = optional_field(
        "Maximum cluster capacity along the z axis."
    )
    initial_x_axis_total_neurons: int | None = optional_field(
        "Initially instantiated cluster size along the x axis, centered within "
        "x_axis_total_neurons. Defaults to x_axis_total_neurons."
    )
    initial_y_axis_total_neurons: int | None = optional_field(
        "Initially instantiated cluster size along the y axis, centered within "
        "y_axis_total_neurons. Defaults to y_axis_total_neurons."
    )
    initial_z_axis_total_neurons: int | None = optional_field(
        "Initially instantiated cluster size along the z axis, centered within "
        "z_axis_total_neurons. Defaults to z_axis_total_neurons."
    )
    entry_sampler_config: SamplerConfig | None = optional_field(
        "Optional sampler used to route inputs into initialized entry-plane neurons."
    )
    max_steps: int | None = optional_field(
        "Maximum recurrent route steps before the cluster stops traversal."
    )
    growth_threshold: int | None = optional_field(
        "Neuron process_signal call count that triggers growth. Counted in "
        "training mode only and includes speculative top-k branch "
        "evaluations, so neurons in popular neighborhoods accrue pressure "
        "without being chosen. Set to None to disable."
    )
    escape_driven_growth_flag: bool | None = optional_field(
        "When True, growth placement targets the empty connection coordinate "
        "with the highest accumulated escape count (routes that selected a "
        "missing in-capacity neuron), falling back to the Manhattan-closest "
        "empty connection when no escapes were recorded. Requires "
        "growth_threshold. Defaults to False."
    )
    mitosis_initialization_flag: bool | None = optional_field(
        "When True, a grown neuron copies the grown-from neuron's parameters "
        "perturbed by 1% relative gaussian noise instead of using a fresh "
        "initialization. Requires growth_threshold. Defaults to False."
    )
    halting_config: HaltingConfig | None = optional_field(
        "Optional learned cluster-level halting module."
    )
    neuron_config: NeuronConfig | None = optional_field(
        "Base neuron configuration used for each cluster coordinate."
    )

    def _registry_owner(self) -> type:
        from emperor.neuron.model import NeuronCluster

        return NeuronCluster

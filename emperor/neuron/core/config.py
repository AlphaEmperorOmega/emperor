from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.base.utils import ConfigBase, optional_field
from emperor.halting.config import HaltingConfig
from emperor.neuron.core.options import (
    TerminalConnectionShapeOptions,
    TerminalRangeOptions,
    TerminalZAxisOffsetOptions,
)
from emperor.sampler.core.config import SamplerConfig

if TYPE_CHECKING:
    from emperor.memory.config import DynamicMemoryConfig


@dataclass
class NucleusConfig(ConfigBase):
    model_config: ConfigBase | None = optional_field(
        "Config for the model applied inside the nucleus."
    )

    def _registry_owner(self) -> type:
        from emperor.neuron.core.layers import Nucleus

        return Nucleus


@dataclass
class AxonsConfig(ConfigBase):
    memory_config: "DynamicMemoryConfig | None" = optional_field(
        "Optional memory model applied after the nucleus. Set to None for identity."
    )

    def _registry_owner(self) -> type:
        from emperor.neuron.core.layers import Axons

        return Axons


@dataclass
class TerminalConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Feature dimension of the tensor routed by the terminal."
    )
    x_axis_position: int | None = optional_field(
        "Current neuron x-axis coordinate."
    )
    y_axis_position: int | None = optional_field(
        "Current neuron y-axis coordinate."
    )
    z_axis_position: int | None = optional_field(
        "Current neuron z-axis coordinate."
    )
    xy_axis_range: TerminalRangeOptions | None = optional_field(
        "Neighbor range on each side of the x and y axes."
    )
    z_axis_range: TerminalRangeOptions | None = optional_field(
        "Forward z-axis neighbor range."
    )
    z_axis_offset: TerminalZAxisOffsetOptions | None = optional_field(
        "Backward z-axis offset applied before the forward z range."
    )
    sampler_config: SamplerConfig | None = optional_field(
        "Sampler configuration used to choose terminal connections."
    )
    connection_shape: TerminalConnectionShapeOptions | None = optional_field(
        "Geometry of the terminal's connection neighborhood. BOX (default) "
        "is the full cartesian box of the configured ranges; CROSS keeps "
        "only the three axis lines through the neuron; SPHERE keeps box "
        "offsets inside the ellipsoid inscribed in the box; DIAGONAL_X "
        "keeps the two xy-plane diagonals; the LINE shapes keep a single "
        "axis line (left-right = x, up-down = y, front-back = z with the "
        "usual offset window) so signals can jump far along one axis "
        "without the quadratic fan-out of a box. The sampler num_experts "
        "must match the resulting connection count. Defaults to BOX."
    )

    def _registry_owner(self) -> type:
        from emperor.neuron.core.layers import Terminal

        return Terminal


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
    coordinate_embedding_flag: bool | None = optional_field(
        "When True, a fixed sinusoidal encoding of the neuron's (x, y, z) "
        "coordinate is added to the nucleus and terminal inputs so "
        "processing and routing are position-aware. Requires "
        "terminal_config.input_dim of at least 3. Defaults to False."
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
    beam_width: int | None = optional_field(
        "Number of routes kept alive per sample during traversal. With a "
        "width of 2 or more, every step expands each live route's top-k "
        "branches, scores candidates by accumulated log route probability, "
        "prunes back to this many routes, and the final output merges the "
        "surviving routes weighted by their softmaxed scores. A width of 1 "
        "keeps the single-route weighted continuation. Defaults to 1."
    )
    growth_threshold: int | None = optional_field(
        "Neuron process_signal call count that triggers growth. Counted in "
        "training mode only and includes speculative top-k branch "
        "evaluations, so neurons in popular neighborhoods accrue pressure "
        "without being chosen. Set to None to disable."
    )
    growth_cooldown_steps: int | None = optional_field(
        "Minimum training forwards between two growth events. The cooldown "
        "counter starts at zero, so the first growth also waits this many "
        "forwards. Requires growth_threshold. Set to None to disable."
    )
    max_total_growths: int | None = optional_field(
        "Lifetime budget of grown neurons; once reached, growth stops "
        "permanently. The growth count persists in checkpoints. Requires "
        "growth_threshold. Set to None for unlimited growth."
    )
    growth_warmup_steps: int | None = optional_field(
        "Number of training forwards over which a grown neuron's process "
        "output fades in linearly. While warming up the neuron returns "
        "weight * output + (1 - weight) * input with weight ramping from "
        "1/growth_warmup_steps to 1, so a fresh neuron starts near-identity "
        "and gradually becomes itself. The countdown advances once per "
        "training forward whether or not the neuron is routed, and persists "
        "in checkpoints. Requires growth_threshold. Set to None to disable."
    )
    pruning_threshold: int | None = optional_field(
        "Synchronized atrophy count that triggers pruning of an idle neuron. "
        "A neuron's atrophy counter resets to zero on any training forward "
        "in which it receives a process_signal call (on any rank) and "
        "increments otherwise; a neuron grown this forward counts as used, "
        "so fresh neurons get a full grace period. When the cross-rank "
        "minimum reaches this threshold the most atrophied neuron outside "
        "the entry plane is removed, at most one per forward. Freed "
        "coordinates become growth candidates again. Set to None to disable."
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
        from emperor.neuron.core.model import NeuronCluster

        return NeuronCluster

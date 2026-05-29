from dataclasses import dataclass

from emperor.base.utils import ConfigBase, optional_field
from emperor.neuron.core.options import (
    TerminalRangeOptions,
    TerminalZAxisOffsetOptions,
)
from emperor.sampler.core.config import SamplerConfig


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
    memory_config: ConfigBase | None = optional_field(
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

    def _registry_owner(self) -> type:
        from emperor.neuron.core.layers import Terminal

        return Terminal

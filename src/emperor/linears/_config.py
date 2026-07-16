from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field


@dataclass
class LinearLayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    bias_flag: bool | None = optional_field("Add a learnable bias to the output.")

    def _registry_owner(self) -> type:
        from emperor.linears._layer import LinearLayer

        return LinearLayer

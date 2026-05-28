from dataclasses import dataclass
from emperor.base.utils import ConfigBase, optional_field


@dataclass
class Conv2dLayerConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input channel count (Conv2d in_channels)."
    )
    output_dim: int | None = optional_field(
        "Output channel count (Conv2d out_channels)."
    )
    kernel_size: int | None = optional_field(
        "Conv2d kernel size."
    )
    stride: int | None = optional_field(
        "Conv2d stride."
    )
    padding: int | None = optional_field(
        "Conv2d padding."
    )
    bias_flag: bool | None = optional_field(
        "Add a learnable bias to the output."
    )

    def _registry_owner(self) -> type:
        from emperor.convs.core.layers import Conv2dLayer

        return Conv2dLayer

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor, nn

from emperor.config import ConfigBase, optional_field
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.linears import LinearLayerConfig


@dataclass
class GptTokenwiseLinearLayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    bias_flag: bool | None = optional_field("Whether the affine projection has bias.")

    def _registry_owner(self) -> type:
        return GptTokenwiseLinearLayer


class GptTokenwiseLinearLayer(Module):
    def __init__(
        self,
        cfg: "GptTokenwiseLinearLayerConfig",
        overrides: "GptTokenwiseLinearLayerConfig | LinearLayerConfig | None" = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.bias_flag: bool = self.cfg.bias_flag
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=self.bias_flag)
        self._initialize_parameters(self.linear)

    def forward(self, input_batch: Tensor) -> Tensor:
        if input_batch.dim() < 2:
            raise ValueError(
                "GptTokenwiseLinearLayer expects at least 2D input with a feature "
                f"dimension, got shape {tuple(input_batch.shape)}."
            )
        if input_batch.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got "
                f"{input_batch.shape[-1]}."
            )
        leading_shape = input_batch.shape[:-1]
        flat = input_batch.reshape(-1, self.input_dim)
        output = self.linear(flat)
        return output.reshape(*leading_shape, self.output_dim)

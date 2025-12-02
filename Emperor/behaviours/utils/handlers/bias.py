from torch import Tensor
from Emperor.base.utils import Module
from Emperor.base.layer import (
    LayerStackConfig,
    LinearLayerStack,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.behaviours.utils.layers import LinearLayerConfig


class BiasHandlerAbstract(Module):
    def __init__(
        self,
        cfg: "LinearLayerConfig",
    ):
        super().__init__()
        self.cfg = getattr(cfg, "linear_layer_config", cfg)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def _init_model(
        self, overrides: LayerStackConfig | None = None
    ) -> LinearLayerStack:
        return LinearLayerStack(self.main_cfg, overrides=overrides)


class AffineBiasTransformHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: "LinearLayerConfig",
    ):
        super().__init__(cfg)
        overrides = LayerStackConfig(input_dim=self.input_dim, output_dim=2)
        self.scalar_offset_generator = self._init_model(overrides)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        parameters = self.scalar_offset_generator(logits)
        bias_scaling_factor, bias_offset = parameters.chunk(2, dim=-1)
        return bias_scaling_factor * bias_params + bias_offset


class ElementwiseBiasHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: "LinearLayerConfig",
    ):
        super().__init__(cfg)
        overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.generator_model = self._init_model(overrides)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        parameters = self.generator_model(logits)
        return bias_params + parameters


class BiasGeneratorHandler(BiasHandlerAbstract):
    def __init__(
        self,
        cfg: "LinearLayerConfig",
    ):
        super().__init__(cfg)
        overrides = LayerStackConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        self.bias_generator = self._init_model(overrides)

    def forward(self, bias_params: None, logits: Tensor) -> Tensor | None:
        return self.bias_generator(logits)

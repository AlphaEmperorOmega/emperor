from torch import Tensor
from Emperor.base.utils import Module
from Emperor.base.layer import (
    LayerStackConfig,
    LinearLayerStack,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class DefaultBiasHandler(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
        bias_params: Tensor | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.bias_params = bias_params

    def forward(self, logits: Tensor) -> Tensor | None:
        return self.bias_params


class AffineBiasTransformHandler(DefaultBiasHandler):
    def __init__(
        self,
        cfg: "ModelConfig",
        bias_params: Tensor | None = None,
    ):
        super().__init__(cfg, bias_params)
        self.scalar_offset_generator = self.__init_scale_and_offset_model()

    def __init_scale_and_offset_model(self) -> LinearLayerStack:
        overrides = LayerStackConfig(output_dim=2)
        return LinearLayerStack(self.cfg, overrides)

    def forward(self, logits: Tensor) -> Tensor:
        parameters = self.scalar_offset_generator(logits)
        bias_scaling_factor, bias_offset = parameters.chunk(2, dim=-1)
        return bias_scaling_factor * self.bias_params + bias_offset


class ElementwiseBiasHandler(DefaultBiasHandler):
    def __init__(
        self,
        cfg: "ModelConfig",
        bias_params: Tensor | None = None,
    ):
        super().__init__(cfg, bias_params)
        self.generator_model = self.__init_scale_and_offset_model()

    def __init_scale_and_offset_model(self) -> LinearLayerStack:
        return LinearLayerStack(self.cfg)

    def forward(self, logits: Tensor) -> Tensor:
        parameters = self.generator_model(logits)
        return self.bias_params + parameters


class BiasGeneratorHandler(DefaultBiasHandler):
    def __init__(
        self,
        cfg: "ModelConfig",
        bias_params: None = None,
    ):
        super().__init__(cfg, bias_params)
        self.__ensure_no_bias_params_are_given()
        self.bias_generator = self.__init_bias_generator_model()

    def __ensure_no_bias_params_are_given(self):
        if self.bias_params is not None:
            raise ValueError(
                "bias_params must be None for this `BiasGeneratorHandler`."
            )

    def __init_bias_generator_model(self) -> LinearLayerStack:
        return LinearLayerStack(self.cfg)

    def forward(self, logits: Tensor) -> Tensor | None:
        return self.bias_generator(logits)

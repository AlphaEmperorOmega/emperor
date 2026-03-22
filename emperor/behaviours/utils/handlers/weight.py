import torch
import torch.nn.functional as F

from torch import Tensor
from emperor.base.utils import Module
from emperor.behaviours.utils.handlers.parameter import DepthMappingLayerStack
from emperor.base.layer import (
    LayerStackConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.behaviours.model import AdaptiveParameterBehaviourConfig


class OuterProductWeightHandler(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
        overrides: "AdaptiveParameterBehaviourConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterBehaviourConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.input_model = self.__init_input_model()
        self.output_model = self.__init_output_model()

    def __init_input_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return self.__init_generator_model(overrides)

    def __init_output_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self.__init_generator_model(overrides)

    def __init_generator_model(
        self, overrides: "LayerStackConfig"
    ) -> DepthMappingLayerStack:
        return DepthMappingLayerStack(self.cfg, overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        input_vectors = self.input_model(logits)
        output_vectors = self.output_model(logits)
        outer_product = self.__compute_outer_product(input_vectors, output_vectors)
        dynamic_params = self.__compute_dynamic_weights(outer_product)
        return weight_params + dynamic_params

    def __compute_dynamic_weights(self, outer_product: Tensor) -> Tensor:
        return outer_product.sum(dim=1)

    def __compute_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        input_vectors = self.__normalize_vectors(input_vectors)
        output_vectors = self.__normalize_vectors(output_vectors)
        outer_product = torch.einsum("bki,bkj->bkij", input_vectors, output_vectors)
        return self.__normalize_vectors(outer_product, True)
        # return torch.einsum("bki,bkj->bkij", input_vectors, output_vectors)

    def __normalize_vectors(
        self,
        outer_product: Tensor,
        normalize_flag: bool = False,
    ) -> Tensor:
        # TODO: Add flag to normalize the the input before or after the outer product
        # return torch.clamp(outer_product, -5.0, 5.0)
        if not normalize_flag:
            return outer_product
        return F.tanh(outer_product)

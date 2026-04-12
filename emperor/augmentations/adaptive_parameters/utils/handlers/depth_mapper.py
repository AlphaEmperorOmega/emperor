import torch

from torch import Tensor
from copy import deepcopy
from emperor.base.utils import Module
from dataclasses import dataclass, asdict, field
from emperor.base.layer import LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig

from emperor.augmentations.adaptive_parameters.utils.handlers._validator import (
    DepthMappingLayerValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.options import DynamicDepthOptions
    from emperor.augmentations.adaptive_parameters.config import (
        AdaptiveParameterAugmentationConfig,
    )


@dataclass
class DepthMappingLayerConfig(LinearLayerConfig):
    generator_depth: "DynamicDepthOptions | None" = field(
        default=None,
        metadata={
            "help": "Depth of the generator network that produces input-dependent weight adjustments."
        },
    )

    def build(self, overrides: "DepthMappingLayerConfig | None" = None) -> "Module":
        return DepthMappingLayer(self, overrides)


class DepthMappingLayer(Module):
    def __init__(
        self,
        cfg: "DepthMappingLayerConfig",
        overrides: "DepthMappingLayerConfig | None" = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        DepthMappingLayerValidator.validate(self)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.bias_flag: bool = self.cfg.bias_flag
        self.generator_depth: int = self.cfg.generator_depth.value

        self.__init_parameters()

    def __init_parameters(self):
        weight_shape = (self.generator_depth, self.input_dim, self.output_dim)
        self.weight_params = self._init_parameter_bank(weight_shape)
        self.bias_params = self.__init_bias_parameters()

    def __init_bias_parameters(self):
        if not self.bias_flag:
            return None
        bias_shape = (self.generator_depth, self.output_dim)
        return self._init_parameter_bank(bias_shape)

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = torch.einsum("bkj,kji->bki", X, self.weight_params)
        return self.__add_bias_parameters(X)

    def __add_bias_parameters(self, X: Tensor) -> Tensor:
        if self.bias_params is not None:
            return X + self.bias_params
        return X


class DepthMappingLayerStack(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.model_config: "LayerStackConfig" = self.__update_layer_stack_config()
        self.generator_depth = self.cfg.generator_depth.value
        self.generator_depth_value = self.cfg.generator_depth.value
        self.model = self.model_config.build()

    def __update_layer_stack_config(self) -> "LayerStackConfig":
        model_config = deepcopy(self.cfg.model_config)
        linear_config = asdict(model_config.layer_config.model_config)
        depth_mapping_config = DepthMappingLayerConfig(
            generator_depth=self.generator_depth, **linear_config
        )
        model_config.layer_config.model_config = depth_mapping_config
        return model_config

    def forward(self, input_batch: Tensor) -> Tensor:
        if not input_batch.dim() == 2:
            raise ValueError("Input batch must be a 2D tensor")

        input_batch = input_batch.unsqueeze(1)
        input_batch = input_batch.repeat(1, self.generator_depth_value, 1)
        return self.model(input_batch)

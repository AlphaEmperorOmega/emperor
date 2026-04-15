import torch

from torch import Tensor
from copy import deepcopy
from emperor.base.utils import Module
from emperor.base.layer.layer import Layer
from dataclasses import dataclass, asdict, field
from emperor.linears.core.config import LinearLayerConfig
from emperor.augmentations.adaptive_parameters.core.handlers._validator import (
    DepthMappingValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.base.layer import LayerStackConfig
    from emperor.augmentations.adaptive_parameters.options import DynamicDepthOptions


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
        DepthMappingValidator.validate(self)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.bias_flag: bool = self.cfg.bias_flag
        self.generator_depth: "DynamicDepthOptions" = self.cfg.generator_depth
        self.depth_value: int = self.generator_depth.value

        self.__init_parameters()

    def __init_parameters(self):
        weight_shape = (self.depth_value, self.input_dim, self.output_dim)
        self.weight_params = self._init_parameter_bank(weight_shape)
        self.bias_params = self.__init_bias_parameters()

    def __init_bias_parameters(self):
        if not self.bias_flag:
            return None
        bias_shape = (self.depth_value, self.output_dim)
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


@dataclass
class DepthMappingHandlerConfig(LinearLayerConfig):
    generator_depth: "DynamicDepthOptions | None" = field(
        default=None,
        metadata={
            "help": "Depth of the generator network that produces input-dependent weight adjustments."
        },
    )
    model_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={
            "help": "Layer stack configuration for the internal generator network."
        },
    )

    def build(self, overrides: "DepthMappingHandlerConfig | None" = None) -> "DepthMappingLayerStack":
        return DepthMappingLayerStack(self, overrides)


class DepthMappingLayerStack(Module):
    def __init__(
        self,
        cfg: "DepthMappingHandlerConfig",
        overrides: "DepthMappingHandlerConfig | None" = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.generator_depth: "DynamicDepthOptions" = self.cfg.generator_depth
        self.depth_value: int = self.generator_depth.value
        self.model_config: "LayerStackConfig" = self.__update_layer_stack_config()
        self.model = self.model_config.build()

    def __update_layer_stack_config(self) -> "LayerStackConfig":
        if isinstance(
            self.cfg.model_config.layer_config.layer_model_config,
            DepthMappingLayerConfig,
        ):
            return self.cfg.model_config
        model_config = deepcopy(self.cfg.model_config)
        DepthMappingValidator.validate_inner_model_is_linear_layer_config(model_config)
        DepthMappingValidator.validate_layer_config_has_no_gate_or_halting(model_config)
        linear_config = asdict(model_config.layer_config.layer_model_config)
        depth_mapping_config = DepthMappingLayerConfig(
            generator_depth=self.generator_depth, **linear_config
        )
        model_config.layer_config.layer_model_config = depth_mapping_config
        return model_config

    def forward(self, X: Tensor) -> Tensor:
        DepthMappingValidator.validate_input_is_2d(X)

        X = X.unsqueeze(1)
        X = X.repeat(1, self.depth_value, 1)
        return Layer.forward_with_state(self.model, X)

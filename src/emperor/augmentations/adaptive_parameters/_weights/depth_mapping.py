from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._weights.validation import (
    DepthMappingValidator,
)
from emperor.layers import Layer, LayerStackConfig
from emperor.linears import LinearLayerConfig
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._options import DynamicDepthOptions


@dataclass
class DepthMappingLayerConfig(LinearLayerConfig):
    generator_depth: "DynamicDepthOptions | None" = field(
        default=None,
        metadata={"help": "Generator depth for adaptive parameters."},
    )

    def _registry_owner(self) -> type:
        return DepthMappingLayer


class DepthMappingLayer(Module):
    VALIDATOR = DepthMappingValidator

    def __init__(
        self,
        cfg: "DepthMappingLayerConfig",
        overrides: "DepthMappingLayerConfig | None" = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.bias_flag: bool = self.cfg.bias_flag
        self.generator_depth: DynamicDepthOptions = self.cfg.generator_depth
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
        metadata={"help": "Generator depth for adaptive parameters."},
    )
    model_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": "Internal generator network config."},
    )

    def _registry_owner(self) -> type:
        return DepthMappingLayerStack


class DepthMappingLayerStack(Module):
    VALIDATOR = DepthMappingValidator

    def __init__(
        self,
        cfg: "DepthMappingHandlerConfig",
        overrides: "DepthMappingHandlerConfig | None" = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.generator_depth: DynamicDepthOptions = self.cfg.generator_depth
        self.depth_value: int = self.generator_depth.value
        self.model_config: LayerStackConfig = self.__build_depth_mapping_stack_config()
        self.model = self.model_config.build()

    def __build_depth_mapping_stack_config(self) -> LayerStackConfig:
        source = self.cfg.model_config
        self.__validate_source_stack_config(source)
        depth_mapping_config = self.__build_depth_mapping_layer_config(
            source.layer_config.layer_model_config
        )
        layer_config = replace(
            source.layer_config,
            layer_model_config=depth_mapping_config,
        )
        return replace(
            source,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            layer_config=layer_config,
        )

    def __validate_source_stack_config(self, source: LayerStackConfig) -> None:
        self.VALIDATOR.validate_inner_model_is_linear_layer_config(source)
        self.VALIDATOR.validate_layer_config_has_no_gate_or_halting(source)

    def __build_depth_mapping_layer_config(
        self,
        source: LinearLayerConfig,
    ) -> DepthMappingLayerConfig:
        if isinstance(source, DepthMappingLayerConfig):
            return replace(source, generator_depth=self.generator_depth)
        return DepthMappingLayerConfig(
            input_dim=source.input_dim,
            output_dim=source.output_dim,
            bias_flag=source.bias_flag,
            generator_depth=self.generator_depth,
        )

    def forward(self, X: Tensor) -> Tensor:
        self.VALIDATOR.validate_input_is_2d(X, self.input_dim)

        X = X.unsqueeze(1)
        X = X.repeat(1, self.depth_value, 1)
        return Layer.run_model_returning_hidden(self.model, X)

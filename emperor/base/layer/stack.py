from copy import deepcopy

from torch.nn import Sequential
from emperor.base.utils import Module
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from .config import LayerConfig, LayerStackConfig
from .layer import Layer
from ._validator import LayerStackValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LayerStack(Module):
    SHARED_INPUT_OUTPUT_DIM = 1
    SEPARATE_INPUT_OUTPUT_DIM = 2

    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "layer_stack_config", cfg)
        self.cfg: "LayerStackConfig" = self._override_config(config, overrides)
        LayerStackValidator.validate(self.cfg)

        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim
        self.num_layers: int = self.cfg.num_layers
        self.last_layer_bias_option: "LastLayerBiasOptions" = (
            self.cfg.last_layer_bias_option
        )
        self.apply_output_pipeline_flag: bool = self.cfg.apply_output_pipeline_flag
        self.layer_config: LayerConfig = self.cfg.layer_config

        self.layer_block_model = self.cfg.layer_type or Layer

    def build(self) -> Layer | Sequential:
        layers = []

        layer_adjustment = self.__add_initial_layer(layers)
        self.__add_hidden_layers(layers, layer_adjustment)
        self.__add_output_layer(layers)

        if len(layers) == 1:
            [model] = layers
            self._initialize_parameters(model)
            return model
        model = Sequential(*layers)
        self._initialize_parameters(model)
        return model

    def __add_initial_layer(self, layers: list) -> int:
        if self.input_dim != self.hidden_dim and self.num_layers > 1:
            layer = self.__create_layer(self.input_dim, self.hidden_dim)
            layers.append(layer)
            return self.SEPARATE_INPUT_OUTPUT_DIM
        return self.SHARED_INPUT_OUTPUT_DIM

    def __add_hidden_layers(self, layers: list, layer_adjustment: int) -> None:
        for _ in range(self.num_layers - layer_adjustment):
            layer = self.__create_layer(self.hidden_dim, self.hidden_dim)
            layers.append(layer)

    def __add_output_layer(self, layers: list) -> None:
        layer_input_dim = self.hidden_dim if self.num_layers > 1 else self.input_dim
        bias_overrides = self.__resolve_last_layer_bias_override()
        if self.apply_output_pipeline_flag:
            layer = self.__create_layer(
                layer_input_dim, self.output_dim, bias_overrides
            )
            layer.mark_as_last_layer()
            layers.append(layer)
            return
        overrides = LayerConfig(
            activation=ActivationOptions.DISABLED,
            residual_flag=False,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
        )
        if bias_overrides is not None:
            overrides = self._override_config(overrides, bias_overrides)
        layer = self.__create_layer(layer_input_dim, self.output_dim, overrides)
        layer.mark_as_last_layer()
        layers.append(layer)

    def __resolve_last_layer_bias_override(self) -> LayerConfig | None:
        if self.last_layer_bias_option == LastLayerBiasOptions.DEFAULT:
            return None
        if not hasattr(self.layer_config.layer_model_config, "bias_flag"):
            return None

        model_config = deepcopy(self.layer_config.layer_model_config)
        match self.last_layer_bias_option:
            case LastLayerBiasOptions.DISABLED:
                model_config.bias_flag = False
            case LastLayerBiasOptions.ENABLED:
                model_config.bias_flag = True
        return LayerConfig(layer_model_config=model_config)

    def __create_layer(
        self,
        input_dim: int,
        output_dim: int,
        overrides: LayerConfig | None = None,
    ) -> Layer:
        residual_flag = False if input_dim != output_dim else None
        dim_overrides = LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            residual_flag=residual_flag,
        )
        if overrides is not None:
            dim_overrides = self._override_config(dim_overrides, overrides)
        return self._override_config(self.layer_config, dim_overrides).build()

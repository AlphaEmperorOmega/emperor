from torch import Tensor
from torch.nn import Sequential
from emperor.base.layer import Layer, LayerStackConfig, LayerState
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer.config import LayerConfig
from emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.base.utils import Module
    from emperor.config import ModelConfig
    from emperor.base.options import ActivationOptions
    from emperor.base.options import LayerNormPositionOptions


class MixtureOfExpertsStack(Module):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "layer_stack_config", cfg)
        self.cfg: "LayerStackConfig" = self._override_config(config, overrides)

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
        if self.cfg.input_dim != self.cfg.hidden_dim and self.cfg.num_layers > 1:
            layers.append(self.__create_layer(self.cfg.input_dim, self.cfg.hidden_dim))
            return 2
        return 1

    def __add_hidden_layers(self, layers: list, layer_adjustment: int) -> None:
        for _ in range(self.cfg.num_layers - layer_adjustment):
            layers.append(self.__create_layer(self.cfg.hidden_dim, self.cfg.hidden_dim))

    def __add_output_layer(self, layers: list) -> None:
        layer_input_dim = self.cfg.hidden_dim if self.cfg.num_layers > 1 else self.cfg.input_dim
        overrides = LayerConfig(
            activation=ActivationOptions.DISABLED,
            residual_flag=False,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
        )
        layer = self.__create_layer(layer_input_dim, self.cfg.output_dim, overrides)
        layer.mark_as_last_layer()
        layers.append(layer)

    def __create_layer(
        self,
        input_dim: int,
        output_dim: int,
        overrides: LayerConfig | None = None,
    ) -> "MixtureOfExpertsLayer":
        residual_flag = False if input_dim != output_dim else None
        dim_overrides = LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            residual_flag=residual_flag,
        )
        if overrides is not None:
            dim_overrides = self._override_config(dim_overrides, overrides)
        cfg = self._override_config(self.cfg.layer_config, dim_overrides)
        return MixtureOfExpertsLayer(cfg)


class MixtureOfExpertsLayer(Layer):
    def __init__(
        self,
        cfg: "LayerConfig",
        overrides: "LayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.probabilities = None
        self.indices = None
        self.total_loss = None

    def __reset_properties(self) -> None:
        self.total_loss = None

    def forward(self, model_inputs: dict | tuple | Tensor) -> tuple | dict:
        residual = self._handle_model_input(model_inputs)
        output = self._handle_model_processing(residual)
        output = self.__maybe_apply_activation(output)
        output = self.__maybe_apply_residual_connection(output, residual)
        return self._handle_model_output(output)

    def _handle_model_input(self, model_inputs: dict | tuple | Tensor) -> Tensor:
        if isinstance(model_inputs, Tensor):
            return model_inputs
        if isinstance(model_inputs, tuple):
            model_inputs, total_loss = model_inputs
            if self.total_loss is None:
                self.total_loss = total_loss
            else:
                self.total_loss = self.total_loss + total_loss
            return model_inputs
        input_batch = model_inputs["input_batch"]
        self.probabilities = model_inputs["probabilities"]
        self.indices = model_inputs["indices"]
        self.total_loss = model_inputs["loss"]
        return input_batch

    def _handle_model_processing(self, model_input: Tensor, state: "LayerState | None" = None) -> Tensor:
        model_output, total_loss = self.model(
            model_input, self.probabilities, self.indices
        )
        if self.total_loss is None:
            self.total_loss = total_loss
        else:
            self.total_loss = self.total_loss + total_loss
        return model_output

    def __maybe_apply_activation(self, input: Tensor) -> Tensor:
        if self.has_activation:
            return self.activation_function(input)
        return input

    def __maybe_apply_residual_connection(self, input: Tensor, residual: Tensor) -> Tensor:
        if self.residual_flag:
            return input + residual
        return input

    def _handle_model_output(self, output: Tensor | LayerState) -> tuple | dict:
        if isinstance(output, LayerState):
            output = output.hidden
        total_loss = self.total_loss
        self.__reset_properties()
        if self.last_layer_flag or self.probabilities is None:
            return output, total_loss
        return {
            "input_batch": output,
            "probabilities": self.probabilities,
            "indices": self.indices,
            "loss": total_loss,
        }

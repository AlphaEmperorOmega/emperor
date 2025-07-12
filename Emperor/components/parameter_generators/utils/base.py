import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.nn import Linear, Sequential
from Emperor.base.utils import Module
from Emperor.base.utils import DataClassBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class LinearBlockStackConfig(DataClassBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the first `Linear` layer"},
    )
    hidden_dim: int | None = field(
        default=None,
        metadata={"help": "Dimension of the hidden `Linear` layers"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the output `Linear` layer"},
    )
    num_layers: int | None = field(
        default=None,
        metadata={"help": "Number of layers in the model"},
    )
    activation: nn.Linear | None = field(
        default=None,
        metadata={"help": "Activation function or layer to use"},
    )
    layer_norm_flag: int | None = field(
        default=None,
        metadata={"help": "Flag indicating whether to apply layer normalization"},
    )
    linear_model: nn.Module | None = field(
        default=None,
        metadata={"help": "Linear model module used for output transformation"},
    )


class LinearBlockStack(Module):
    def __init__(
        self,
        cfg: "LinearBlockStackConfig | ModelConfig",
        overrides: "LinearBlockStackConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_model_config", cfg)
        self.cfg: "LinearBlockStackConfig" = self._overwrite_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.output_dim = self.cfg.output_dim
        self.num_layers = self.cfg.num_layers
        self.activation = self.cfg.activation()
        self.layer_norm_flag = self.cfg.layer_norm_flag
        self.linear_model = self.cfg.linear_model

    def build_model(self) -> Linear | Sequential:
        layers = []

        layer_adjustment = self.__add_initial_layer(layers)
        self.__add_hidden_layers(layers, layer_adjustment)
        self.__add_output_layer(layers)

        model = Sequential(*layers)
        self._initialize_parameters(model)
        return model

    def __add_initial_layer(self, layers: list) -> int:
        if self.input_dim != self.hidden_dim and self.num_layers > 1:
            layer = self.__create_layer(self.input_dim, self.hidden_dim, False)
            layers.append(layer)
            return 2
        return 1

    def __add_hidden_layers(self, layers: list, layer_adjustment: int) -> None:
        for _ in range(self.num_layers - layer_adjustment):
            layer = self.__create_layer(self.hidden_dim, self.hidden_dim)
            layers.append(layer)

    def __add_output_layer(self, layers: list) -> None:
        layer_input = self.hidden_dim if self.num_layers > 1 else self.input_dim
        layers.append(Linear(layer_input, self.output_dim))

    def __create_layer(
        self,
        input_dim: int,
        output_dim: int,
        residual_connection_flag: bool = True,
    ):
        layer_norm_module = None
        if self.layer_norm_flag:
            layer_norm_module = nn.LayerNorm(output_dim)
        return LayerBlock(
            model=self.linear_model(
                input_dim,
                output_dim,
            ),
            activation_function_module=self.activation,
            layer_norm_module=layer_norm_module,
            residual_connection_flag=residual_connection_flag,
        )


class LayerBlock(Module):
    def __init__(
        self,
        model: "Module",
        activation_function_module: nn.Module | None = nn.ReLU(),
        layer_norm_module: nn.Module | None = None,
        residual_connection_flag: bool = False,
        is_adaptive_computation: bool = False,
    ):
        super().__init__()
        self.model = model
        self.activation_function_module = activation_function_module
        self.layer_norm_module = layer_norm_module
        self.residual_connection_flag = residual_connection_flag
        self.is_adaptive_computation = is_adaptive_computation

    def create_adaptive_computation_module(self):
        pass
        # TODO: In the future add a layer that can compute a
        # score for each token in the input, when the
        # sum of scores from the previews layers reaches 1
        # update the skip mask in order to make sure that the
        # further layers no longer process that token
        # and move it to the end of the model via resudual connection

    def forward(
        self,
        input_batch: torch.Tensor,
        skip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.model(input_batch)
        if self.layer_norm_module is not None:
            output = self.layer_norm_module(output)
        if self.activation_function_module is not None:
            output = self.activation_function_module(output)
        if self.residual_connection_flag:
            output = output + input_batch

        return output

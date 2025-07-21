import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.nn import Linear, Sequential
from torch.nn.modules import loss
from torch.types import Tensor
from Emperor.base.utils import Module
from Emperor.base.utils import DataClassBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.layers.utils.enums import ActivationFunctionOptions
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
        layer_norm_output_dim = None
        if self.layer_norm_flag:
            layer_norm_output_dim = output_dim
        return LayerBlock(
            model=self.linear_model(
                input_dim,
                output_dim,
            ),
            activation_function=self.activation,
            layer_norm_output_dim=layer_norm_output_dim,
            residual_connection_flag=residual_connection_flag,
        )


class LayerBlock(Module):
    def __init__(
        self,
        model: "Module",
        activation_function: "ActivationFunctionOptions | None" = None,
        layer_norm_output_dim: int | None = None,
        residual_connection_flag: bool = False,
        is_adaptive_computation: bool = False,
        dropout_probability: float = 0.0,
    ):
        super().__init__()

        self.model = model
        self.activation_function = activation_function
        self.layer_norm_output_dim = layer_norm_output_dim
        self.residual_connection_flag = residual_connection_flag
        self.is_adaptive_computation = is_adaptive_computation
        self.dropout_probability = dropout_probability

        self.has_activation = self.activation_function is not None
        self.has_dropout = self.dropout_probability > 0.0

        self.dropout_module = self.__init_dropout_module()
        self.layer_norm_module = self.__init_layer_norm_module()
        # self.block_data = LayerBlockOuputs()

    def __init_dropout_module(self) -> nn.Module | None:
        if self.has_dropout:
            return nn.Dropout(self.dropout_probability)
        return None

    def __init_layer_norm_module(self) -> nn.Module | None:
        if self.layer_norm_output_dim is not None:
            return nn.LayerNorm(self.layer_norm_output_dim)
        return None

    def create_adaptive_computation_module(self):
        pass

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # TODO: Ensure that the skip_maks will be used
        # in the future.
        output = self.model(input_batch)
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            output, skip_mask, loss = output

        if self.layer_norm_output_dim is not None:
            output = self.layer_norm_module(output)
        if self.has_activation:
            output = self.activation_function(output)
        if self.has_dropout:
            output = self.dropout_module(output)
        if self.residual_connection_flag:
            output = output + input_batch

        if is_tuple:
            return output, loss
        return output

    # TODO: In the future instead multiple function inputs
    # use a dataset class to encapsulate the inputs and outputs
    # of the block
    # def forward(
    #     self,
    #     data: LayerBlockData,
    # ) -> LayerBlockData:
    #     output = self.model(data.tensor)
    #     is_tuple = isinstance(output, tuple)
    #     if is_tuple:
    #         output, _, loss = output
    #         self.block_data.loss = loss
    #     if self.layer_norm_output_dim is not None:
    #         output = self.layer_norm_module(output)
    #     if self.has_activation:
    #         output = self.activation_function(output)
    #     if self.has_dropout:
    #         output = self.dropout_module(output)
    #     if self.residual_connection_flag:
    #         output = output + input_batch
    #     self.block_data.tensor = output
    #     return self.blcok_data


# @dataclass
# class LayerBlockData:
#     tensor: Tensor | None = None
#     skip_mask: Tensor | None = None
#     loss: Tensor | None = None

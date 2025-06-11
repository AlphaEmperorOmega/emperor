import copy
from typing import Callable, List
import torch.nn as nn
from torch import Tensor
from Emperor.base.utils import Module
from Emperor.base.models import Classifier

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.experiments.layers.layers_options import ParameterLayerOptions
    from Emperor.components.parameter_generators.layers import ParameterLayerBase


class ClassifierExperiment(Classifier):
    def __init__(
        self,
        model: "ParameterLayerBase",
        learning_rate: float = 0.1,
    ):
        super().__init__()
        self.lr = learning_rate
        self.model = model
        self.plotProgress = False


class SingleLayerClassifierModel(ClassifierExperiment):
    def __init__(
        self,
        model: "ParameterLayerBase",
        learning_rate: float = 0.1,
    ):
        super().__init__(learning_rate)
        self.model = nn.Sequential(nn.Flatten(), model)

    def forward(self, input_batch: Tensor):
        output = self.model(input_batch)
        auxiliary_loss = 0.0
        return output, auxiliary_loss


class MultiLayerClassifierModel(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
        hidden_layer_callback: Callable,
        learning_rate: float = 0.1,
        num_hidden_layers: int = 1,
        first_layer_preset: "ParameterLayerOptions | None" = None,
    ):
        super().__init__(learning_rate)
        self.cfg = cfg
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_callback = hidden_layer_callback
        self.first_layer_preset = first_layer_preset
        layers = self.__generate_model_layers()
        self.model = nn.Sequential(nn.Flatten(), *layers)

    def __create_layer_shapes(self) -> List:
        layer_shape = []
        layer_shape.append([self.cfg.input_dim, self.cfg.hidden_dim])
        for _ in range(self.num_hidden_layers):
            layer_shape.append([self.cfg.hidden_dim, self.cfg.hidden_dim])
        layer_shape.append([self.cfg.hidden_dim, self.cfg.output_dim])

        return layer_shape

    def __generate_model_layers(self) -> List:
        cfg = copy.deepcopy(self.cfg)
        layers = []
        for layer_idx, layer_shape in enumerate(self.__create_layer_shapes()):
            is_not_last_layer = layer_idx != (self.num_hidden_layers + 2 - 1)
            # model_type = self.model_type.value
            # if self.first_layer_preset is not None and layer_idx == 0:
            #     model_type = self.first_layer_preset.value
            model_preset = self.hidden_layer_callback(cfg, layer_shape)
            residual_flag = False if layer_idx == 0 else is_not_last_layer
            layer = LinearLayerDecorator(
                model=model_preset,
                activation_function=nn.ReLU() if is_not_last_layer else None,
                layer_norm_flag=is_not_last_layer,
                residual_connection_flag=residual_flag,
                output_dim=layer_shape[-1],
            )
            layers.append(layer)

        return layers

    def forward(self, input_batch: Tensor):
        output = self.model(input_batch)
        auxiliary_loss = 0.0
        return output, auxiliary_loss


class LinearLayerDecorator(Module):
    def __init__(
        self,
        model: "ParameterLayerBase",
        activation_function: nn.Module | None = nn.ReLU(),
        layer_norm_flag: bool = True,
        residual_connection_flag: bool = False,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.activation_function = activation_function
        self.layer_norm_flag = layer_norm_flag
        if self.layer_norm_flag:
            self.layer_norm = nn.LayerNorm(output_dim)
        self.residual_connection_flag = residual_connection_flag

    def forward(self, input_batch: Tensor) -> Tensor:
        output = self.model(input_batch)
        if self.layer_norm_flag:
            output = self.layer_norm(output)
        if self.activation_function is not None:
            output = self.activation_function(output)
        if self.residual_connection_flag:
            output = output + input_batch

        return output

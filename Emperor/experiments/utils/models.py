import copy
import torch
import torch.nn as nn

from typing import Callable
from torch import Tensor
from Emperor.base.layer import LinearLayerStack
from Emperor.base.models import Classifier


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.generators.utils.layers import ParameterLayerBase


class ClassifierExperiment(Classifier):
    def __init__(
        self,
        learning_rate: float = 0.1,
    ):
        super().__init__()
        self.lr = learning_rate
        self.plotProgress = False


class SingleLayerClassifierModel(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
        model: "ParameterLayerBase",
        learning_rate: float = 0.1,
    ):
        super().__init__(learning_rate)
        self.model = nn.Sequential(nn.Flatten(), model(cfg))

    def forward(self, input_batch: Tensor):
        output = self.model(input_batch)
        if isinstance(output, tuple):
            output_tensor, skip_mask, auxiliary_loss = output
            return output_tensor, auxiliary_loss
        return output, torch.tensor(0.0)


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
        # layers = self.__generate_model_layers()
        # self.model = nn.Sequential(nn.Flatten(), *layers)
        self.model = self.__generate_model_layers()

    def __generate_model_layers(self) -> nn.Linear | nn.Sequential:
        cfg = ModelConfig(
            input_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
            num_layers=5,
            activation=nn.ReLU,
            layer_norm_flag=False,
            model_type=nn.Linear,
        )
        return LinearLayerStack(cfg).build_model()

    def forward(self, input_batch: Tensor):
        output = self.model(input_batch)
        if isinstance(output, tuple):
            output_tensor, skip_mask, auxiliary_loss = output
            return output_tensor, auxiliary_loss

        return output, torch.tensor(0.0)


# def __create_layer_shapes(self) -> List:
#     layer_shape = []
#     layer_shape.append([self.cfg.input_dim, self.cfg.hidden_dim])
#     for _ in range(self.num_hidden_layers):
#         layer_shape.append([self.cfg.hidden_dim, self.cfg.hidden_dim])
#     layer_shape.append([self.cfg.hidden_dim, self.cfg.output_dim])
#
#     return layer_shape
#
# def __generate_model_layers(self) -> List:
#     cfg = copy.deepcopy(self.cfg)
#     layers = []
#     for layer_idx, layer_shape in enumerate(self.__create_layer_shapes()):
#         is_inner_layer_flag = layer_idx != (self.num_hidden_layers + 2 - 1)
#         # model_type = self.model_type.value
#         # if self.first_layer_preset is not None and layer_idx == 0:
#         #     model_type = self.first_layer_preset.value
#         _, output_dim = layer_shape
#         layer_block_inputs = (
#             self.hidden_layer_callback(cfg, layer_shape),
#             nn.ReLU() if is_inner_layer_flag else None,
#             nn.LayerNorm(output_dim) if is_inner_layer_flag else None,
#             False if layer_idx == 0 else is_inner_layer_flag,
#         )
#         layer = Layer(*layer_block_inputs)
#         layers.append(layer)
#     return layers

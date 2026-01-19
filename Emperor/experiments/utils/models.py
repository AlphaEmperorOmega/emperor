import torch
import torch.nn as nn

from torch import Tensor
from Emperor.base.layer import LayerStack
from Emperor.base.models import Classifier

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class ClassifierExperiment(Classifier):
    def __init__(
        self,
        cfg: "ModelConfig",
        model_type,
        learning_rate: float = 0.1,
        flatten_flag: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.lr = learning_rate
        self.model_type = model_type
        self.flatten_flag = flatten_flag
        self.plotProgress = False

        self.model = self.build()

    def build(self):
        model = self.model_type(self.cfg)
        if issubclass(self.model_type, LayerStack):
            model = model.build_model()
        if self.flatten_flag:
            return nn.Sequential(nn.Flatten(), model)
        return model

    def forward(self, input_batch: Tensor):
        output = self.model(input_batch)
        if isinstance(output, tuple):
            if len(output) == 3:
                output_tensor, skip_mask, auxiliary_loss = output
            else:
                output_tensor, auxiliary_loss = output
            return output_tensor, auxiliary_loss
        return output, torch.tensor(0.0)

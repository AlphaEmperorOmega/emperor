import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._biases.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters._biases.config import (
    SigmoidGatedDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
)
from emperor.layers import Layer


class SigmoidGatedDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: SigmoidGatedDynamicBiasConfig,
        overrides: SigmoidGatedDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.VALIDATOR.ensure_parameters_exist(bias_params)
        gate = torch.sigmoid(Layer.run_model_returning_hidden(self.model, logits))
        return bias_params * gate


class TanhGatedDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: TanhGatedDynamicBiasConfig,
        overrides: TanhGatedDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.VALIDATOR.ensure_parameters_exist(bias_params)
        gate = torch.tanh(Layer.run_model_returning_hidden(self.model, logits))
        return bias_params * gate

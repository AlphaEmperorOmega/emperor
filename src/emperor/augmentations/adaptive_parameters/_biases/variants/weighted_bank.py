import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._biases.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters._biases.config import (
    WeightedBankDynamicBiasConfig,
)
from emperor.layers import Layer


class WeightedBankDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: WeightedBankDynamicBiasConfig,
        overrides: WeightedBankDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.bank_expansion_factor = self.cfg.bank_expansion_factor.value
        weight_bank_shape = self.__get_weight_bank_shape()
        self.weight_bank = self._init_parameter_bank(weight_bank_shape)
        self.model = self._init_model(self.bank_expansion_factor)

    def __get_weight_bank_shape(self) -> tuple[int, int]:
        return self.bank_expansion_factor, self.output_dim

    def forward(self, _bias_params: Tensor, logits: Tensor) -> Tensor:
        bank_logits = Layer.run_model_returning_hidden(self.model, logits)
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        return torch.matmul(bank_distribution, self.weight_bank)

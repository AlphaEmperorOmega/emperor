import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicBiasValidator,
)
from emperor.augmentations.adaptive_parameters.core.bias.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters.core.bias.config import (
    WeightedBankDynamicBiasConfig,
)
from emperor.base.layer import Layer


class WeightedBankDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: WeightedBankDynamicBiasConfig,
        overrides: WeightedBankDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        DynamicBiasValidator.validate_bank_expansion_factor(self)
        self.bank_expansion_factor = self.cfg.bank_expansion_factor.value
        self.weight_bank = self._init_parameter_bank(
            (self.bank_expansion_factor, self.output_dim)
        )
        self.model = self._init_model(self.bank_expansion_factor)

    def forward(self, _bias_params: Tensor, logits: Tensor) -> Tensor:
        bank_logits = Layer.run_model_returning_hidden(self.model, logits)
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        return torch.matmul(bank_distribution, self.weight_bank)

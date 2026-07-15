import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters.core.weight.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters.core.weight.config import (
    SoftWeightedBankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight.depth_mapper import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)


class SoftWeightedBankDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: SoftWeightedBankDynamicWeightConfig,
        overrides: SoftWeightedBankDynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.VALIDATOR.validate_bank_expansion_factor(self)
        self.depth_value = self.generator_depth.value
        self.bank_expansion_factor = self.cfg.bank_expansion_factor.value
        self.weight_bank = self._init_parameter_bank(
            (
                self.depth_value,
                self.input_dim,
                self.bank_expansion_factor,
                self.output_dim,
            )
        )

        self.model = self._init_model()

    def _init_model(self) -> DepthMappingLayerStack:
        overrides = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim * self.bank_expansion_factor,
        )
        return super()._init_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        bank_logits = self.model(X)
        bank_logits = bank_logits.view(
            -1, self.depth_value, self.input_dim, self.bank_expansion_factor
        )

        bank_distribution = torch.softmax(bank_logits, dim=-1)
        compressed_params = torch.einsum(
            "bdik,diko->bdio", bank_distribution, self.weight_bank
        )

        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + compressed_params.sum(dim=1)

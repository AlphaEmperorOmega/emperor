import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicWeightValidator,
)
from emperor.augmentations.adaptive_parameters.core.weight.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters.core.weight.config import (
    LayeredWeightedBankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight.depth_mapper import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)


class LayeredWeightedBankDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: LayeredWeightedBankDynamicWeightConfig,
        overrides: LayeredWeightedBankDynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        DynamicWeightValidator.validate_bank_expansion_factor(self)
        self.bank_expansion_factor = self.cfg.bank_expansion_factor.value
        self.depth_value = self.generator_depth.value
        broadcast_batch = 1
        self.weight_bank = self._init_parameter_bank(
            (
                broadcast_batch,
                self.depth_value,
                self.bank_expansion_factor * self.input_dim,
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
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        bank_distribution_reshaped = bank_distribution.unsqueeze(dim=-1)
        batched_weighted_bank = self.weight_bank * bank_distribution_reshaped
        split_weights_by_factor = batched_weighted_bank.view(
            -1,
            self.depth_value,
            self.input_dim,
            self.bank_expansion_factor,
            self.output_dim,
        )
        depth_and_expansion_reduced_weights = split_weights_by_factor.sum(dim=(1, 3))
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + depth_and_expansion_reduced_weights

import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._weights.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters._weights.config import (
    SoftWeightedBankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
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
        self.depth_value = self.generator_depth.value
        self.bank_expansion_factor = self.cfg.bank_expansion_factor.value
        self.expanded_bank_row_count = self.input_dim * self.bank_expansion_factor
        weight_bank_shape = self.__get_weight_bank_shape()
        self.weight_bank = self._init_parameter_bank(weight_bank_shape)

        self.model = self._init_model()

    def __get_weight_bank_shape(self) -> tuple[int, int, int]:
        return (
            self.depth_value,
            self.expanded_bank_row_count,
            self.output_dim,
        )

    def _init_model(self) -> DepthMappingLayerStack:
        overrides = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim * self.expanded_bank_row_count,
        )
        return super()._init_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        flattened_weight_bank_mixture_logits = self.model(X)
        weight_bank_mixture_logits = (
            self.__reshape_logits_into_per_input_weight_bank_mixtures(
                flattened_weight_bank_mixture_logits
            )
        )

        weight_bank_mixture_weights = torch.softmax(weight_bank_mixture_logits, dim=-1)
        weight_bank_mixing_equation = "bdim,dmo->bdio"
        per_depth_weight_updates = torch.einsum(
            weight_bank_mixing_equation, weight_bank_mixture_weights, self.weight_bank
        )

        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        combined_weight_update = per_depth_weight_updates.sum(dim=1)
        return decayed_weight_params + combined_weight_update

    def __reshape_logits_into_per_input_weight_bank_mixtures(
        self,
        flattened_weight_bank_mixture_logits: Tensor,
    ) -> Tensor:
        per_input_weight_bank_mixture_shape = (
            -1,
            self.depth_value,
            self.input_dim,
            self.expanded_bank_row_count,
        )
        return flattened_weight_bank_mixture_logits.view(
            per_input_weight_bank_mixture_shape
        )

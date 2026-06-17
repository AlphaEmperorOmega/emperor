from torch import Tensor
from emperor.augmentations.adaptive_parameters.core.mask.base import AxisMaskAbstract
from emperor.augmentations.adaptive_parameters.core.mask.config import (
    PerAxisScoreMaskConfig,
)


class PerAxisScoreMask(AxisMaskAbstract):
    def __init__(
        self,
        cfg: PerAxisScoreMaskConfig,
        overrides: PerAxisScoreMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self._target_axis_count)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        axis_scores = self._compute_generator_soft_values(logits)
        hard_mask = self._compute_hard_mask(axis_scores)
        soft_mask = self._compute_soft_mask(axis_scores)
        return self._apply_hybrid_mask(weight_params, hard_mask, soft_mask)

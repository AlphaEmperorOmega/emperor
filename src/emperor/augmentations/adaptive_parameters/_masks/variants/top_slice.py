import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._masks.base import AxisMaskAbstract
from emperor.augmentations.adaptive_parameters._masks.config import (
    TopSliceAxisMaskConfig,
)


class TopSliceAxisMask(AxisMaskAbstract):
    def __init__(
        self,
        cfg: TopSliceAxisMaskConfig,
        overrides: TopSliceAxisMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self._target_axis_count)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        axis_scores = self._compute_generator_soft_values(logits)
        if self.mask_transition_width is not None and self.mask_transition_width > 1.0:
            transition_scores = self.__compute_transition_scores(axis_scores)
            hard_mask = self._compute_hard_mask(transition_scores)
            soft_mask = self._compute_soft_mask(axis_scores)
            return self._apply_hybrid_mask(weight_params, hard_mask, soft_mask)

        thresholded_axis_mask = self._compute_hard_mask(axis_scores)
        hard_mask = thresholded_axis_mask.cumprod(dim=-1)
        soft_mask = self._compute_soft_mask(axis_scores)
        return self._apply_hybrid_mask(weight_params, hard_mask, soft_mask)

    def __compute_transition_scores(self, axis_scores: Tensor) -> Tensor:
        binary_mask = self._compute_hard_mask(axis_scores)
        boundary_pos = binary_mask.cumprod(dim=-1).sum(dim=-1, keepdim=True)
        positions = torch.arange(
            axis_scores.shape[-1],
            device=axis_scores.device,
            dtype=axis_scores.dtype,
        )
        margins = boundary_pos - positions
        width = self.mask_transition_width
        return ((margins + width / 2.0) / width).clamp(0.0, 1.0)

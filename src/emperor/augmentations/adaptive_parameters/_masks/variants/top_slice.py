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
        gradual_transition_enabled = (
            self.mask_transition_width is not None and self.mask_transition_width > 1.0
        )
        if gradual_transition_enabled:
            transition_scores = self.__compute_transition_scores(axis_scores)
            transition_hard_mask = self._compute_hard_mask(transition_scores)
            axis_soft_mask = self._compute_soft_mask(axis_scores)
            return self._apply_hybrid_mask(
                weight_params,
                transition_hard_mask,
                axis_soft_mask,
            )

        thresholded_axis_mask = self._compute_hard_mask(axis_scores)
        top_slice_hard_mask = thresholded_axis_mask.cumprod(dim=-1)
        axis_soft_mask = self._compute_soft_mask(axis_scores)
        return self._apply_hybrid_mask(
            weight_params,
            top_slice_hard_mask,
            axis_soft_mask,
        )

    def __compute_transition_scores(self, axis_scores: Tensor) -> Tensor:
        thresholded_axis_mask = self._compute_hard_mask(axis_scores)
        leading_active_mask = thresholded_axis_mask.cumprod(dim=-1)
        boundary_position = leading_active_mask.sum(dim=-1, keepdim=True)
        axis_position_count = axis_scores.shape[-1]
        axis_positions = torch.arange(
            axis_position_count,
            device=axis_scores.device,
            dtype=axis_scores.dtype,
        )
        boundary_margins = boundary_position - axis_positions
        saturated_transition_width = self._saturate_scalar_to_dtype(
            self.mask_transition_width,
            boundary_margins,
            strictly_positive=True,
        )
        scaled_boundary_margins = boundary_margins / saturated_transition_width
        centered_transition_scores = scaled_boundary_margins + 0.5
        bounded_transition_scores = centered_transition_scores.clamp(0.0, 1.0)
        return bounded_transition_scores

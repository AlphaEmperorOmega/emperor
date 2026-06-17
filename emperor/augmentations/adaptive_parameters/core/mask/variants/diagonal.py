import torch

from torch import Tensor
from emperor.augmentations.adaptive_parameters.core.mask.base import AxisMaskAbstract
from emperor.augmentations.adaptive_parameters.core.mask.config import (
    DiagonalAxisMaskConfig,
)


class DiagonalAxisMask(AxisMaskAbstract):
    def __init__(
        self,
        cfg: DiagonalAxisMaskConfig,
        overrides: DiagonalAxisMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(1)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction = self._compute_generator_soft_values(logits)
        diagonal_scores = self.__compute_diagonal_scores(weight_params, keep_fraction)
        hard_mask = self._compute_hard_mask(diagonal_scores)
        soft_mask = self._compute_soft_mask(diagonal_scores)
        return self._apply_hybrid_mask(weight_params, hard_mask, soft_mask)

    def __compute_diagonal_scores(
        self,
        weight_params: Tensor,
        keep_fraction: Tensor,
    ) -> Tensor:
        row_count = weight_params.shape[-2]
        col_count = weight_params.shape[-1]
        row_indices = torch.arange(
            row_count, device=weight_params.device, dtype=keep_fraction.dtype
        )
        col_indices = torch.arange(
            col_count, device=weight_params.device, dtype=keep_fraction.dtype
        )
        min_diagonal_shift = 1 - row_count
        diagonal_shift = (
            keep_fraction.squeeze(-1) * (row_count + col_count) - row_count
        ).clamp(min=float(min_diagonal_shift))
        boundary = (row_count - 1 - row_indices).unsqueeze(0).unsqueeze(
            -1
        ) + diagonal_shift[:, None, None]
        margins = boundary - col_indices.unsqueeze(0).unsqueeze(0)
        width = (
            self.mask_transition_width
            if self.mask_transition_width is not None
            else 2.0
        )
        return ((margins + width / 2.0) / width).clamp(0.0, 1.0)

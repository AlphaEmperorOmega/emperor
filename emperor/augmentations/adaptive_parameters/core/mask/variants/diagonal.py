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
        row_count, column_count = weight_params.shape[-2:]
        device = weight_params.device
        dtype = keep_fraction.dtype
        row_positions, column_positions = self.__create_axis_positions(
            row_count, column_count, device=device, dtype=dtype
        )
        diagonal_shift = self.__compute_diagonal_shift(
            keep_fraction, row_count, column_count
        )
        margins = self.__compute_diagonal_margins(
            row_positions, column_positions, row_count, diagonal_shift
        )
        return self.__convert_margins_to_transition_scores(margins)

    @staticmethod
    def __create_axis_positions(
        row_count: int,
        column_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        row_positions = torch.arange(row_count, device=device, dtype=dtype)
        column_positions = torch.arange(column_count, device=device, dtype=dtype)
        return row_positions, column_positions

    @staticmethod
    def __compute_diagonal_shift(
        keep_fraction: Tensor,
        row_count: int,
        column_count: int,
    ) -> Tensor:
        min_diagonal_shift = 1 - row_count
        axis_count = row_count + column_count
        scaled_keep_fraction = keep_fraction.squeeze(-1) * axis_count
        diagonal_shift = scaled_keep_fraction - row_count
        return diagonal_shift.clamp(min=float(min_diagonal_shift))

    @staticmethod
    def __compute_diagonal_margins(
        row_positions: Tensor,
        column_positions: Tensor,
        row_count: int,
        diagonal_shift: Tensor,
    ) -> Tensor:
        reversed_row_positions = row_count - 1 - row_positions
        row_boundaries = reversed_row_positions.unsqueeze(0).unsqueeze(-1)
        diagonal_boundary = row_boundaries + diagonal_shift[:, None, None]
        return diagonal_boundary - column_positions.unsqueeze(0).unsqueeze(0)

    def __convert_margins_to_transition_scores(self, margins: Tensor) -> Tensor:
        transition_width = (
            self.mask_transition_width
            if self.mask_transition_width is not None
            else 2.0
        )
        half_transition_width = transition_width / 2.0
        shifted_margins = margins + half_transition_width
        transition_scores = shifted_margins / transition_width
        return transition_scores.clamp(0.0, 1.0)

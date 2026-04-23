import torch

from torch import Tensor
from torch.nn import Sequential
from dataclasses import dataclass
from emperor.base.registry import subclass_registry
from emperor.base.layer import Layer, LayerStackConfig
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.augmentations.adaptive_parameters.core._validator import RowMaskValidator
from emperor.augmentations.adaptive_parameters.options import (
    MaskDimensionOptions,
    RowMaskOptions,
)


@dataclass
class RowMaskConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input dimensionality of the row mask module."
    )
    output_dim: int | None = optional_field(
        "Output dimensionality of the row mask module."
    )
    model_type: RowMaskOptions | None = optional_field(
        "Row masking strategy applied to the weight matrix after dynamic updates."
    )
    mask_dimension_option: MaskDimensionOptions | None = optional_field(
        "Specifies whether masking is applied across rows or columns of the weight matrix."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Configuration for the internal generator network."
    )

    def _registry_owner(self) -> type:
        return RowMaskAbstract


@subclass_registry
class RowMaskAbstract(Module):
    MASK_SURROGATE_SCALE = 5.0

    def __init__(
        self,
        cfg: RowMaskConfig,
        overrides: RowMaskConfig | None = None,
    ):
        super().__init__()
        self.cfg: RowMaskConfig = self._override_config(cfg, overrides)
        RowMaskValidator.validate(self)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.mask_dimension_option = self.cfg.mask_dimension_option
        self.model_config = self.cfg.model_config

    def _init_generator(self, output_dim: int) -> "Layer | Sequential":
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=output_dim,
        )
        generator_model = self.model_config.build(overrides)
        RowMaskValidator.validate_generator_model(generator_model)
        return generator_model

    def _run_generator(
        self, generator_model: "Layer | Sequential", logits: Tensor
    ) -> Tensor:
        return Layer.forward_with_state(generator_model, logits)

    @property
    def _count_dim(self) -> int:
        return -1 if self.mask_dimension_option == MaskDimensionOptions.COLUMN else -2

    @property
    def _magnitude_dim(self) -> int:
        return -2 if self.mask_dimension_option == MaskDimensionOptions.COLUMN else -1

    @property
    def _broadcast_dim(self) -> int:
        return -2 if self.mask_dimension_option == MaskDimensionOptions.COLUMN else -1

    @property
    def _mask_count(self) -> int:
        return (
            self.output_dim
            if self.mask_dimension_option == MaskDimensionOptions.COLUMN
            else self.input_dim
        )

    def _resolve_training_mask(self, hard_mask: Tensor, soft_mask: Tensor) -> Tensor:
        if not self.training:
            return hard_mask
        return hard_mask + soft_mask - soft_mask.detach()

    def _apply_weight_straight_through(
        self, sparsified_weights: Tensor, weight_params: Tensor
    ) -> Tensor:
        if self.training:
            return sparsified_weights + (weight_params - weight_params.detach())
        return sparsified_weights


@RowMaskAbstract.register(RowMaskOptions.GLOBAL_SCORE)
class GlobalScoreRowMask(RowMaskAbstract):
    def __init__(
        self,
        cfg: RowMaskConfig,
        overrides: RowMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self) -> "Layer | Sequential":
        return self._init_generator(1)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logit = self._run_generator(self.score_generator, logits)
        keep_fraction = torch.sigmoid(keep_fraction_logit)
        total_count = weight_params.shape[self._count_dim]
        continuous_items_to_keep = torch.clamp(
            keep_fraction.squeeze(-1) * total_count,
            min=1.0,
            max=float(total_count),
        )
        items_to_keep = torch.clamp(
            (keep_fraction * total_count).long(), min=1
        ).squeeze(-1)
        magnitudes = weight_params.norm(dim=self._magnitude_dim)
        hard_mask = self.__build_hard_mask(magnitudes, items_to_keep)
        soft_mask = self.__build_soft_mask(magnitudes, continuous_items_to_keep)
        effective_mask = self._resolve_training_mask(hard_mask, soft_mask)
        sparsified_weights = weight_params * effective_mask.unsqueeze(self._broadcast_dim)
        return self._apply_weight_straight_through(sparsified_weights, weight_params)

    def __build_hard_mask(
        self,
        row_magnitudes: Tensor,
        rows_to_keep: Tensor,
    ) -> Tensor:
        sorted_magnitudes, _ = row_magnitudes.sort(dim=-1, descending=True)
        thresholds = sorted_magnitudes.gather(-1, (rows_to_keep - 1).unsqueeze(-1))
        return (row_magnitudes >= thresholds).float()

    def __build_soft_mask(
        self,
        row_magnitudes: Tensor,
        rows_to_keep: Tensor,
    ) -> Tensor:
        rank_indices = row_magnitudes.argsort(dim=-1, descending=True).argsort(dim=-1)
        ranks = rank_indices.float() + 1.0
        scaled_distance = rows_to_keep.unsqueeze(-1) - ranks + 0.5
        return torch.sigmoid(self.MASK_SURROGATE_SCALE * scaled_distance)


@RowMaskAbstract.register(RowMaskOptions.PER_ROW_SCORE)
class PerRowScoreRowMask(RowMaskAbstract):
    def __init__(
        self,
        cfg: RowMaskConfig,
        overrides: RowMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self) -> "Layer | Sequential":
        return self._init_generator(self._mask_count)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logits = self._run_generator(self.score_generator, logits)
        keep_fractions = torch.sigmoid(keep_fraction_logits)
        hard_mask = (keep_fractions >= 0.5).float()
        effective_mask = self._resolve_training_mask(hard_mask, keep_fractions)
        sparsified_weights = weight_params * effective_mask.unsqueeze(self._broadcast_dim)
        return self._apply_weight_straight_through(sparsified_weights, weight_params)


@RowMaskAbstract.register(RowMaskOptions.TOP_SLICE)
class TopSliceRowMask(RowMaskAbstract):
    def __init__(
        self,
        cfg: RowMaskConfig,
        overrides: RowMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self) -> "Layer | Sequential":
        return self._init_generator(1)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logit = self._run_generator(self.score_generator, logits)
        keep_fraction = torch.sigmoid(keep_fraction_logit)
        total_count = weight_params.shape[self._count_dim]
        continuous_items_to_keep = torch.clamp(
            keep_fraction * total_count,
            min=1.0,
            max=float(total_count),
        )
        items_to_keep = torch.clamp((keep_fraction * total_count).long(), min=1)
        indices = torch.arange(total_count, device=weight_params.device)
        hard_mask = (indices.unsqueeze(0) < items_to_keep).float()
        soft_mask = torch.sigmoid(
            self.MASK_SURROGATE_SCALE
            * (continuous_items_to_keep - indices.unsqueeze(0) + 0.5)
        )
        effective_mask = self._resolve_training_mask(hard_mask, soft_mask)
        sparsified_weights = weight_params * effective_mask.unsqueeze(self._broadcast_dim)
        return self._apply_weight_straight_through(sparsified_weights, weight_params)


@RowMaskAbstract.register(RowMaskOptions.DIAGONAL)
class DiagonalRowMask(RowMaskAbstract):
    def __init__(
        self,
        cfg: RowMaskConfig,
        overrides: RowMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self) -> "Layer | Sequential":
        return self._init_generator(1)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logit = self._run_generator(self.score_generator, logits)
        keep_fraction = torch.sigmoid(keep_fraction_logit)
        row_count = weight_params.shape[-2]
        col_count = weight_params.shape[-1]
        min_diagonal_shift = 1 - row_count
        continuous_diagonal_shift = (
            keep_fraction.squeeze(-1) * (row_count + col_count) - row_count
        ).clamp(min=min_diagonal_shift)
        raw_diagonal_shift = (keep_fraction * (row_count + col_count)).long().squeeze(
            -1
        ) - row_count
        diagonal_shift = raw_diagonal_shift.clamp(min=min_diagonal_shift)
        row_indices = torch.arange(row_count, device=weight_params.device)
        col_indices = torch.arange(col_count, device=weight_params.device)
        hard_mask = (
            col_indices.unsqueeze(0)
            <= (row_count - 1 - row_indices).unsqueeze(1)
            + diagonal_shift[..., None, None]
        ).float()
        soft_boundary = (
            (row_count - 1 - row_indices).unsqueeze(1)
            + continuous_diagonal_shift[..., None, None]
        )
        soft_mask = torch.sigmoid(
            self.MASK_SURROGATE_SCALE * (soft_boundary - col_indices.unsqueeze(0) + 0.5)
        )
        effective_mask = self._resolve_training_mask(hard_mask, soft_mask)
        sparsified_weights = weight_params * effective_mask
        return self._apply_weight_straight_through(sparsified_weights, weight_params)

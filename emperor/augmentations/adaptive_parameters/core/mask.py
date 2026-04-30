import torch

from torch import Tensor
from torch.nn import Sequential
from dataclasses import dataclass, field
from emperor.base.registry import subclass_registry
from emperor.base.layer import Layer, LayerStackConfig
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.augmentations.adaptive_parameters.core._validator import AxisMaskValidator
from emperor.augmentations.adaptive_parameters.options import (
    AxisMaskOptions,
    MaskDimensionOptions,
)


@dataclass
class AxisMaskConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input dimensionality of the axis mask module."
    )
    output_dim: int | None = optional_field(
        "Output dimensionality of the axis mask module."
    )
    model_type: AxisMaskOptions | None = optional_field(
        "Axis masking strategy applied to the weight matrix after dynamic updates."
    )
    mask_dimension_option: MaskDimensionOptions | None = optional_field(
        "Specifies whether masking is applied across rows or columns of the weight matrix."
    )
    mask_threshold: float | None = optional_field(
        "Threshold applied to axis keep scores when deciding whether to preserve a row or column."
    )
    mask_surrogate_scale: float | None = optional_field(
        "Steepness factor for the training-time sigmoid surrogate used around the axis mask threshold. "
        "Set to 0.0 to bypass the surrogate and use the raw axis scores directly."
    )
    mask_floor: float | None = optional_field(
        "Baseline mask value assigned to structurally dropped regions. "
        "Use 0.0 to keep exact zeros, or a small positive value to attenuate "
        "dropped rows, columns, or diagonal regions instead of removing them fully."
    )
    mask_transition_width: float | None = optional_field(
        "Width of the smooth transition zone around the mask boundary for TOP_SLICE. "
        "When set to 1 or below, the standard cumulative product is used. "
        "When greater than 1, a margin-based gradient replaces the hard cutoff."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Configuration for the internal generator network."
    )

    def _registry_owner(self) -> type:
        return AxisMaskAbstract


@subclass_registry
class AxisMaskAbstract(Module):
    def __init__(
        self,
        cfg: AxisMaskConfig,
        overrides: AxisMaskConfig | None = None,
    ):
        super().__init__()
        self.cfg: AxisMaskConfig = self._override_config(cfg, overrides)
        AxisMaskValidator.validate(self)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.mask_dimension_option: AxisMaskAbstract = self.cfg.mask_dimension_option
        self.mask_threshold: float = self.cfg.mask_threshold
        self.mask_surrogate_scale: float = self.cfg.mask_surrogate_scale
        self.mask_floor: float = self.cfg.mask_floor
        self.mask_transition_width: float = self.cfg.mask_transition_width
        self.model_config: LayerStackConfig = self.cfg.model_config

    def _init_generator(self, output_dim: int) -> "Layer | Sequential":
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=output_dim,
        )
        generator_model = self.model_config.build(overrides)
        AxisMaskValidator.validate_generator_model(generator_model)
        return generator_model

    @property
    def _target_axis_count(self) -> int:
        return (
            self.input_dim
            if self.mask_dimension_option == MaskDimensionOptions.ROW
            else self.output_dim
        )

    @property
    def _source_axis_count(self) -> int:
        return (
            self.input_dim
            if self.mask_dimension_option == MaskDimensionOptions.COLUMN
            else self.output_dim
        )

    @property
    def __target_broadcast_dim(self) -> int:
        return -2 if self.mask_dimension_option == MaskDimensionOptions.COLUMN else -1

    @property
    def __source_broadcast_dim(self) -> int:
        return -1 if self.mask_dimension_option == MaskDimensionOptions.COLUMN else -2

    @property
    def __score_dim(self) -> int:
        return -2 if self.mask_dimension_option == MaskDimensionOptions.COLUMN else -1

    def _compute_generator_soft_values(
        self, generator_model: "Layer | Sequential", logits: Tensor
    ) -> Tensor:
        mask_logits = Layer.forward_with_state(generator_model, logits)
        return torch.sigmoid(mask_logits)

    def _compute_masked_weight_scores(
        self,
        weight_params: Tensor,
        soft_axis_mask: Tensor,
    ) -> Tensor:
        masked_weights = weight_params * soft_axis_mask.unsqueeze(
            self.__source_broadcast_dim
        )
        return masked_weights.abs().mean(dim=self.__score_dim)

    def _compute_hard_mask(self, scores: Tensor) -> Tensor:
        return (scores >= self.mask_threshold).to(dtype=scores.dtype)

    def _compute_soft_mask(self, scores: Tensor) -> Tensor:
        if self.mask_surrogate_scale == 0.0:
            return scores.clamp(0.0, 1.0)
        return torch.sigmoid(self.mask_surrogate_scale * (scores - self.mask_threshold))

    def _apply_hybrid_mask(
        self,
        weight_params: Tensor,
        hard_mask: Tensor,
        soft_mask: Tensor,
    ) -> Tensor:
        mask_floor = hard_mask.new_tensor(self.mask_floor)
        adjusted_hard_mask = (mask_floor + (1.0 - mask_floor) * hard_mask).clamp(
            min=mask_floor, max=1.0
        )
        final_mask = adjusted_hard_mask * soft_mask
        if weight_params.dim() == 2:
            weight_params = weight_params.unsqueeze(dim=0)
        if final_mask.dim() == 3 and weight_params.dim() == 3:
            return weight_params * final_mask
        return weight_params * final_mask.unsqueeze(self.__target_broadcast_dim)


@AxisMaskAbstract.register(AxisMaskOptions.WEIGHT_INFORMED_SCORE)
class WeightInformedScoreAxisMask(AxisMaskAbstract):
    def __init__(
        self,
        cfg: AxisMaskConfig,
        overrides: AxisMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self) -> "Layer | Sequential":
        return self._init_generator(self._source_axis_count)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        source_axis_soft_mask = self._compute_generator_soft_values(
            self.score_generator, logits
        )
        axis_scores = self._compute_masked_weight_scores(
            weight_params, source_axis_soft_mask
        )
        hard_mask = self._compute_hard_mask(axis_scores)
        soft_mask = self._compute_soft_mask(axis_scores)
        return self._apply_hybrid_mask(weight_params, hard_mask, soft_mask)


@AxisMaskAbstract.register(AxisMaskOptions.PER_AXIS_SCORE)
class PerAxisScoreMask(AxisMaskAbstract):
    def __init__(
        self,
        cfg: AxisMaskConfig,
        overrides: AxisMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self) -> "Layer | Sequential":
        return self._init_generator(self._target_axis_count)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        axis_scores = self._compute_generator_soft_values(self.score_generator, logits)
        hard_mask = self._compute_hard_mask(axis_scores)
        soft_mask = self._compute_soft_mask(axis_scores)
        return self._apply_hybrid_mask(weight_params, hard_mask, soft_mask)


@AxisMaskAbstract.register(AxisMaskOptions.TOP_SLICE)
class TopSliceAxisMask(AxisMaskAbstract):
    def __init__(
        self,
        cfg: AxisMaskConfig,
        overrides: AxisMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self) -> "Layer | Sequential":
        return self._init_generator(self._target_axis_count)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        axis_scores = self._compute_generator_soft_values(self.score_generator, logits)
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


@AxisMaskAbstract.register(AxisMaskOptions.DIAGONAL)
class DiagonalAxisMask(AxisMaskAbstract):
    def __init__(
        self,
        cfg: AxisMaskConfig,
        overrides: AxisMaskConfig | None = None,
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
        keep_fraction = self._compute_generator_soft_values(
            self.score_generator, logits
        )
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

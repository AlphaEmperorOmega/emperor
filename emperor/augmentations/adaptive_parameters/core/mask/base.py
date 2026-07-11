import torch

from torch import Tensor
from emperor.base.layer import Layer, LayerStack, LayerStackConfig
from emperor.base.module import Module
from emperor.augmentations.adaptive_parameters.core._validator import AxisMaskValidator
from emperor.augmentations.adaptive_parameters.core.mask.config import AxisMaskConfig
from emperor.augmentations.adaptive_parameters.options import MaskDimensionOptions


class AxisMaskAbstract(Module):
    def __init__(self, cfg: AxisMaskConfig, overrides: AxisMaskConfig | None = None):
        super().__init__()
        self.cfg: AxisMaskConfig = self._override_config(cfg, overrides)
        AxisMaskValidator.validate(self)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.mask_threshold: float = self.cfg.mask_threshold
        self.mask_surrogate_scale: float = self.cfg.mask_surrogate_scale
        self.mask_floor: float = self.cfg.mask_floor
        self.model_config: LayerStackConfig = self.cfg.model_config

    def _init_model(self, output_dim: int) -> "Layer | LayerStack":
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=output_dim,
        )
        generator_model = self.model_config.build(overrides)
        AxisMaskValidator.validate_generator_model(generator_model)
        return generator_model

    @property
    def mask_dimension_option(self) -> MaskDimensionOptions | None:
        return getattr(self.cfg, "mask_dimension_option", None)

    @property
    def mask_transition_width(self) -> float | None:
        return getattr(self.cfg, "mask_transition_width", None)

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

    def _compute_generator_soft_values(self, logits: Tensor) -> Tensor:
        mask_logits = Layer.run_model_returning_hidden(self.model, logits)
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

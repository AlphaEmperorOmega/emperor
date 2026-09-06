"""Configuration-owned weight normalization with trainable scale and clamp limit."""

import torch
from torch import Tensor, nn

from emperor.augmentations.adaptive_parameters._options import (
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._weights.config import (
    DynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters._weights.validation import (
    DynamicWeightValidator,
)


class WeightNormalizationPolicy(nn.Module):
    VALIDATOR = DynamicWeightValidator

    def __init__(self, cfg: DynamicWeightConfig):
        super().__init__()
        self.cfg = cfg
        self.normalization_option = getattr(
            self.cfg, "normalization_option", WeightNormalizationOptions.DISABLED
        )
        self.normalization_position_option = getattr(
            self.cfg,
            "normalization_position_option",
            WeightNormalizationPositionOptions.DISABLED,
        )
        self.VALIDATOR.validate_normalization_option(self.normalization_option)
        self.VALIDATOR.validate_normalization_position_option(
            self.normalization_position_option
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.clamp_limit = nn.Parameter(torch.tensor(1.0))

    def normalize_before_outer_product(self, vectors: Tensor) -> Tensor:
        self.VALIDATOR.validate_normalization_position_option(
            self.normalization_position_option
        )
        if (
            self.normalization_position_option
            is WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
        ):
            return self(vectors)
        return vectors

    def normalize_after_outer_product(self, outer_product: Tensor) -> Tensor:
        self.VALIDATOR.validate_normalization_position_option(
            self.normalization_position_option
        )
        if (
            self.normalization_position_option
            is WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT
        ):
            return self(outer_product)
        return outer_product

    def forward(self, vectors: Tensor) -> Tensor:
        self.VALIDATOR.validate_normalization_option(self.normalization_option)
        match self.normalization_option:
            case WeightNormalizationOptions.CLAMP:
                return self.__apply_symmetric_clamp(vectors)
            case WeightNormalizationOptions.L2_SCALE:
                return self.__apply_stable_l2_normalization(vectors) * self.scale
            case WeightNormalizationOptions.SOFT_CLAMP:
                return self.__apply_stable_soft_clamp(vectors)
            case WeightNormalizationOptions.RMS:
                return self.__apply_stable_rms_normalization(vectors) * self.scale
            case WeightNormalizationOptions.SIGMOID_SCALE:
                return (torch.sigmoid(vectors) * 2 - 1) * self.scale
            case WeightNormalizationOptions.DISABLED:
                return vectors

    def __apply_symmetric_clamp(self, vectors: Tensor) -> Tensor:
        clamp_limit_magnitude = self.clamp_limit.abs()
        return torch.clamp(
            vectors,
            -clamp_limit_magnitude,
            clamp_limit_magnitude,
        )

    def __apply_stable_l2_normalization(self, vectors: Tensor) -> Tensor:
        (
            accumulator_vectors,
            maximum_magnitude,
            magnitude_scaled_vectors,
            contains_nonzero_value,
        ) = self.__scale_vectors_by_maximum_magnitude(vectors)
        squared_magnitude_scaled_vectors = magnitude_scaled_vectors.square()
        scaled_squared_l2_norm = squared_magnitude_scaled_vectors.sum(
            dim=-1,
            keepdim=True,
        )
        stable_scaled_squared_l2_norm = torch.where(
            contains_nonzero_value,
            scaled_squared_l2_norm,
            torch.ones_like(scaled_squared_l2_norm),
        )
        scaled_l2_norm = stable_scaled_squared_l2_norm.sqrt()
        l2_norm = maximum_magnitude * scaled_l2_norm

        finite_l2_norm = torch.isfinite(l2_norm)
        safe_l2_norm = torch.where(
            finite_l2_norm,
            l2_norm,
            torch.ones_like(l2_norm),
        )
        minimum_l2_norm = max(1e-12, torch.finfo(vectors.dtype).tiny)
        normalized_by_l2_norm = accumulator_vectors / safe_l2_norm.clamp_min(
            minimum_l2_norm
        )
        normalized_by_scaled_l2_norm = magnitude_scaled_vectors / scaled_l2_norm
        normalized_vectors = torch.where(
            finite_l2_norm, normalized_by_l2_norm, normalized_by_scaled_l2_norm
        )
        return normalized_vectors.to(dtype=vectors.dtype)

    @staticmethod
    def __scale_vectors_by_maximum_magnitude(
        vectors: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if vectors.dtype in (torch.float16, torch.bfloat16):
            accumulator_vectors = vectors.float()
        else:
            accumulator_vectors = vectors
        maximum_magnitude = accumulator_vectors.abs().amax(
            dim=-1,
            keepdim=True,
        )
        contains_nonzero_value = maximum_magnitude > 0
        safe_maximum_magnitude = torch.where(
            contains_nonzero_value,
            maximum_magnitude,
            torch.ones_like(maximum_magnitude),
        )
        magnitude_scaled_vectors = accumulator_vectors / safe_maximum_magnitude
        return (
            accumulator_vectors,
            maximum_magnitude,
            magnitude_scaled_vectors,
            contains_nonzero_value,
        )

    def __apply_stable_soft_clamp(self, vectors: Tensor) -> Tensor:
        clamp_limit_magnitude = self.clamp_limit.abs()
        minimum_safe_denominator = torch.finfo(vectors.dtype).eps
        safe_clamp_denominator = clamp_limit_magnitude.clamp_min(
            minimum_safe_denominator
        )
        scaled_vectors = vectors / safe_clamp_denominator
        return clamp_limit_magnitude * torch.tanh(scaled_vectors)

    def __apply_stable_rms_normalization(self, vectors: Tensor) -> Tensor:
        (
            accumulator_vectors,
            maximum_magnitude,
            magnitude_scaled_vectors,
            contains_nonzero_value,
        ) = self.__scale_vectors_by_maximum_magnitude(vectors)
        scaled_squared_mean = magnitude_scaled_vectors.square().mean(
            dim=-1,
            keepdim=True,
        )
        stable_scaled_squared_mean = torch.where(
            contains_nonzero_value,
            scaled_squared_mean,
            torch.ones_like(scaled_squared_mean),
        )
        scaled_root_mean_square = stable_scaled_squared_mean.sqrt()
        root_mean_square = maximum_magnitude * scaled_root_mean_square
        minimum_root_mean_square = max(1e-8, torch.finfo(vectors.dtype).tiny)
        normalized_vectors = accumulator_vectors / (
            root_mean_square + minimum_root_mean_square
        )
        return normalized_vectors.to(dtype=vectors.dtype)

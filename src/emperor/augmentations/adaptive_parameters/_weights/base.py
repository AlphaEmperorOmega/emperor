import torch
import torch.nn as nn
from torch import Tensor

from emperor.augmentations.adaptive_parameters._options import (
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._weights.config import (
    DynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters._weights.validation import (
    DynamicWeightValidator,
)
from emperor.nn import Module


class DynamicWeightAbstract(Module):
    VALIDATOR = DynamicWeightValidator

    def __init__(
        self,
        cfg: "DynamicWeightConfig",
        overrides: "DynamicWeightConfig | None" = None,
    ):
        super().__init__()
        self.cfg: DynamicWeightConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.generator_depth = self.cfg.generator_depth
        self.decay_schedule_option = self.cfg.decay_schedule
        self.decay_rate = self.cfg.decay_rate
        self.decay_warmup_batches = self.cfg.decay_warmup_batches or 0
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.clamp_limit = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("decay_step", torch.zeros(1))
        self.register_buffer("warmup_step", torch.zeros(1))

    def _init_model(
        self, overrides: "DepthMappingHandlerConfig"
    ) -> DepthMappingLayerStack:
        return DepthMappingLayerStack(self.cfg, overrides)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement forward().")

    def _compute_dynamic_weights(self, outer_product: Tensor) -> Tensor:
        return outer_product.sum(dim=1)

    def _compute_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        match self.normalization_position_option:
            case WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT:
                return self._compute_prenormalized_outer_product(
                    input_vectors, output_vectors
                )
            case WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT:
                return self._compute_postnormalized_outer_product(
                    input_vectors, output_vectors
                )
            case WeightNormalizationPositionOptions.DISABLED:
                return self._compute_raw_outer_product(input_vectors, output_vectors)
            case _:
                raise ValueError(
                    "Unsupported normalization_position_option value: "
                    f"{self.normalization_position_option!r}."
                )

    def _compute_prenormalized_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        input_vectors = self._apply_normalization_transform(input_vectors)
        output_vectors = self._apply_normalization_transform(output_vectors)
        return self._compute_raw_outer_product(input_vectors, output_vectors)

    def _compute_postnormalized_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        outer_product = self._compute_raw_outer_product(input_vectors, output_vectors)
        return self._apply_normalization_transform(outer_product)

    def _compute_raw_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        return torch.einsum("bki,bkj->bkij", input_vectors, output_vectors)

    def _apply_normalization_transform(
        self,
        vectors: Tensor,
    ) -> Tensor:
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
            case _:
                raise ValueError(
                    "Unsupported normalization_option value: "
                    f"{self.normalization_option!r}."
                )

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

    def _maybe_apply_weight_decay(self, weight_params: Tensor) -> Tensor:
        if (
            self.decay_schedule_option is None
            or self.decay_schedule_option == WeightDecayScheduleOptions.DISABLED
        ):
            return weight_params
        if self.warmup_step < self.decay_warmup_batches:
            if self.training:
                self.warmup_step += 1
            return weight_params
        decay_factor = self.__compute_decay_factor_by_schedule(
            self.decay_schedule_option
        )
        if self.training:
            self.decay_step += 1
        return weight_params * decay_factor

    def __compute_decay_factor_by_schedule(
        self,
        schedule: WeightDecayScheduleOptions,
    ) -> Tensor:
        match schedule:
            case WeightDecayScheduleOptions.EXPONENTIAL:
                return self.__compute_exponential_decay_factor()
            case WeightDecayScheduleOptions.LINEAR:
                return self.__compute_linear_decay_factor()
            case WeightDecayScheduleOptions.MULTIPLICATIVE:
                return self.__compute_multiplicative_decay_factor()
            case _:
                raise ValueError(f"Unsupported decay_schedule value: {schedule!r}.")

    def __compute_exponential_decay_factor(self) -> Tensor:
        maximum_finite_decay_rate = torch.finfo(self.decay_step.dtype).max
        dtype_aligned_decay_rate = self.decay_step.new_tensor(self.decay_rate)
        bounded_decay_rate = dtype_aligned_decay_rate.clamp(
            max=maximum_finite_decay_rate
        )
        exponential_decay_exponent = -bounded_decay_rate * self.decay_step
        exponential_decay_factor = torch.exp(exponential_decay_exponent)
        return exponential_decay_factor

    def __compute_linear_decay_factor(self) -> Tensor:
        unbounded_linear_decay_factor = 1.0 - self.decay_rate * self.decay_step
        nonnegative_linear_decay_factor = torch.clamp(
            unbounded_linear_decay_factor, min=0.0
        )
        return nonnegative_linear_decay_factor

    def __compute_multiplicative_decay_factor(self) -> Tensor:
        multiplicative_decay_base = self.decay_step.new_tensor(1.0 - self.decay_rate)
        multiplicative_decay_factor = torch.pow(
            multiplicative_decay_base, self.decay_step
        )
        return multiplicative_decay_factor

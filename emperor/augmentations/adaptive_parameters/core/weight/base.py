import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicWeightValidator,
)
from emperor.augmentations.adaptive_parameters.core.weight.config import (
    DynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight.depth_mapper import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters.options import (
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.base.module import Module


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
                return torch.clamp(vectors, -self.clamp_limit, self.clamp_limit)
            case WeightNormalizationOptions.L2_SCALE:
                return F.normalize(vectors, dim=-1) * self.scale
            case WeightNormalizationOptions.SOFT_CLAMP:
                return self.clamp_limit * torch.tanh(vectors / self.clamp_limit)
            case WeightNormalizationOptions.RMS:
                rms = vectors.pow(2).mean(dim=-1, keepdim=True).sqrt()
                return vectors / (rms + 1e-8) * self.scale
            case WeightNormalizationOptions.SIGMOID_SCALE:
                return (torch.sigmoid(vectors) * 2 - 1) * self.scale
            case WeightNormalizationOptions.DISABLED:
                return vectors
            case _:
                raise ValueError(
                    "Unsupported normalization_option value: "
                    f"{self.normalization_option!r}."
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
                return torch.exp(-self.decay_rate * self.decay_step)
            case WeightDecayScheduleOptions.LINEAR:
                linear_decay_factor = 1.0 - self.decay_rate * self.decay_step
                return torch.clamp(
                    linear_decay_factor,
                    min=0.0,
                )
            case WeightDecayScheduleOptions.MULTIPLICATIVE:
                decay_base = self.decay_step.new_tensor(1.0 - self.decay_rate)
                return torch.pow(decay_base, self.decay_step)
            case _:
                raise ValueError(f"Unsupported decay_schedule value: {schedule!r}.")

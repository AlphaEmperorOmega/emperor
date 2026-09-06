from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from emperor.augmentations.adaptive_parameters._options import (
    WeightDecayScheduleOptions,
)

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._biases.config import (
        DynamicBiasConfig,
    )
    from emperor.augmentations.adaptive_parameters._weights.config import (
        DynamicWeightConfig,
    )


class DecayPolicy(nn.Module):
    def __init__(self, cfg: DynamicWeightConfig | DynamicBiasConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decay_schedule_option = self.cfg.decay_schedule
        self.decay_rate = self.cfg.decay_rate
        self.decay_warmup_batches = self.cfg.decay_warmup_batches or 0
        self.register_buffer("decay_step", torch.zeros(1))
        self.register_buffer("warmup_step", torch.zeros(1))

    def forward(self, parameters: Tensor) -> Tensor:
        if self.__is_decay_schedule_disabled():
            return parameters
        if self.warmup_step < self.decay_warmup_batches:
            if self.training:
                self.warmup_step += 1
            return parameters

        decay_factor = self.__compute_decay_factor_by_schedule()
        if self.training:
            self.decay_step += 1
        return parameters * decay_factor

    def __is_decay_schedule_disabled(self) -> bool:
        return (
            self.decay_schedule_option is None
            or self.decay_schedule_option == WeightDecayScheduleOptions.DISABLED
        )

    def __compute_decay_factor_by_schedule(self) -> Tensor:
        match self.decay_schedule_option:
            case WeightDecayScheduleOptions.EXPONENTIAL:
                return self.__compute_exponential_decay_factor()
            case WeightDecayScheduleOptions.LINEAR:
                return self.__compute_linear_decay_factor()
            case WeightDecayScheduleOptions.MULTIPLICATIVE:
                return self.__compute_multiplicative_decay_factor()
            case _:
                raise ValueError(
                    f"Unsupported decay_schedule value: {self.decay_schedule_option!r}."
                )

    def __compute_exponential_decay_factor(self) -> Tensor:
        active_decay_rate = cast(float, self.decay_rate)
        maximum_finite_decay_rate = torch.finfo(self.decay_step.dtype).max
        dtype_aligned_decay_rate = self.decay_step.new_tensor(active_decay_rate)
        bounded_decay_rate = dtype_aligned_decay_rate.clamp(
            max=maximum_finite_decay_rate
        )
        exponential_decay_exponent = -bounded_decay_rate * self.decay_step
        exponential_decay_factor = torch.exp(exponential_decay_exponent)
        return exponential_decay_factor

    def __compute_linear_decay_factor(self) -> Tensor:
        active_decay_rate = cast(float, self.decay_rate)
        unbounded_linear_decay_factor = 1.0 - active_decay_rate * self.decay_step
        nonnegative_linear_decay_factor = torch.clamp(
            unbounded_linear_decay_factor, min=0.0
        )
        return nonnegative_linear_decay_factor

    def __compute_multiplicative_decay_factor(self) -> Tensor:
        active_decay_rate = cast(float, self.decay_rate)
        multiplicative_decay_base = self.decay_step.new_tensor(1.0 - active_decay_rate)
        multiplicative_decay_factor = torch.pow(
            multiplicative_decay_base,
            self.decay_step,
        )
        return multiplicative_decay_factor

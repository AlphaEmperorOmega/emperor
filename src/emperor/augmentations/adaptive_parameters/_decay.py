from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

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


class _DecayPolicy:
    __slots__ = (
        "decay_rate",
        "decay_schedule_option",
        "decay_warmup_batches",
    )

    def __init__(self, cfg: DynamicWeightConfig | DynamicBiasConfig) -> None:
        self.decay_schedule_option = cfg.decay_schedule
        self.decay_rate = cfg.decay_rate
        self.decay_warmup_batches = cfg.decay_warmup_batches or 0

    def apply(
        self,
        parameters: Tensor,
        *,
        decay_step: Tensor,
        warmup_step: Tensor,
        training: bool,
    ) -> Tensor:
        if self.__is_decay_schedule_disabled():
            return parameters
        if warmup_step < self.decay_warmup_batches:
            if training:
                warmup_step += 1
            return parameters

        active_decay_rate = cast(float, self.decay_rate)
        decay_factor = self.__compute_decay_factor_by_schedule(
            self.decay_schedule_option, active_decay_rate, decay_step
        )
        if training:
            decay_step += 1
        return parameters * decay_factor

    def __is_decay_schedule_disabled(self) -> bool:
        return (
            self.decay_schedule_option is None
            or self.decay_schedule_option == WeightDecayScheduleOptions.DISABLED
        )

    def __compute_decay_factor_by_schedule(
        self,
        schedule: WeightDecayScheduleOptions,
        decay_rate: float,
        decay_step: Tensor,
    ) -> Tensor:
        match schedule:
            case WeightDecayScheduleOptions.EXPONENTIAL:
                return self.__compute_exponential_decay_factor(decay_rate, decay_step)
            case WeightDecayScheduleOptions.LINEAR:
                return self.__compute_linear_decay_factor(decay_rate, decay_step)
            case WeightDecayScheduleOptions.MULTIPLICATIVE:
                return self.__compute_multiplicative_decay_factor(
                    decay_rate, decay_step
                )
            case _:
                raise ValueError(f"Unsupported decay_schedule value: {schedule!r}.")

    def __compute_exponential_decay_factor(
        self,
        decay_rate: float,
        decay_step: Tensor,
    ) -> Tensor:
        maximum_finite_decay_rate = torch.finfo(decay_step.dtype).max
        dtype_aligned_decay_rate = decay_step.new_tensor(decay_rate)
        bounded_decay_rate = dtype_aligned_decay_rate.clamp(
            max=maximum_finite_decay_rate
        )
        exponential_decay_exponent = -bounded_decay_rate * decay_step
        exponential_decay_factor = torch.exp(exponential_decay_exponent)
        return exponential_decay_factor

    def __compute_linear_decay_factor(
        self,
        decay_rate: float,
        decay_step: Tensor,
    ) -> Tensor:
        unbounded_linear_decay_factor = 1.0 - decay_rate * decay_step
        nonnegative_linear_decay_factor = torch.clamp(
            unbounded_linear_decay_factor, min=0.0
        )
        return nonnegative_linear_decay_factor

    def __compute_multiplicative_decay_factor(
        self,
        decay_rate: float,
        decay_step: Tensor,
    ) -> Tensor:
        multiplicative_decay_base = decay_step.new_tensor(1.0 - decay_rate)
        multiplicative_decay_factor = torch.pow(
            multiplicative_decay_base,
            decay_step,
        )
        return multiplicative_decay_factor

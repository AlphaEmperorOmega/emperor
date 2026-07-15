import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicBiasValidator,
)
from emperor.augmentations.adaptive_parameters.core.bias.config import DynamicBiasConfig
from emperor.augmentations.adaptive_parameters.options import (
    WeightDecayScheduleOptions,
)
from emperor.base.layer import Layer, LayerStack, LayerStackConfig
from emperor.base.module import Module


class DynamicBiasAbstract(Module):
    def __init__(
        self,
        cfg: "DynamicBiasConfig",
        overrides: "DynamicBiasConfig | None" = None,
    ):
        super().__init__()
        self.cfg: DynamicBiasConfig = self._override_config(cfg, overrides)
        DynamicBiasValidator.validate(self)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.decay_schedule_option = self.cfg.decay_schedule
        self.decay_rate = self.cfg.decay_rate
        self.decay_warmup_batches = self.cfg.decay_warmup_batches or 0
        self.model_config = self.cfg.model_config
        self.register_buffer("decay_step", torch.zeros(1))
        self.register_buffer("warmup_step", torch.zeros(1))

    def _init_model(self, output_dim: int) -> "Layer | LayerStack":
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=output_dim,
        )
        generator_model = self.model_config.build(overrides)
        DynamicBiasValidator.validate_generator_model(generator_model)
        return generator_model

    def _maybe_apply_bias_decay(self, bias_params: Tensor) -> Tensor:
        if (
            self.decay_schedule_option is None
            or self.decay_schedule_option == WeightDecayScheduleOptions.DISABLED
        ):
            return bias_params
        if self.warmup_step < self.decay_warmup_batches:
            if self.training:
                self.warmup_step += 1
            return bias_params
        decay_factor = self.__compute_decay_factor_by_schedule(
            self.decay_schedule_option
        )
        if self.training:
            self.decay_step += 1
        return bias_params * decay_factor

    def __compute_decay_factor_by_schedule(
        self,
        schedule: WeightDecayScheduleOptions,
    ) -> Tensor:
        step = self.decay_step
        rate = self.decay_rate
        match schedule:
            case WeightDecayScheduleOptions.EXPONENTIAL:
                return torch.exp(-rate * step)
            case WeightDecayScheduleOptions.LINEAR:
                return torch.clamp(1.0 - rate * step, min=0.0)
            case WeightDecayScheduleOptions.MULTIPLICATIVE:
                decay_base = step.new_tensor(1.0 - rate)
                return torch.pow(decay_base, step)
            case _:
                raise ValueError(f"Unsupported decay_schedule value: {schedule!r}.")

import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._biases.config import DynamicBiasConfig
from emperor.augmentations.adaptive_parameters._biases.validation import (
    DynamicBiasValidator,
)
from emperor.augmentations.adaptive_parameters._options import (
    WeightDecayScheduleOptions,
)
from emperor.layers import Layer, LayerStack, LayerStackConfig
from emperor.nn import Module


class DynamicBiasAbstract(Module):
    VALIDATOR = DynamicBiasValidator

    def __init__(
        self,
        cfg: "DynamicBiasConfig",
        overrides: "DynamicBiasConfig | None" = None,
    ):
        super().__init__()
        self.cfg: DynamicBiasConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
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
        self.VALIDATOR.validate_generator_model(generator_model)
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
        exponential_decay_exponent = -self.decay_rate * self.decay_step
        return torch.exp(exponential_decay_exponent)

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

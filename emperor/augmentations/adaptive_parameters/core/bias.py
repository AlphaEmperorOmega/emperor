import torch

from torch import Tensor
from dataclasses import dataclass
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.base.layer import Layer, LayerStack, LayerStackConfig
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    WeightDecayScheduleOptions,
)
from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicBiasValidator,
)


@dataclass
class DynamicBiasConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    decay_schedule: WeightDecayScheduleOptions | None = optional_field(
        "Base bias decay schedule."
    )
    decay_rate: float | None = optional_field("Decay rate for the selected schedule.")
    decay_warmup_batches: int | None = optional_field(
        "Warmup batches before bias decay starts."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Internal generator network config."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            f"DynamicBiasConfig is abstract and has no registered "
            f"DynamicBias class; instantiate a concrete leaf config instead."
        )


@dataclass
class AffineTransformDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        return AffineTransformDynamicBias


@dataclass
class AdditiveDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        return AdditiveDynamicBias


@dataclass
class MultiplicativeDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        return MultiplicativeDynamicBias


@dataclass
class SigmoidGatedDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        return SigmoidGatedDynamicBias


@dataclass
class TanhGatedDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        return TanhGatedDynamicBias


@dataclass
class GeneratorDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        return GeneratorDynamicBias


@dataclass
class WeightedBankDynamicBiasConfig(DynamicBiasConfig):
    bank_expansion_factor: BankExpansionFactorOptions | None = optional_field(
        "Bias bank expansion factor."
    )

    def _registry_owner(self) -> type:
        return WeightedBankDynamicBias


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


class AffineTransformDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: AffineTransformDynamicBiasConfig,
        overrides: AffineTransformDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        affine_parameter_dim = 2
        self.model = self._init_model(affine_parameter_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        DynamicBiasValidator.ensure_parameters_exist(bias_params)
        affine_parameters = Layer.run_model_returning_hidden(self.model, logits)
        bias_scale, bias_offset = affine_parameters.chunk(2, dim=-1)
        return bias_scale * bias_params + bias_offset


class AdditiveDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: AdditiveDynamicBiasConfig,
        overrides: AdditiveDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        DynamicBiasValidator.ensure_parameters_exist(bias_params)
        bias_params = self._maybe_apply_bias_decay(bias_params)
        generated_bias_offset = Layer.run_model_returning_hidden(self.model, logits)
        return bias_params + generated_bias_offset


class MultiplicativeDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: MultiplicativeDynamicBiasConfig,
        overrides: MultiplicativeDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        DynamicBiasValidator.ensure_parameters_exist(bias_params)
        bias_scale = Layer.run_model_returning_hidden(self.model, logits)
        return bias_params * bias_scale


class SigmoidGatedDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: SigmoidGatedDynamicBiasConfig,
        overrides: SigmoidGatedDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        DynamicBiasValidator.ensure_parameters_exist(bias_params)
        gate = torch.sigmoid(Layer.run_model_returning_hidden(self.model, logits))
        return bias_params * gate


class TanhGatedDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: TanhGatedDynamicBiasConfig,
        overrides: TanhGatedDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        DynamicBiasValidator.ensure_parameters_exist(bias_params)
        gate = torch.tanh(Layer.run_model_returning_hidden(self.model, logits))
        return bias_params * gate


class GeneratorDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: GeneratorDynamicBiasConfig,
        overrides: GeneratorDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, _bias_params: Tensor, logits: Tensor) -> Tensor:
        return Layer.run_model_returning_hidden(self.model, logits)


class WeightedBankDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: WeightedBankDynamicBiasConfig,
        overrides: WeightedBankDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        DynamicBiasValidator.validate_bank_expansion_factor(self)
        self.bank_expansion_factor = self.cfg.bank_expansion_factor.value
        self.weight_bank = self._init_parameter_bank(
            (self.bank_expansion_factor, self.output_dim)
        )
        self.model = self._init_model(self.bank_expansion_factor)

    def forward(self, _bias_params: Tensor, logits: Tensor) -> Tensor:
        bank_logits = Layer.run_model_returning_hidden(self.model, logits)
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        return torch.matmul(bank_distribution, self.weight_bank)

import torch

from torch import Tensor
from dataclasses import dataclass
from torch.nn import Sequential
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.base.registry import subclass_registry
from emperor.base.layer import Layer, LayerStackConfig
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    WeightDecayScheduleOptions,
)
from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicBiasValidator,
)


@dataclass
class DynamicBiasConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input dimensionality of the dynamic bias module."
    )
    output_dim: int | None = optional_field(
        "Output dimensionality of the dynamic bias module."
    )
    bias_flag: bool | None = optional_field(
        "Indicates whether the associated linear layer includes a bias parameter."
    )
    model_type: DynamicBiasOptions | None = optional_field(
        "Dynamic bias strategy used to generate input-dependent bias updates."
    )
    bank_expansion_factor: int | None = optional_field(
        "Number of entries in the bank used by the WEIGHTED_BANK bias strategy."
    )
    decay_schedule: WeightDecayScheduleOptions | None = optional_field(
        "Schedule used to decay the base bias parameters across forward passes."
    )
    decay_rate: float | None = optional_field(
        "Decay coefficient for the selected schedule. Its interpretation depends on the schedule type."
    )
    decay_warmup_batches: int | None = optional_field(
        "Number of initial batches to skip before applying bias decay."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Configuration for the internal generator network."
    )

    def _registry_owner(self) -> type:
        return DynamicBiasAbstract


@subclass_registry
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
        self.bias_flag = self.cfg.bias_flag
        self.model_type = self.cfg.model_type
        self.bank_expansion_factor = self.cfg.bank_expansion_factor
        self.decay_schedule_option = self.cfg.decay_schedule
        self.decay_rate = self.cfg.decay_rate
        self.decay_warmup_batches = self.cfg.decay_warmup_batches or 0
        self.model_config = self.cfg.model_config
        self.register_buffer("decay_step", torch.zeros(1))
        self.register_buffer("warmup_step", torch.zeros(1))

    def _init_output_generator(self, output_dim: int) -> "Layer | Sequential":
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=output_dim,
        )
        generator_model = self.model_config.build(overrides)
        DynamicBiasValidator.validate_generator_model(generator_model)
        return generator_model

    def _run_generator(
        self, generator_model: "Layer | Sequential", logits: Tensor
    ) -> Tensor:
        return Layer.forward_with_state(generator_model, logits)

    def _require_bias_params(self, bias_params: Tensor | None) -> Tensor:
        DynamicBiasValidator.ensure_parameters_exist(bias_params)
        return bias_params

    def _maybe_apply_bias_decay(self, bias_params: Tensor) -> Tensor:
        if (
            self.decay_schedule_option is None
            or self.decay_schedule_option == WeightDecayScheduleOptions.DISABLED
        ):
            return bias_params
        if self.warmup_step < self.decay_warmup_batches:
            self.warmup_step += 1
            return bias_params
        decay_factor = self.__compute_decay_factor_by_schedule(
            self.decay_schedule_option
        )
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


@DynamicBiasAbstract.register(DynamicBiasOptions.SCALE_AND_OFFSET)
class AffineTransformDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        affine_parameter_dim = 2
        self.scalar_offset_generator = self._init_output_generator(affine_parameter_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        bias_params = self._require_bias_params(bias_params)
        affine_parameters = self._run_generator(self.scalar_offset_generator, logits)
        bias_scale, bias_offset = affine_parameters.chunk(2, dim=-1)
        return bias_scale * bias_params + bias_offset


@DynamicBiasAbstract.register(DynamicBiasOptions.ADDITIVE)
class AdditiveDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.generator_model = self._init_output_generator(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        bias_params = self._require_bias_params(bias_params)
        bias_params = self._maybe_apply_bias_decay(bias_params)
        generated_bias_offset = self._run_generator(self.generator_model, logits)
        return bias_params + generated_bias_offset


@DynamicBiasAbstract.register(DynamicBiasOptions.MULTIPLICATIVE)
class MultiplicativeDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.scale_generator = self._init_output_generator(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        bias_params = self._require_bias_params(bias_params)
        bias_scale = self._run_generator(self.scale_generator, logits)
        return bias_params * bias_scale


@DynamicBiasAbstract.register(DynamicBiasOptions.SIGMOID_MULTIPLICATIVE)
class SigmoidGatedDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.gate_generator = self._init_output_generator(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        bias_params = self._require_bias_params(bias_params)
        gate = torch.sigmoid(self._run_generator(self.gate_generator, logits))
        return bias_params * gate


@DynamicBiasAbstract.register(DynamicBiasOptions.TANH_MULTIPLICATIVE)
class TanhGatedDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.gate_generator = self._init_output_generator(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        bias_params = self._require_bias_params(bias_params)
        gate = torch.tanh(self._run_generator(self.gate_generator, logits))
        return bias_params * gate


@DynamicBiasAbstract.register(DynamicBiasOptions.DYNAMIC_PARAMETERS)
class GeneratorDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.bias_generator = self._init_output_generator(self.output_dim)

    def forward(self, _bias_params: Tensor, logits: Tensor) -> Tensor:
        return self._run_generator(self.bias_generator, logits)


@DynamicBiasAbstract.register(DynamicBiasOptions.WEIGHTED_BANK)
class WeightedBankDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: DynamicBiasConfig,
        overrides: DynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        DynamicBiasValidator.validate_bank_expansion_factor(self)
        self.weight_bank = self._init_parameter_bank(
            (self.bank_expansion_factor, self.output_dim)
        )
        self.distribution_generator = self._init_output_generator(
            self.bank_expansion_factor
        )

    def forward(self, _bias_params: Tensor, logits: Tensor) -> Tensor:
        bank_logits = self._run_generator(self.distribution_generator, logits)
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        return torch.matmul(bank_distribution, self.weight_bank)

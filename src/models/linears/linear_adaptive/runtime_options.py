from __future__ import annotations

from dataclasses import dataclass

from emperor.augmentations.adaptive_parameters.core.bias import DynamicBiasConfig
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import AxisMaskConfig
from emperor.augmentations.adaptive_parameters.core.weight import DynamicWeightConfig
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.memory.config import DynamicMemoryConfig
from emperor.memory.options import MemoryPositionOptions

_PACKAGE = "models.linears.linear_adaptive"


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{_PACKAGE}: {name!r} must be positive")


def _non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{_PACKAGE}: {name!r} must be non-negative")


def _probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{_PACKAGE}: {name!r} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class StackOptions:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: ResidualConnectionOptions
    dropout_probability: float
    bias_flag: bool

    def __post_init__(self) -> None:
        _positive("stack.hidden_dim", self.hidden_dim)
        _positive("stack.num_layers", self.num_layers)
        _probability("stack.dropout_probability", self.dropout_probability)


@dataclass(frozen=True, slots=True)
class GateOptions:
    enabled: bool
    option: LayerGateOptions | None
    activation: ActivationOptions | None
    stack: StackOptions
    shared_config: GateConfig | None = None


@dataclass(frozen=True, slots=True)
class HaltingOptions:
    enabled: bool
    threshold: float
    dropout_probability: float
    hidden_state_mode: HaltingHiddenStateModeOptions
    stack: StackOptions

    def __post_init__(self) -> None:
        _probability("halting.threshold", self.threshold)
        _probability("halting.dropout_probability", self.dropout_probability)


@dataclass(frozen=True, slots=True)
class MemoryOptions:
    enabled: bool
    option: type[DynamicMemoryConfig]
    position: MemoryPositionOptions
    test_time_training_learning_rate: float | None
    test_time_training_num_inner_steps: int | None
    stack: StackOptions

    def __post_init__(self) -> None:
        if self.test_time_training_learning_rate is not None:
            _positive(
                "memory.test_time_training_learning_rate",
                self.test_time_training_learning_rate,
            )
        if self.test_time_training_num_inner_steps is not None:
            _positive(
                "memory.test_time_training_num_inner_steps",
                self.test_time_training_num_inner_steps,
            )


@dataclass(frozen=True, slots=True)
class RecurrenceOptions:
    enabled: bool
    max_steps: int
    layer_norm_position: LayerNormPositionOptions
    gate: GateOptions
    halting: HaltingOptions

    def __post_init__(self) -> None:
        _positive("recurrence.max_steps", self.max_steps)


@dataclass(frozen=True, slots=True)
class GeneratorStackOptions:
    """A resolved generator stack plus whether it replaces the shared stack."""

    independent: bool
    stack: StackOptions


@dataclass(frozen=True, slots=True)
class AdaptiveWeightOptions:
    enabled: bool
    option: type[DynamicWeightConfig] | None
    generator_depth: DynamicDepthOptions
    normalization_option: WeightNormalizationOptions
    normalization_position_option: WeightNormalizationPositionOptions
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack: GeneratorStackOptions

    def __post_init__(self) -> None:
        if self.enabled and self.option is None:
            raise ValueError(
                f"{_PACKAGE}: 'weight.option' must be set when weight is enabled"
            )
        _non_negative("weight.decay_rate", self.decay_rate)
        _non_negative("weight.decay_warmup_batches", self.decay_warmup_batches)


@dataclass(frozen=True, slots=True)
class AdaptiveBiasOptions:
    enabled: bool
    option: type[DynamicBiasConfig] | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack: GeneratorStackOptions

    def __post_init__(self) -> None:
        if self.enabled and self.option is None:
            raise ValueError(
                f"{_PACKAGE}: 'bias.option' must be set when bias is enabled"
            )
        _non_negative("bias.decay_rate", self.decay_rate)
        _non_negative("bias.decay_warmup_batches", self.decay_warmup_batches)


@dataclass(frozen=True, slots=True)
class AdaptiveDiagonalOptions:
    enabled: bool
    option: type[DynamicDiagonalConfig] | None
    generator_stack: GeneratorStackOptions

    def __post_init__(self) -> None:
        if self.enabled and self.option is None:
            raise ValueError(
                f"{_PACKAGE}: 'diagonal.option' must be set when diagonal is enabled"
            )


@dataclass(frozen=True, slots=True)
class AdaptiveMaskOptions:
    enabled: bool
    row_mask_option: type[AxisMaskConfig] | None
    dimension_option: MaskDimensionOptions
    threshold: float
    surrogate_scale: float
    floor: float
    transition_width: float
    generator_stack: GeneratorStackOptions

    def __post_init__(self) -> None:
        if self.enabled and self.row_mask_option is None:
            raise ValueError(
                f"{_PACKAGE}: 'mask.row_mask_option' must be set when mask is enabled"
            )
        _probability("mask.threshold", self.threshold)
        _probability("mask.floor", self.floor)
        _positive("mask.surrogate_scale", self.surrogate_scale)
        _positive("mask.transition_width", self.transition_width)


@dataclass(frozen=True, slots=True)
class AdaptiveProjectionOptions:
    weight_option: type[DynamicWeightConfig] | None
    weight_generator_depth: DynamicDepthOptions
    weight_decay_schedule: WeightDecayScheduleOptions
    weight_decay_rate: float
    weight_decay_warmup_batches: int
    weight_normalization_option: WeightNormalizationOptions
    weight_normalization_position_option: WeightNormalizationPositionOptions
    weight_bank_expansion_factor: BankExpansionFactorOptions
    bias_option: type[DynamicBiasConfig] | None
    bias_decay_schedule: WeightDecayScheduleOptions
    bias_decay_rate: float
    bias_decay_warmup_batches: int
    bias_bank_expansion_factor: BankExpansionFactorOptions
    diagonal_option: type[DynamicDiagonalConfig] | None
    row_mask_option: type[AxisMaskConfig] | None
    mask_dimension_option: MaskDimensionOptions
    mask_threshold: float
    mask_surrogate_scale: float
    mask_floor: float
    mask_transition_width: float

    def __post_init__(self) -> None:
        _non_negative("projection.weight_decay_rate", self.weight_decay_rate)
        _non_negative(
            "projection.weight_decay_warmup_batches",
            self.weight_decay_warmup_batches,
        )
        _non_negative("projection.bias_decay_rate", self.bias_decay_rate)
        _non_negative(
            "projection.bias_decay_warmup_batches",
            self.bias_decay_warmup_batches,
        )
        _probability("projection.mask_threshold", self.mask_threshold)
        _probability("projection.mask_floor", self.mask_floor)
        _positive("projection.mask_surrogate_scale", self.mask_surrogate_scale)
        _positive("projection.mask_transition_width", self.mask_transition_width)


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    batch_size: int
    learning_rate: float
    input_dim: int
    hidden_dim: int
    output_dim: int
    stack: StackOptions
    submodule_stack: StackOptions
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions
    recurrence: RecurrenceOptions
    adaptive_generator_stack: StackOptions
    weight: AdaptiveWeightOptions
    bias: AdaptiveBiasOptions
    diagonal: AdaptiveDiagonalOptions
    mask: AdaptiveMaskOptions
    input_projection: AdaptiveProjectionOptions
    output_projection: AdaptiveProjectionOptions

    def __post_init__(self) -> None:
        _positive("batch_size", self.batch_size)
        _positive("learning_rate", self.learning_rate)
        _positive("input_dim", self.input_dim)
        _positive("hidden_dim", self.hidden_dim)
        _positive("output_dim", self.output_dim)
        if self.gate.enabled and self.gate.shared_config is not None:
            raise ValueError(
                f"{_PACKAGE}: enabled gate and shared gate config are mutually "
                "exclusive"
            )

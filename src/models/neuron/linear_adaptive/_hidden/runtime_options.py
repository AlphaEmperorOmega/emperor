from __future__ import annotations

from dataclasses import dataclass

from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConnectionOptions,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions


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


@dataclass(frozen=True, slots=True)
class MemoryOptions:
    enabled: bool
    option: type[DynamicMemoryConfig]
    position: MemoryPositionOptions
    test_time_training_learning_rate: float | None
    test_time_training_num_inner_steps: int | None
    stack: StackOptions


@dataclass(frozen=True, slots=True)
class RecurrenceOptions:
    enabled: bool
    max_steps: int
    layer_norm_position: LayerNormPositionOptions
    gate: GateOptions
    halting: HaltingOptions


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


@dataclass(frozen=True, slots=True)
class AdaptiveBiasOptions:
    enabled: bool
    option: type[DynamicBiasConfig] | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack: GeneratorStackOptions


@dataclass(frozen=True, slots=True)
class AdaptiveDiagonalOptions:
    enabled: bool
    option: type[DynamicDiagonalConfig] | None
    generator_stack: GeneratorStackOptions


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
    halting_option: type[HaltingConfig] = StickBreakingConfig
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig

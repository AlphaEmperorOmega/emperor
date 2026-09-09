from dataclasses import dataclass, field

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
    NormalizationOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.linears.linear._residual import ResidualStackOptions


@dataclass(frozen=True, slots=True)
class MainStackOptions:
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool


@dataclass(frozen=True, slots=True)
class ControllerStackOptions:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class GateOptions:
    enabled: bool
    option: LayerGateOptions | None
    activation: ActivationOptions | None
    stack: ControllerStackOptions
    shared_config: GateConfig | None = None


@dataclass(frozen=True, slots=True)
class HaltingOptions:
    enabled: bool
    threshold: float
    dropout_probability: float
    hidden_state_mode: HaltingHiddenStateModeOptions
    stack: ControllerStackOptions


@dataclass(frozen=True, slots=True)
class MemoryOptions:
    enabled: bool
    implementation: type[DynamicMemoryConfig]
    position: MemoryPositionOptions
    test_time_training_learning_rate: float | None
    test_time_training_num_inner_steps: int | None
    stack: ControllerStackOptions


@dataclass(frozen=True, slots=True)
class RecurrenceOptions:
    enabled: bool
    max_steps: int
    initial_iterations: int = field(default=2, kw_only=True)
    gradient_transition_count: int | None = field(default=None, kw_only=True)
    no_gradient_transition_count: int | None = field(default=None, kw_only=True)
    iteration_increment: int = field(default=1, kw_only=True)
    forward_calls_before_iteration_increment: int = field(default=1, kw_only=True)
    smooth_iteration_growth_flag: bool = field(default=False, kw_only=True)
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    gate: GateOptions
    halting: HaltingOptions


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    batch_size: int
    learning_rate: float
    input_dim: int
    hidden_dim: int
    output_dim: int
    stack: MainStackOptions
    submodule_stack: ControllerStackOptions
    residual_stack: ResidualStackOptions | None = field(
        default=None,
        kw_only=True,
    )
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions
    recurrence: RecurrenceOptions
    halting_option: type[HaltingConfig] = StickBreakingConfig
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class MainStackOptions:
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: ResidualConnectionOptions
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool


@dataclass(frozen=True, slots=True)
class ControllerStackOptions:
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
    layer_norm_position: LayerNormPositionOptions
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
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions
    recurrence: RecurrenceOptions

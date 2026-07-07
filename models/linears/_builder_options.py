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
from models.linears._controller_stack import SubmoduleStackSource


@dataclass(frozen=True)
class MainLayerStackOptions:
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: ResidualConnectionOptions
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool


@dataclass(frozen=True)
class LayerControllerOptions:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    gate_stack_source: SubmoduleStackSource
    stack_halting_flag: bool
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    halting_stack_source: SubmoduleStackSource
    shared_gate_config: GateConfig | None = None


@dataclass(frozen=True)
class DynamicMemoryOptions:
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_test_time_training_learning_rate: float | None
    memory_test_time_training_num_inner_steps: int | None
    memory_stack_source: SubmoduleStackSource


@dataclass(frozen=True)
class RecurrentControllerOptions:
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_gate_stack_source: SubmoduleStackSource
    recurrent_halting_flag: bool
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    recurrent_halting_stack_source: SubmoduleStackSource

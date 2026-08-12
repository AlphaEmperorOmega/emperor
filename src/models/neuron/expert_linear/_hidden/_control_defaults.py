from dataclasses import dataclass
from types import ModuleType

from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.neuron.expert_linear._hidden.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
)


@dataclass(frozen=True, slots=True)
class ControlDefaults:
    layer: ExpertsLayerControllerOptions
    memory: ExpertsDynamicMemoryOptions
    recurrent: ExpertsRecurrentControllerOptions


def router_stack_defaults(config: ModuleType) -> ExpertsSubmoduleStackOptions:
    return ExpertsSubmoduleStackOptions(
        hidden_dim=config.ROUTER_STACK_HIDDEN_DIM,
        num_layers=config.ROUTER_STACK_NUM_LAYERS,
        last_layer_bias_option=config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.ROUTER_STACK_ACTIVATION,
        layer_norm_position=config.ROUTER_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.ROUTER_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.ROUTER_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.ROUTER_BIAS_FLAG,
    )


def main_control_defaults(config: ModuleType) -> ControlDefaults:
    return ControlDefaults(
        layer=_layer_options(
            stack_gate_flag=config.STACK_GATE_FLAG,
            gate_option=config.GATE_OPTION,
            gate_activation=config.GATE_ACTIVATION,
            gate_stack_source=_main_gate_stack_source(config),
            stack_halting_flag=config.STACK_HALTING_FLAG,
            halting_option=config.HALTING_OPTION,
            halting_threshold=config.HALTING_THRESHOLD,
            halting_dropout=config.HALTING_DROPOUT,
            halting_hidden_state_mode=config.HALTING_HIDDEN_STATE_MODE,
            halting_stack_source=_main_halting_stack_source(config),
            halting_output_dim=config.HALTING_OUTPUT_DIM,
        ),
        memory=_memory_options(
            memory_flag=config.MEMORY_FLAG,
            memory_option=config.MEMORY_OPTION,
            memory_position_option=config.MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=(
                config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
            ),
            memory_test_time_training_num_inner_steps=(
                config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
            ),
            memory_stack_source=_main_memory_stack_source(config),
        ),
        recurrent=_recurrent_options(
            recurrent_flag=config.RECURRENT_FLAG,
            recurrent_max_steps=config.RECURRENT_MAX_STEPS,
            recurrent_initial_iterations=config.RECURRENT_INITIAL_ITERATIONS,
            recurrent_gradient_transition_count=(
                config.RECURRENT_GRADIENT_TRANSITION_COUNT
            ),
            recurrent_iteration_increment=config.RECURRENT_ITERATION_INCREMENT,
            recurrent_forward_calls_before_iteration_increment=(
                config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT
            ),
            recurrent_layer_norm_position=config.RECURRENT_LAYER_NORM_POSITION,
            recurrent_stack_gate_flag=config.RECURRENT_STACK_GATE_FLAG,
            recurrent_gate_option=config.RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=_main_recurrent_gate_stack_source(config),
            recurrent_stack_halting_flag=config.RECURRENT_STACK_HALTING_FLAG,
            recurrent_halting_option=config.RECURRENT_HALTING_OPTION,
            recurrent_halting_threshold=config.RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=(
                config.RECURRENT_HALTING_HIDDEN_STATE_MODE
            ),
            recurrent_halting_stack_source=(
                _main_recurrent_halting_stack_source(config)
            ),
        ),
    )


def expert_control_defaults(config: ModuleType) -> ControlDefaults:
    return ControlDefaults(
        layer=_layer_options(
            stack_gate_flag=config.EXPERT_STACK_GATE_FLAG,
            gate_option=config.EXPERT_GATE_OPTION,
            gate_activation=config.EXPERT_GATE_ACTIVATION,
            gate_stack_source=_expert_gate_stack_source(config),
            stack_halting_flag=config.EXPERT_STACK_HALTING_FLAG,
            halting_option=config.EXPERT_HALTING_OPTION,
            halting_threshold=config.EXPERT_HALTING_THRESHOLD,
            halting_dropout=config.EXPERT_HALTING_DROPOUT,
            halting_hidden_state_mode=config.EXPERT_HALTING_HIDDEN_STATE_MODE,
            halting_stack_source=_expert_halting_stack_source(config),
            halting_output_dim=config.EXPERT_HALTING_OUTPUT_DIM,
        ),
        memory=_memory_options(
            memory_flag=config.EXPERT_MEMORY_FLAG,
            memory_option=config.EXPERT_MEMORY_OPTION,
            memory_position_option=config.EXPERT_MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=(
                config.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
            ),
            memory_test_time_training_num_inner_steps=(
                config.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
            ),
            memory_stack_source=_expert_memory_stack_source(config),
        ),
        recurrent=_recurrent_options(
            recurrent_flag=config.EXPERT_RECURRENT_FLAG,
            recurrent_max_steps=config.EXPERT_RECURRENT_MAX_STEPS,
            recurrent_initial_iterations=2,
            recurrent_gradient_transition_count=None,
            recurrent_iteration_increment=1,
            recurrent_forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=config.EXPERT_RECURRENT_LAYER_NORM_POSITION,
            recurrent_stack_gate_flag=config.EXPERT_RECURRENT_STACK_GATE_FLAG,
            recurrent_gate_option=config.EXPERT_RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.EXPERT_RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=_expert_recurrent_gate_stack_source(config),
            recurrent_stack_halting_flag=config.EXPERT_RECURRENT_STACK_HALTING_FLAG,
            recurrent_halting_option=config.EXPERT_RECURRENT_HALTING_OPTION,
            recurrent_halting_threshold=config.EXPERT_RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.EXPERT_RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=(
                config.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
            ),
            recurrent_halting_stack_source=(
                _expert_recurrent_halting_stack_source(config)
            ),
        ),
    )


def _main_gate_stack_source(config: ModuleType) -> ExpertsSubmoduleStackSource:
    return _stack_source(
        config.GATE_STACK_INDEPENDENT_FLAG,
        config.GATE_STACK_HIDDEN_DIM,
        config.GATE_STACK_NUM_LAYERS,
        config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.GATE_STACK_ACTIVATION,
        config.GATE_STACK_LAYER_NORM_POSITION,
        config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        config.GATE_STACK_RESIDUAL_MODEL_FLAG,
        config.GATE_STACK_DROPOUT_PROBABILITY,
        config.GATE_STACK_BIAS_FLAG,
    )


def _main_halting_stack_source(config: ModuleType) -> ExpertsSubmoduleStackSource:
    return _stack_source(
        config.HALTING_STACK_INDEPENDENT_FLAG,
        config.HALTING_STACK_HIDDEN_DIM,
        config.HALTING_STACK_NUM_LAYERS,
        config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.HALTING_STACK_ACTIVATION,
        config.HALTING_STACK_LAYER_NORM_POSITION,
        config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        config.HALTING_STACK_RESIDUAL_MODEL_FLAG,
        config.HALTING_STACK_DROPOUT_PROBABILITY,
        config.HALTING_STACK_BIAS_FLAG,
    )


def _main_memory_stack_source(config: ModuleType) -> ExpertsSubmoduleStackSource:
    return _stack_source(
        config.MEMORY_STACK_INDEPENDENT_FLAG,
        config.MEMORY_STACK_HIDDEN_DIM,
        config.MEMORY_STACK_NUM_LAYERS,
        config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        config.MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.MEMORY_STACK_ACTIVATION,
        config.MEMORY_STACK_LAYER_NORM_POSITION,
        config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
        config.MEMORY_STACK_RESIDUAL_MODEL_FLAG,
        config.MEMORY_STACK_DROPOUT_PROBABILITY,
        config.MEMORY_STACK_BIAS_FLAG,
    )


def _main_recurrent_gate_stack_source(
    config: ModuleType,
) -> ExpertsSubmoduleStackSource:
    return _stack_source(
        config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
        config.RECURRENT_GATE_STACK_HIDDEN_DIM,
        config.RECURRENT_GATE_STACK_NUM_LAYERS,
        config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        config.RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.RECURRENT_GATE_STACK_ACTIVATION,
        config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
        config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        config.RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
        config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
        config.RECURRENT_GATE_STACK_BIAS_FLAG,
    )


def _main_recurrent_halting_stack_source(
    config: ModuleType,
) -> ExpertsSubmoduleStackSource:
    return _stack_source(
        config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        config.RECURRENT_HALTING_STACK_HIDDEN_DIM,
        config.RECURRENT_HALTING_STACK_NUM_LAYERS,
        config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.RECURRENT_HALTING_STACK_ACTIVATION,
        config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        config.RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        config.RECURRENT_HALTING_STACK_BIAS_FLAG,
    )


def _expert_gate_stack_source(config: ModuleType) -> ExpertsSubmoduleStackSource:
    return _stack_source(
        config.EXPERT_GATE_STACK_INDEPENDENT_FLAG,
        config.EXPERT_GATE_STACK_HIDDEN_DIM,
        config.EXPERT_GATE_STACK_NUM_LAYERS,
        config.EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        config.EXPERT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.EXPERT_GATE_STACK_ACTIVATION,
        config.EXPERT_GATE_STACK_LAYER_NORM_POSITION,
        config.EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        config.EXPERT_GATE_STACK_RESIDUAL_MODEL_FLAG,
        config.EXPERT_GATE_STACK_DROPOUT_PROBABILITY,
        config.EXPERT_GATE_STACK_BIAS_FLAG,
    )


def _expert_halting_stack_source(config: ModuleType) -> ExpertsSubmoduleStackSource:
    return _stack_source(
        config.EXPERT_HALTING_STACK_INDEPENDENT_FLAG,
        config.EXPERT_HALTING_STACK_HIDDEN_DIM,
        config.EXPERT_HALTING_STACK_NUM_LAYERS,
        config.EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        config.EXPERT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.EXPERT_HALTING_STACK_ACTIVATION,
        config.EXPERT_HALTING_STACK_LAYER_NORM_POSITION,
        config.EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        config.EXPERT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        config.EXPERT_HALTING_STACK_DROPOUT_PROBABILITY,
        config.EXPERT_HALTING_STACK_BIAS_FLAG,
    )


def _expert_memory_stack_source(config: ModuleType) -> ExpertsSubmoduleStackSource:
    return _stack_source(
        config.EXPERT_MEMORY_STACK_INDEPENDENT_FLAG,
        config.EXPERT_MEMORY_STACK_HIDDEN_DIM,
        config.EXPERT_MEMORY_STACK_NUM_LAYERS,
        config.EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        config.EXPERT_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.EXPERT_MEMORY_STACK_ACTIVATION,
        config.EXPERT_MEMORY_STACK_LAYER_NORM_POSITION,
        config.EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
        config.EXPERT_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
        config.EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY,
        config.EXPERT_MEMORY_STACK_BIAS_FLAG,
    )


def _expert_recurrent_gate_stack_source(
    config: ModuleType,
) -> ExpertsSubmoduleStackSource:
    return _stack_source(
        config.EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
        config.EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM,
        config.EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS,
        config.EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        config.EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.EXPERT_RECURRENT_GATE_STACK_ACTIVATION,
        config.EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
        config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
        config.EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
        config.EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG,
    )


def _expert_recurrent_halting_stack_source(
    config: ModuleType,
) -> ExpertsSubmoduleStackSource:
    return _stack_source(
        config.EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        config.EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM,
        config.EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS,
        config.EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        config.EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.EXPERT_RECURRENT_HALTING_STACK_ACTIVATION,
        config.EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        config.EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        config.EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG,
    )


def _stack_source(
    independent_flag: bool,
    hidden_dim: int | None,
    num_layers: int | None,
    last_layer_bias_option: LastLayerBiasOptions | None,
    apply_output_pipeline_flag: bool | None,
    activation: ActivationOptions | None,
    layer_norm_position: LayerNormPositionOptions | None,
    residual_connection_option: type[ResidualConfig] | None,
    residual_model_flag: bool,
    dropout_probability: float | None,
    bias_flag: bool | None,
) -> ExpertsSubmoduleStackSource:
    return ExpertsSubmoduleStackSource(
        independent_flag=independent_flag,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        activation=activation,
        layer_norm_position=layer_norm_position,
        residual_connection_option=residual_connection_option,
        residual_model_flag=residual_model_flag,
        dropout_probability=dropout_probability,
        bias_flag=bias_flag,
    )


def _layer_options(
    *,
    stack_gate_flag: bool,
    gate_option: LayerGateOptions | None,
    gate_activation: ActivationOptions | None,
    gate_stack_source: ExpertsSubmoduleStackSource,
    stack_halting_flag: bool,
    halting_option: type[HaltingConfig],
    halting_threshold: float,
    halting_dropout: float,
    halting_hidden_state_mode: HaltingHiddenStateModeOptions,
    halting_stack_source: ExpertsSubmoduleStackSource,
    halting_output_dim: int,
) -> ExpertsLayerControllerOptions:
    return ExpertsLayerControllerOptions(
        stack_gate_flag=stack_gate_flag,
        gate_option=gate_option,
        gate_activation=gate_activation,
        gate_stack_source=gate_stack_source,
        stack_halting_flag=stack_halting_flag,
        halting_option=halting_option,
        halting_threshold=halting_threshold,
        halting_dropout=halting_dropout,
        halting_hidden_state_mode=halting_hidden_state_mode,
        halting_stack_source=halting_stack_source,
        halting_output_dim=halting_output_dim,
    )


def _memory_options(
    *,
    memory_flag: bool,
    memory_option: type[DynamicMemoryConfig],
    memory_position_option: MemoryPositionOptions,
    memory_test_time_training_learning_rate: float | None,
    memory_test_time_training_num_inner_steps: int | None,
    memory_stack_source: ExpertsSubmoduleStackSource,
) -> ExpertsDynamicMemoryOptions:
    return ExpertsDynamicMemoryOptions(
        memory_flag=memory_flag,
        memory_option=memory_option,
        memory_position_option=memory_position_option,
        memory_test_time_training_learning_rate=(
            memory_test_time_training_learning_rate
        ),
        memory_test_time_training_num_inner_steps=(
            memory_test_time_training_num_inner_steps
        ),
        memory_stack_source=memory_stack_source,
    )


def _recurrent_options(
    *,
    recurrent_flag: bool,
    recurrent_max_steps: int,
    recurrent_initial_iterations: int,
    recurrent_gradient_transition_count: int | None,
    recurrent_iteration_increment: int,
    recurrent_forward_calls_before_iteration_increment: int,
    recurrent_layer_norm_position: LayerNormPositionOptions,
    recurrent_stack_gate_flag: bool,
    recurrent_gate_option: LayerGateOptions | None,
    recurrent_gate_activation: ActivationOptions | None,
    recurrent_gate_stack_source: ExpertsSubmoduleStackSource,
    recurrent_stack_halting_flag: bool,
    recurrent_halting_option: type[HaltingConfig],
    recurrent_halting_threshold: float,
    recurrent_halting_dropout: float,
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions,
    recurrent_halting_stack_source: ExpertsSubmoduleStackSource,
) -> ExpertsRecurrentControllerOptions:
    return ExpertsRecurrentControllerOptions(
        recurrent_flag=recurrent_flag,
        recurrent_max_steps=recurrent_max_steps,
        recurrent_initial_iterations=recurrent_initial_iterations,
        recurrent_gradient_transition_count=recurrent_gradient_transition_count,
        recurrent_iteration_increment=recurrent_iteration_increment,
        recurrent_forward_calls_before_iteration_increment=(
            recurrent_forward_calls_before_iteration_increment
        ),
        recurrent_layer_norm_position=recurrent_layer_norm_position,
        recurrent_stack_gate_flag=recurrent_stack_gate_flag,
        recurrent_gate_option=recurrent_gate_option,
        recurrent_gate_activation=recurrent_gate_activation,
        recurrent_gate_stack_source=recurrent_gate_stack_source,
        recurrent_stack_halting_flag=recurrent_stack_halting_flag,
        recurrent_halting_option=recurrent_halting_option,
        recurrent_halting_threshold=recurrent_halting_threshold,
        recurrent_halting_dropout=recurrent_halting_dropout,
        recurrent_halting_hidden_state_mode=recurrent_halting_hidden_state_mode,
        recurrent_halting_stack_source=recurrent_halting_stack_source,
    )

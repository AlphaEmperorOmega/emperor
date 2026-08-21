from dataclasses import dataclass
from typing import cast

import models.neuron.expert_linear_adaptive.config as config
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.neuron.expert_linear_adaptive._hidden.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
)
from models.neuron.expert_linear_adaptive._residual import (
    ResidualStackOptions,
    ResidualStackSource,
    resolve_residual_stack_options,
)


@dataclass(frozen=True, kw_only=True)
class ControlDefaultValues:
    residual_stack_independent_flag: bool = config.RESIDUAL_STACK_INDEPENDENT_FLAG
    residual_stack_hidden_dim: int | None = config.RESIDUAL_STACK_HIDDEN_DIM
    residual_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.RESIDUAL_STACK_LAYER_NORM_POSITION
    )
    residual_stack_num_layers: int | None = config.RESIDUAL_STACK_NUM_LAYERS
    residual_stack_activation: ActivationOptions | None = (
        config.RESIDUAL_STACK_ACTIVATION
    )
    residual_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION
    )
    residual_stack_residual_model_flag: bool = config.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG
    residual_stack_dropout_probability: float | None = (
        config.RESIDUAL_STACK_DROPOUT_PROBABILITY
    )
    residual_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION
    )
    residual_stack_apply_output_postprocessing_flag: bool | None = (
        config.RESIDUAL_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    residual_stack_bias_flag: bool | None = config.RESIDUAL_STACK_BIAS_FLAG
    expert_stack_gate_flag: bool = config.EXPERT_STACK_GATE_FLAG
    expert_gate_option: LayerGateOptions | None = config.EXPERT_GATE_OPTION
    expert_gate_activation: ActivationOptions | None = config.EXPERT_GATE_ACTIVATION
    expert_gate_stack_independent_flag: bool = config.EXPERT_GATE_STACK_INDEPENDENT_FLAG
    expert_gate_stack_hidden_dim: int | None = config.EXPERT_GATE_STACK_HIDDEN_DIM
    expert_gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.EXPERT_GATE_STACK_LAYER_NORM_POSITION
    )
    expert_gate_stack_num_layers: int | None = config.EXPERT_GATE_STACK_NUM_LAYERS
    expert_gate_stack_activation: ActivationOptions | None = (
        config.EXPERT_GATE_STACK_ACTIVATION
    )
    expert_gate_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
    )
    expert_gate_stack_residual_model_flag: bool = (
        config.EXPERT_GATE_STACK_RESIDUAL_MODEL_FLAG
    )
    expert_gate_stack_dropout_probability: float | None = (
        config.EXPERT_GATE_STACK_DROPOUT_PROBABILITY
    )
    expert_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    expert_gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_gate_stack_bias_flag: bool | None = config.EXPERT_GATE_STACK_BIAS_FLAG
    expert_stack_halting_flag: bool = config.EXPERT_STACK_HALTING_FLAG
    expert_halting_threshold: float = config.EXPERT_HALTING_THRESHOLD
    expert_halting_dropout: float = config.EXPERT_HALTING_DROPOUT
    expert_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.EXPERT_HALTING_HIDDEN_STATE_MODE
    )
    expert_halting_output_dim: int = config.EXPERT_HALTING_OUTPUT_DIM
    expert_halting_stack_independent_flag: bool = (
        config.EXPERT_HALTING_STACK_INDEPENDENT_FLAG
    )
    expert_halting_stack_hidden_dim: int | None = config.EXPERT_HALTING_STACK_HIDDEN_DIM
    expert_halting_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.EXPERT_HALTING_STACK_LAYER_NORM_POSITION
    )
    expert_halting_stack_num_layers: int | None = config.EXPERT_HALTING_STACK_NUM_LAYERS
    expert_halting_stack_activation: ActivationOptions | None = (
        config.EXPERT_HALTING_STACK_ACTIVATION
    )
    expert_halting_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    )
    expert_halting_stack_residual_model_flag: bool = (
        config.EXPERT_HALTING_STACK_RESIDUAL_MODEL_FLAG
    )
    expert_halting_stack_dropout_probability: float | None = (
        config.EXPERT_HALTING_STACK_DROPOUT_PROBABILITY
    )
    expert_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
    )
    expert_halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_halting_stack_bias_flag: bool | None = config.EXPERT_HALTING_STACK_BIAS_FLAG
    expert_memory_flag: bool = config.EXPERT_MEMORY_FLAG
    expert_memory_option: type[DynamicMemoryConfig] = config.EXPERT_MEMORY_OPTION
    expert_memory_position_option: MemoryPositionOptions = (
        config.EXPERT_MEMORY_POSITION_OPTION
    )
    expert_memory_test_time_training_learning_rate: float | None = (
        config.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
    )
    expert_memory_test_time_training_num_inner_steps: int | None = (
        config.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
    )
    expert_memory_stack_independent_flag: bool = (
        config.EXPERT_MEMORY_STACK_INDEPENDENT_FLAG
    )
    expert_memory_stack_hidden_dim: int | None = config.EXPERT_MEMORY_STACK_HIDDEN_DIM
    expert_memory_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.EXPERT_MEMORY_STACK_LAYER_NORM_POSITION
    )
    expert_memory_stack_num_layers: int | None = config.EXPERT_MEMORY_STACK_NUM_LAYERS
    expert_memory_stack_activation: ActivationOptions | None = (
        config.EXPERT_MEMORY_STACK_ACTIVATION
    )
    expert_memory_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
    )
    expert_memory_stack_residual_model_flag: bool = (
        config.EXPERT_MEMORY_STACK_RESIDUAL_MODEL_FLAG
    )
    expert_memory_stack_dropout_probability: float | None = (
        config.EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY
    )
    expert_memory_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION
    )
    expert_memory_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_memory_stack_bias_flag: bool | None = config.EXPERT_MEMORY_STACK_BIAS_FLAG
    expert_recurrent_flag: bool = config.EXPERT_RECURRENT_FLAG
    expert_recurrent_max_steps: int = config.EXPERT_RECURRENT_MAX_STEPS
    expert_recurrent_layer_norm_position: LayerNormPositionOptions = (
        config.EXPERT_RECURRENT_LAYER_NORM_POSITION
    )
    expert_recurrent_stack_gate_flag: bool = config.EXPERT_RECURRENT_STACK_GATE_FLAG
    expert_recurrent_gate_option: LayerGateOptions | None = (
        config.EXPERT_RECURRENT_GATE_OPTION
    )
    expert_recurrent_gate_activation: ActivationOptions | None = (
        config.EXPERT_RECURRENT_GATE_ACTIVATION
    )
    expert_recurrent_gate_stack_independent_flag: bool = (
        config.EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG
    )
    expert_recurrent_gate_stack_hidden_dim: int | None = (
        config.EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM
    )
    expert_recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION
    )
    expert_recurrent_gate_stack_num_layers: int | None = (
        config.EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS
    )
    expert_recurrent_gate_stack_activation: ActivationOptions | None = (
        config.EXPERT_RECURRENT_GATE_STACK_ACTIVATION
    )
    expert_recurrent_gate_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
    expert_recurrent_gate_stack_residual_model_flag: bool = (
        config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
    )
    expert_recurrent_gate_stack_dropout_probability: float | None = (
        config.EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
    )
    expert_recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    expert_recurrent_gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_recurrent_gate_stack_bias_flag: bool | None = (
        config.EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG
    )
    expert_recurrent_stack_halting_flag: bool = (
        config.EXPERT_RECURRENT_STACK_HALTING_FLAG
    )
    expert_recurrent_halting_threshold: float = (
        config.EXPERT_RECURRENT_HALTING_THRESHOLD
    )
    expert_recurrent_halting_dropout: float = config.EXPERT_RECURRENT_HALTING_DROPOUT
    expert_recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
    )
    expert_recurrent_halting_stack_independent_flag: bool = (
        config.EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG
    )
    expert_recurrent_halting_stack_hidden_dim: int | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM
    )
    expert_recurrent_halting_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    ) = config.EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
    expert_recurrent_halting_stack_num_layers: int | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS
    )
    expert_recurrent_halting_stack_activation: ActivationOptions | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_ACTIVATION
    )
    expert_recurrent_halting_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    expert_recurrent_halting_stack_residual_model_flag: bool = (
        config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
    )
    expert_recurrent_halting_stack_dropout_probability: float | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
    )
    expert_recurrent_halting_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    ) = config.EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
    expert_recurrent_halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_recurrent_halting_stack_bias_flag: bool | None = (
        config.EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG
    )
    router_noisy_topk_flag: bool = config.ROUTER_NOISY_TOPK_FLAG
    router_stack_hidden_dim: int = config.ROUTER_STACK_HIDDEN_DIM
    router_stack_num_layers: int = config.ROUTER_STACK_NUM_LAYERS
    router_stack_activation: ActivationOptions = config.ROUTER_STACK_ACTIVATION
    router_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION
    )
    router_stack_residual_model_flag: bool = config.ROUTER_STACK_RESIDUAL_MODEL_FLAG
    router_stack_dropout_probability: float = config.ROUTER_STACK_DROPOUT_PROBABILITY
    router_stack_layer_norm_position: LayerNormPositionOptions = (
        config.ROUTER_STACK_LAYER_NORM_POSITION
    )
    router_stack_last_layer_bias_option: LastLayerBiasOptions = (
        config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_stack_apply_output_postprocessing_flag: bool = (
        config.ROUTER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_bias_flag: bool = config.ROUTER_BIAS_FLAG
    router_stack_gate_flag: bool = config.ROUTER_STACK_GATE_FLAG
    router_gate_option: LayerGateOptions | None = config.ROUTER_GATE_OPTION
    router_gate_activation: ActivationOptions | None = config.ROUTER_GATE_ACTIVATION
    router_gate_stack_independent_flag: bool = config.ROUTER_GATE_STACK_INDEPENDENT_FLAG
    router_gate_stack_hidden_dim: int | None = config.ROUTER_GATE_STACK_HIDDEN_DIM
    router_gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_GATE_STACK_LAYER_NORM_POSITION
    )
    router_gate_stack_num_layers: int | None = config.ROUTER_GATE_STACK_NUM_LAYERS
    router_gate_stack_activation: ActivationOptions | None = (
        config.ROUTER_GATE_STACK_ACTIVATION
    )
    router_gate_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.ROUTER_GATE_STACK_RESIDUAL_CONNECTION_OPTION
    )
    router_gate_stack_residual_model_flag: bool = (
        config.ROUTER_GATE_STACK_RESIDUAL_MODEL_FLAG
    )
    router_gate_stack_dropout_probability: float | None = (
        config.ROUTER_GATE_STACK_DROPOUT_PROBABILITY
    )
    router_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_gate_stack_bias_flag: bool | None = config.ROUTER_GATE_STACK_BIAS_FLAG
    router_stack_halting_flag: bool = config.ROUTER_STACK_HALTING_FLAG
    router_halting_threshold: float = config.ROUTER_HALTING_THRESHOLD
    router_halting_dropout: float = config.ROUTER_HALTING_DROPOUT
    router_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.ROUTER_HALTING_HIDDEN_STATE_MODE
    )
    router_halting_output_dim: int = config.ROUTER_HALTING_OUTPUT_DIM
    router_halting_stack_independent_flag: bool = (
        config.ROUTER_HALTING_STACK_INDEPENDENT_FLAG
    )
    router_halting_stack_hidden_dim: int | None = config.ROUTER_HALTING_STACK_HIDDEN_DIM
    router_halting_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_HALTING_STACK_LAYER_NORM_POSITION
    )
    router_halting_stack_num_layers: int | None = config.ROUTER_HALTING_STACK_NUM_LAYERS
    router_halting_stack_activation: ActivationOptions | None = (
        config.ROUTER_HALTING_STACK_ACTIVATION
    )
    router_halting_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.ROUTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    )
    router_halting_stack_residual_model_flag: bool = (
        config.ROUTER_HALTING_STACK_RESIDUAL_MODEL_FLAG
    )
    router_halting_stack_dropout_probability: float | None = (
        config.ROUTER_HALTING_STACK_DROPOUT_PROBABILITY
    )
    router_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_halting_stack_bias_flag: bool | None = config.ROUTER_HALTING_STACK_BIAS_FLAG
    router_memory_flag: bool = config.ROUTER_MEMORY_FLAG
    router_memory_option: type[DynamicMemoryConfig] = config.ROUTER_MEMORY_OPTION
    router_memory_position_option: MemoryPositionOptions = (
        config.ROUTER_MEMORY_POSITION_OPTION
    )
    router_memory_test_time_training_learning_rate: float | None = (
        config.ROUTER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
    )
    router_memory_test_time_training_num_inner_steps: int | None = (
        config.ROUTER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
    )
    router_memory_stack_independent_flag: bool = (
        config.ROUTER_MEMORY_STACK_INDEPENDENT_FLAG
    )
    router_memory_stack_hidden_dim: int | None = config.ROUTER_MEMORY_STACK_HIDDEN_DIM
    router_memory_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_MEMORY_STACK_LAYER_NORM_POSITION
    )
    router_memory_stack_num_layers: int | None = config.ROUTER_MEMORY_STACK_NUM_LAYERS
    router_memory_stack_activation: ActivationOptions | None = (
        config.ROUTER_MEMORY_STACK_ACTIVATION
    )
    router_memory_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.ROUTER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
    )
    router_memory_stack_residual_model_flag: bool = (
        config.ROUTER_MEMORY_STACK_RESIDUAL_MODEL_FLAG
    )
    router_memory_stack_dropout_probability: float | None = (
        config.ROUTER_MEMORY_STACK_DROPOUT_PROBABILITY
    )
    router_memory_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_memory_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_memory_stack_bias_flag: bool | None = config.ROUTER_MEMORY_STACK_BIAS_FLAG
    router_recurrent_flag: bool = config.ROUTER_RECURRENT_FLAG
    router_recurrent_max_steps: int = config.ROUTER_RECURRENT_MAX_STEPS
    router_recurrent_layer_norm_position: LayerNormPositionOptions = (
        config.ROUTER_RECURRENT_LAYER_NORM_POSITION
    )
    router_recurrent_stack_gate_flag: bool = config.ROUTER_RECURRENT_STACK_GATE_FLAG
    router_recurrent_gate_option: LayerGateOptions | None = (
        config.ROUTER_RECURRENT_GATE_OPTION
    )
    router_recurrent_gate_activation: ActivationOptions | None = (
        config.ROUTER_RECURRENT_GATE_ACTIVATION
    )
    router_recurrent_gate_stack_independent_flag: bool = (
        config.ROUTER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG
    )
    router_recurrent_gate_stack_hidden_dim: int | None = (
        config.ROUTER_RECURRENT_GATE_STACK_HIDDEN_DIM
    )
    router_recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION
    )
    router_recurrent_gate_stack_num_layers: int | None = (
        config.ROUTER_RECURRENT_GATE_STACK_NUM_LAYERS
    )
    router_recurrent_gate_stack_activation: ActivationOptions | None = (
        config.ROUTER_RECURRENT_GATE_STACK_ACTIVATION
    )
    router_recurrent_gate_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
    router_recurrent_gate_stack_residual_model_flag: bool = (
        config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
    )
    router_recurrent_gate_stack_dropout_probability: float | None = (
        config.ROUTER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
    )
    router_recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_recurrent_gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_recurrent_gate_stack_bias_flag: bool | None = (
        config.ROUTER_RECURRENT_GATE_STACK_BIAS_FLAG
    )
    router_recurrent_stack_halting_flag: bool = (
        config.ROUTER_RECURRENT_STACK_HALTING_FLAG
    )
    router_recurrent_halting_threshold: float = (
        config.ROUTER_RECURRENT_HALTING_THRESHOLD
    )
    router_recurrent_halting_dropout: float = config.ROUTER_RECURRENT_HALTING_DROPOUT
    router_recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE
    )
    router_recurrent_halting_stack_independent_flag: bool = (
        config.ROUTER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG
    )
    router_recurrent_halting_stack_hidden_dim: int | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_HIDDEN_DIM
    )
    router_recurrent_halting_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    ) = config.ROUTER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
    router_recurrent_halting_stack_num_layers: int | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_NUM_LAYERS
    )
    router_recurrent_halting_stack_activation: ActivationOptions | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_ACTIVATION
    )
    router_recurrent_halting_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    router_recurrent_halting_stack_residual_model_flag: bool = (
        config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
    )
    router_recurrent_halting_stack_dropout_probability: float | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
    )
    router_recurrent_halting_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    ) = config.ROUTER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
    router_recurrent_halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_recurrent_halting_stack_bias_flag: bool | None = (
        config.ROUTER_RECURRENT_HALTING_STACK_BIAS_FLAG
    )
    stack_gate_flag: bool = config.STACK_GATE_FLAG
    gate_option: LayerGateOptions | None = config.GATE_OPTION
    gate_activation: ActivationOptions | None = config.GATE_ACTIVATION
    gate_stack_independent_flag: bool = config.GATE_STACK_INDEPENDENT_FLAG
    gate_stack_hidden_dim: int | None = config.GATE_STACK_HIDDEN_DIM
    gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.GATE_STACK_LAYER_NORM_POSITION
    )
    gate_stack_num_layers: int | None = config.GATE_STACK_NUM_LAYERS
    gate_stack_activation: ActivationOptions | None = config.GATE_STACK_ACTIVATION
    gate_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.GATE_STACK_RESIDUAL_CONNECTION_OPTION
    )
    gate_stack_residual_model_flag: bool = config.GATE_STACK_RESIDUAL_MODEL_FLAG
    gate_stack_dropout_probability: float | None = config.GATE_STACK_DROPOUT_PROBABILITY
    gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    gate_stack_bias_flag: bool | None = config.GATE_STACK_BIAS_FLAG
    stack_halting_flag: bool = config.STACK_HALTING_FLAG
    halting_threshold: float = config.HALTING_THRESHOLD
    halting_dropout: float = config.HALTING_DROPOUT
    halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.HALTING_HIDDEN_STATE_MODE
    )
    halting_output_dim: int = config.HALTING_OUTPUT_DIM
    halting_stack_independent_flag: bool = config.HALTING_STACK_INDEPENDENT_FLAG
    halting_stack_hidden_dim: int | None = config.HALTING_STACK_HIDDEN_DIM
    halting_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.HALTING_STACK_LAYER_NORM_POSITION
    )
    halting_stack_num_layers: int | None = config.HALTING_STACK_NUM_LAYERS
    halting_stack_activation: ActivationOptions | None = config.HALTING_STACK_ACTIVATION
    halting_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    )
    halting_stack_residual_model_flag: bool = config.HALTING_STACK_RESIDUAL_MODEL_FLAG
    halting_stack_dropout_probability: float | None = (
        config.HALTING_STACK_DROPOUT_PROBABILITY
    )
    halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.HALTING_STACK_LAST_LAYER_BIAS_OPTION
    )
    halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    halting_stack_bias_flag: bool | None = config.HALTING_STACK_BIAS_FLAG
    memory_flag: bool = config.MEMORY_FLAG
    memory_option: type[DynamicMemoryConfig] = config.MEMORY_OPTION
    memory_position_option: MemoryPositionOptions = config.MEMORY_POSITION_OPTION
    memory_test_time_training_learning_rate: float | None = (
        config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
    )
    memory_test_time_training_num_inner_steps: int | None = (
        config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
    )
    memory_stack_independent_flag: bool = config.MEMORY_STACK_INDEPENDENT_FLAG
    memory_stack_hidden_dim: int | None = config.MEMORY_STACK_HIDDEN_DIM
    memory_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.MEMORY_STACK_LAYER_NORM_POSITION
    )
    memory_stack_num_layers: int | None = config.MEMORY_STACK_NUM_LAYERS
    memory_stack_activation: ActivationOptions | None = config.MEMORY_STACK_ACTIVATION
    memory_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
    )
    memory_stack_residual_model_flag: bool = config.MEMORY_STACK_RESIDUAL_MODEL_FLAG
    memory_stack_dropout_probability: float | None = (
        config.MEMORY_STACK_DROPOUT_PROBABILITY
    )
    memory_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION
    )
    memory_stack_apply_output_postprocessing_flag: bool | None = (
        config.MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    memory_stack_bias_flag: bool | None = config.MEMORY_STACK_BIAS_FLAG
    recurrent_flag: bool = config.RECURRENT_FLAG
    recurrent_max_steps: int = config.RECURRENT_MAX_STEPS
    recurrent_initial_iterations: int | None = config.RECURRENT_INITIAL_ITERATIONS
    recurrent_gradient_transition_count: int | None = (
        config.RECURRENT_GRADIENT_TRANSITION_COUNT
    )
    recurrent_no_gradient_transition_count: int | None = (
        config.RECURRENT_NO_GRADIENT_TRANSITION_COUNT
    )
    recurrent_iteration_increment: int = config.RECURRENT_ITERATION_INCREMENT
    recurrent_forward_calls_before_iteration_increment: int = (
        config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT
    )
    recurrent_smooth_iteration_growth_flag: bool = (
        config.RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG
    )
    recurrent_layer_norm_position: LayerNormPositionOptions = (
        config.RECURRENT_LAYER_NORM_POSITION
    )
    recurrent_stack_gate_flag: bool = config.RECURRENT_STACK_GATE_FLAG
    recurrent_gate_option: LayerGateOptions | None = config.RECURRENT_GATE_OPTION
    recurrent_gate_activation: ActivationOptions | None = (
        config.RECURRENT_GATE_ACTIVATION
    )
    recurrent_gate_stack_independent_flag: bool = (
        config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG
    )
    recurrent_gate_stack_hidden_dim: int | None = config.RECURRENT_GATE_STACK_HIDDEN_DIM
    recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION
    )
    recurrent_gate_stack_num_layers: int | None = config.RECURRENT_GATE_STACK_NUM_LAYERS
    recurrent_gate_stack_activation: ActivationOptions | None = (
        config.RECURRENT_GATE_STACK_ACTIVATION
    )
    recurrent_gate_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
    )
    recurrent_gate_stack_residual_model_flag: bool = (
        config.RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
    )
    recurrent_gate_stack_dropout_probability: float | None = (
        config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
    )
    recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
    )
    recurrent_gate_stack_apply_output_postprocessing_flag: bool | None = (
        config.RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    recurrent_gate_stack_bias_flag: bool | None = config.RECURRENT_GATE_STACK_BIAS_FLAG
    recurrent_stack_halting_flag: bool = config.RECURRENT_STACK_HALTING_FLAG
    recurrent_halting_threshold: float = config.RECURRENT_HALTING_THRESHOLD
    recurrent_halting_dropout: float = config.RECURRENT_HALTING_DROPOUT
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        config.RECURRENT_HALTING_HIDDEN_STATE_MODE
    )
    recurrent_halting_stack_independent_flag: bool = (
        config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG
    )
    recurrent_halting_stack_hidden_dim: int | None = (
        config.RECURRENT_HALTING_STACK_HIDDEN_DIM
    )
    recurrent_halting_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
    )
    recurrent_halting_stack_num_layers: int | None = (
        config.RECURRENT_HALTING_STACK_NUM_LAYERS
    )
    recurrent_halting_stack_activation: ActivationOptions | None = (
        config.RECURRENT_HALTING_STACK_ACTIVATION
    )
    recurrent_halting_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
    )
    recurrent_halting_stack_residual_model_flag: bool = (
        config.RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
    )
    recurrent_halting_stack_dropout_probability: float | None = (
        config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
    )
    recurrent_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
    )
    recurrent_halting_stack_apply_output_postprocessing_flag: bool | None = (
        config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    recurrent_halting_stack_bias_flag: bool | None = (
        config.RECURRENT_HALTING_STACK_BIAS_FLAG
    )
    shared_gate_config: GateConfig | None = None
    router_options: ExpertsRouterOptions | None = None
    router_stack_options: ExpertsSubmoduleStackOptions | None = None
    router_layer_controller_options: ExpertsLayerControllerOptions | None = None
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None
    layer_controller_options: ExpertsLayerControllerOptions | None = None
    dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None = None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None
    recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None
    halting_option: type[HaltingConfig] = config.HALTING_OPTION
    expert_halting_option: type[HaltingConfig] = config.EXPERT_HALTING_OPTION
    router_halting_option: type[HaltingConfig] = config.ROUTER_HALTING_OPTION
    recurrent_halting_option: type[HaltingConfig] = config.RECURRENT_HALTING_OPTION
    expert_recurrent_halting_option: type[HaltingConfig] = (
        config.EXPERT_RECURRENT_HALTING_OPTION
    )
    router_recurrent_halting_option: type[HaltingConfig] = (
        config.ROUTER_RECURRENT_HALTING_OPTION
    )


@dataclass(frozen=True, slots=True)
class _ExpertControlDefaults:
    layer: ExpertsLayerControllerOptions
    memory: ExpertsDynamicMemoryOptions
    recurrent: ExpertsRecurrentControllerOptions


@dataclass(frozen=True, slots=True)
class _RouterControlDefaults:
    router: ExpertsRouterOptions
    stack: ExpertsSubmoduleStackOptions
    layer: ExpertsLayerControllerOptions
    memory: ExpertsDynamicMemoryOptions
    recurrent: ExpertsRecurrentControllerOptions


@dataclass(frozen=True, slots=True)
class _LayerControlDefaults:
    layer: ExpertsLayerControllerOptions
    memory: ExpertsDynamicMemoryOptions


@dataclass(frozen=True, slots=True)
class _RecurrentDefaults:
    recurrent: ExpertsRecurrentControllerOptions
    residual_stack: ResidualStackOptions


@dataclass(frozen=True, slots=True)
class ControlDefaults:
    expert_layer_controller_options: ExpertsLayerControllerOptions
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions
    router_options: ExpertsRouterOptions
    router_stack_options: ExpertsSubmoduleStackOptions
    router_layer_controller_options: ExpertsLayerControllerOptions
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions
    layer_controller_options: ExpertsLayerControllerOptions
    dynamic_memory_options: ExpertsDynamicMemoryOptions
    recurrent_controller_options: ExpertsRecurrentControllerOptions
    residual_stack_options: ResidualStackOptions


@dataclass(frozen=True, slots=True)
class _StackSourceValues:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float | None
    bias_flag: bool | None


@dataclass(frozen=True, slots=True)
class _LayerControllerValues:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    gate_stack_source: ExpertsSubmoduleStackSource
    stack_halting_flag: bool
    halting_option: type[HaltingConfig]
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    halting_stack_source: ExpertsSubmoduleStackSource
    halting_output_dim: int


@dataclass(frozen=True, slots=True)
class _MemoryValues:
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_test_time_training_learning_rate: float | None
    memory_test_time_training_num_inner_steps: int | None
    memory_stack_source: ExpertsSubmoduleStackSource


@dataclass(frozen=True, slots=True)
class _RecurrentControllerValues:
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_stack_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_gate_stack_source: ExpertsSubmoduleStackSource
    recurrent_stack_halting_flag: bool
    recurrent_halting_option: type[HaltingConfig]
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    recurrent_halting_stack_source: ExpertsSubmoduleStackSource


def _stack_source(values: _StackSourceValues) -> ExpertsSubmoduleStackSource:
    return ExpertsSubmoduleStackSource(
        independent_flag=values.independent_flag,
        hidden_dim=values.hidden_dim,
        num_layers=values.num_layers,
        last_layer_bias_option=values.last_layer_bias_option,
        apply_output_postprocessing_flag=values.apply_output_postprocessing_flag,
        activation=values.activation,
        layer_norm_position=values.layer_norm_position,
        residual_connection_option=values.residual_connection_option,
        residual_model_flag=values.residual_model_flag,
        dropout_probability=values.dropout_probability,
        bias_flag=values.bias_flag,
    )


def _layer_controller_options(
    values: _LayerControllerValues,
) -> ExpertsLayerControllerOptions:
    return ExpertsLayerControllerOptions(
        stack_gate_flag=values.stack_gate_flag,
        gate_option=values.gate_option,
        gate_activation=values.gate_activation,
        gate_stack_source=values.gate_stack_source,
        stack_halting_flag=values.stack_halting_flag,
        halting_option=values.halting_option,
        halting_threshold=values.halting_threshold,
        halting_dropout=values.halting_dropout,
        halting_hidden_state_mode=values.halting_hidden_state_mode,
        halting_stack_source=values.halting_stack_source,
        halting_output_dim=values.halting_output_dim,
    )


def _dynamic_memory_options(values: _MemoryValues) -> ExpertsDynamicMemoryOptions:
    return ExpertsDynamicMemoryOptions(
        memory_flag=values.memory_flag,
        memory_option=values.memory_option,
        memory_position_option=values.memory_position_option,
        memory_test_time_training_learning_rate=(
            values.memory_test_time_training_learning_rate
        ),
        memory_test_time_training_num_inner_steps=(
            values.memory_test_time_training_num_inner_steps
        ),
        memory_stack_source=values.memory_stack_source,
    )


def _recurrent_controller_options(
    values: _RecurrentControllerValues,
) -> ExpertsRecurrentControllerOptions:
    return ExpertsRecurrentControllerOptions(
        recurrent_flag=values.recurrent_flag,
        recurrent_max_steps=values.recurrent_max_steps,
        recurrent_layer_norm_position=values.recurrent_layer_norm_position,
        recurrent_stack_gate_flag=values.recurrent_stack_gate_flag,
        recurrent_gate_option=values.recurrent_gate_option,
        recurrent_gate_activation=values.recurrent_gate_activation,
        recurrent_gate_stack_source=values.recurrent_gate_stack_source,
        recurrent_stack_halting_flag=values.recurrent_stack_halting_flag,
        recurrent_halting_option=values.recurrent_halting_option,
        recurrent_halting_threshold=values.recurrent_halting_threshold,
        recurrent_halting_dropout=values.recurrent_halting_dropout,
        recurrent_halting_hidden_state_mode=(
            values.recurrent_halting_hidden_state_mode
        ),
        recurrent_halting_stack_source=values.recurrent_halting_stack_source,
    )


def _expert_control_defaults(values: ControlDefaultValues) -> _ExpertControlDefaults:
    expert_layer_controller_options = cast(
        ExpertsLayerControllerOptions | None, values.expert_layer_controller_options
    )
    expert_dynamic_memory_options = cast(
        ExpertsDynamicMemoryOptions | None, values.expert_dynamic_memory_options
    )
    expert_recurrent_controller_options = cast(
        ExpertsRecurrentControllerOptions | None,
        values.expert_recurrent_controller_options,
    )
    expert_layer_controller_options = (
        expert_layer_controller_options
        or _layer_controller_options(
            _LayerControllerValues(
                stack_gate_flag=values.expert_stack_gate_flag,
                gate_option=values.expert_gate_option,
                gate_activation=values.expert_gate_activation,
                gate_stack_source=_stack_source(
                    _StackSourceValues(
                        independent_flag=values.expert_gate_stack_independent_flag,
                        hidden_dim=values.expert_gate_stack_hidden_dim,
                        num_layers=values.expert_gate_stack_num_layers,
                        last_layer_bias_option=(
                            values.expert_gate_stack_last_layer_bias_option
                        ),
                        apply_output_postprocessing_flag=(
                            values.expert_gate_stack_apply_output_postprocessing_flag
                        ),
                        activation=values.expert_gate_stack_activation,
                        layer_norm_position=values.expert_gate_stack_layer_norm_position,
                        residual_connection_option=(
                            values.expert_gate_stack_residual_connection_option
                        ),
                        residual_model_flag=values.expert_gate_stack_residual_model_flag,
                        dropout_probability=values.expert_gate_stack_dropout_probability,
                        bias_flag=values.expert_gate_stack_bias_flag,
                    )
                ),
                stack_halting_flag=values.expert_stack_halting_flag,
                halting_option=values.expert_halting_option,
                halting_threshold=values.expert_halting_threshold,
                halting_dropout=values.expert_halting_dropout,
                halting_hidden_state_mode=values.expert_halting_hidden_state_mode,
                halting_stack_source=_stack_source(
                    _StackSourceValues(
                        independent_flag=values.expert_halting_stack_independent_flag,
                        hidden_dim=values.expert_halting_stack_hidden_dim,
                        num_layers=values.expert_halting_stack_num_layers,
                        last_layer_bias_option=(
                            values.expert_halting_stack_last_layer_bias_option
                        ),
                        apply_output_postprocessing_flag=(
                            values.expert_halting_stack_apply_output_postprocessing_flag
                        ),
                        activation=values.expert_halting_stack_activation,
                        layer_norm_position=values.expert_halting_stack_layer_norm_position,
                        residual_connection_option=(
                            values.expert_halting_stack_residual_connection_option
                        ),
                        residual_model_flag=values.expert_halting_stack_residual_model_flag,
                        dropout_probability=values.expert_halting_stack_dropout_probability,
                        bias_flag=values.expert_halting_stack_bias_flag,
                    )
                ),
                halting_output_dim=values.expert_halting_output_dim,
            )
        )
    )
    expert_dynamic_memory_options = (
        expert_dynamic_memory_options
        or _dynamic_memory_options(
            _MemoryValues(
                memory_flag=values.expert_memory_flag,
                memory_option=values.expert_memory_option,
                memory_position_option=values.expert_memory_position_option,
                memory_test_time_training_learning_rate=(
                    values.expert_memory_test_time_training_learning_rate
                ),
                memory_test_time_training_num_inner_steps=(
                    values.expert_memory_test_time_training_num_inner_steps
                ),
                memory_stack_source=_stack_source(
                    _StackSourceValues(
                        independent_flag=values.expert_memory_stack_independent_flag,
                        hidden_dim=values.expert_memory_stack_hidden_dim,
                        num_layers=values.expert_memory_stack_num_layers,
                        last_layer_bias_option=(
                            values.expert_memory_stack_last_layer_bias_option
                        ),
                        apply_output_postprocessing_flag=(
                            values.expert_memory_stack_apply_output_postprocessing_flag
                        ),
                        activation=values.expert_memory_stack_activation,
                        layer_norm_position=values.expert_memory_stack_layer_norm_position,
                        residual_connection_option=(
                            values.expert_memory_stack_residual_connection_option
                        ),
                        residual_model_flag=values.expert_memory_stack_residual_model_flag,
                        dropout_probability=values.expert_memory_stack_dropout_probability,
                        bias_flag=values.expert_memory_stack_bias_flag,
                    )
                ),
            )
        )
    )
    expert_recurrent_controller_options = (
        expert_recurrent_controller_options
        or _recurrent_controller_options(
            _RecurrentControllerValues(
                recurrent_flag=values.expert_recurrent_flag,
                recurrent_max_steps=values.expert_recurrent_max_steps,
                recurrent_layer_norm_position=(
                    values.expert_recurrent_layer_norm_position
                ),
                recurrent_stack_gate_flag=values.expert_recurrent_stack_gate_flag,
                recurrent_gate_option=values.expert_recurrent_gate_option,
                recurrent_gate_activation=values.expert_recurrent_gate_activation,
                recurrent_gate_stack_source=_stack_source(
                    _StackSourceValues(
                        independent_flag=(
                            values.expert_recurrent_gate_stack_independent_flag
                        ),
                        hidden_dim=values.expert_recurrent_gate_stack_hidden_dim,
                        num_layers=values.expert_recurrent_gate_stack_num_layers,
                        last_layer_bias_option=(
                            values.expert_recurrent_gate_stack_last_layer_bias_option
                        ),
                        apply_output_postprocessing_flag=(
                            values.expert_recurrent_gate_stack_apply_output_postprocessing_flag
                        ),
                        activation=values.expert_recurrent_gate_stack_activation,
                        layer_norm_position=(
                            values.expert_recurrent_gate_stack_layer_norm_position
                        ),
                        residual_connection_option=(
                            values.expert_recurrent_gate_stack_residual_connection_option
                        ),
                        residual_model_flag=values.expert_recurrent_gate_stack_residual_model_flag,
                        dropout_probability=(
                            values.expert_recurrent_gate_stack_dropout_probability
                        ),
                        bias_flag=values.expert_recurrent_gate_stack_bias_flag,
                    )
                ),
                recurrent_stack_halting_flag=values.expert_recurrent_stack_halting_flag,
                recurrent_halting_option=values.expert_recurrent_halting_option,
                recurrent_halting_threshold=(values.expert_recurrent_halting_threshold),
                recurrent_halting_dropout=values.expert_recurrent_halting_dropout,
                recurrent_halting_hidden_state_mode=(
                    values.expert_recurrent_halting_hidden_state_mode
                ),
                recurrent_halting_stack_source=_stack_source(
                    _StackSourceValues(
                        independent_flag=(
                            values.expert_recurrent_halting_stack_independent_flag
                        ),
                        hidden_dim=values.expert_recurrent_halting_stack_hidden_dim,
                        num_layers=values.expert_recurrent_halting_stack_num_layers,
                        last_layer_bias_option=(
                            values.expert_recurrent_halting_stack_last_layer_bias_option
                        ),
                        apply_output_postprocessing_flag=(
                            values.expert_recurrent_halting_stack_apply_output_postprocessing_flag
                        ),
                        activation=values.expert_recurrent_halting_stack_activation,
                        layer_norm_position=(
                            values.expert_recurrent_halting_stack_layer_norm_position
                        ),
                        residual_connection_option=(
                            values.expert_recurrent_halting_stack_residual_connection_option
                        ),
                        residual_model_flag=values.expert_recurrent_halting_stack_residual_model_flag,
                        dropout_probability=(
                            values.expert_recurrent_halting_stack_dropout_probability
                        ),
                        bias_flag=values.expert_recurrent_halting_stack_bias_flag,
                    )
                ),
            )
        )
    )
    return _ExpertControlDefaults(
        layer=expert_layer_controller_options,
        memory=expert_dynamic_memory_options,
        recurrent=expert_recurrent_controller_options,
    )


def _router_control_defaults(values: ControlDefaultValues) -> _RouterControlDefaults:
    return _RouterControlDefaults(
        router=_router_options_defaults(values),
        stack=_router_stack_defaults(values),
        layer=_router_layer_defaults(values),
        memory=_router_memory_defaults(values),
        recurrent=_router_recurrent_defaults(values),
    )


def _router_options_defaults(values: ControlDefaultValues) -> ExpertsRouterOptions:
    provided = cast(ExpertsRouterOptions | None, values.router_options)
    return provided or ExpertsRouterOptions(
        noisy_topk_flag=values.router_noisy_topk_flag,
    )


def _router_stack_defaults(
    values: ControlDefaultValues,
) -> ExpertsSubmoduleStackOptions:
    provided = cast(
        ExpertsSubmoduleStackOptions | None,
        values.router_stack_options,
    )
    return provided or ExpertsSubmoduleStackOptions(
        hidden_dim=values.router_stack_hidden_dim,
        num_layers=values.router_stack_num_layers,
        last_layer_bias_option=values.router_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=values.router_stack_apply_output_postprocessing_flag,
        activation=values.router_stack_activation,
        layer_norm_position=values.router_stack_layer_norm_position,
        residual_connection_option=values.router_stack_residual_connection_option,
        residual_model_flag=values.router_stack_residual_model_flag,
        dropout_probability=values.router_stack_dropout_probability,
        bias_flag=values.router_bias_flag,
    )


def _router_layer_defaults(
    values: ControlDefaultValues,
) -> ExpertsLayerControllerOptions:
    provided = cast(
        ExpertsLayerControllerOptions | None,
        values.router_layer_controller_options,
    )
    return provided or _layer_controller_options(
        _LayerControllerValues(
            stack_gate_flag=values.router_stack_gate_flag,
            gate_option=values.router_gate_option,
            gate_activation=values.router_gate_activation,
            gate_stack_source=_stack_source(
                _StackSourceValues(
                    independent_flag=values.router_gate_stack_independent_flag,
                    hidden_dim=values.router_gate_stack_hidden_dim,
                    num_layers=values.router_gate_stack_num_layers,
                    last_layer_bias_option=(
                        values.router_gate_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.router_gate_stack_apply_output_postprocessing_flag
                    ),
                    activation=values.router_gate_stack_activation,
                    layer_norm_position=values.router_gate_stack_layer_norm_position,
                    residual_connection_option=(
                        values.router_gate_stack_residual_connection_option
                    ),
                    residual_model_flag=values.router_gate_stack_residual_model_flag,
                    dropout_probability=values.router_gate_stack_dropout_probability,
                    bias_flag=values.router_gate_stack_bias_flag,
                )
            ),
            stack_halting_flag=values.router_stack_halting_flag,
            halting_option=values.router_halting_option,
            halting_threshold=values.router_halting_threshold,
            halting_dropout=values.router_halting_dropout,
            halting_hidden_state_mode=values.router_halting_hidden_state_mode,
            halting_stack_source=_stack_source(
                _StackSourceValues(
                    independent_flag=values.router_halting_stack_independent_flag,
                    hidden_dim=values.router_halting_stack_hidden_dim,
                    num_layers=values.router_halting_stack_num_layers,
                    last_layer_bias_option=(
                        values.router_halting_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.router_halting_stack_apply_output_postprocessing_flag
                    ),
                    activation=values.router_halting_stack_activation,
                    layer_norm_position=values.router_halting_stack_layer_norm_position,
                    residual_connection_option=(
                        values.router_halting_stack_residual_connection_option
                    ),
                    residual_model_flag=values.router_halting_stack_residual_model_flag,
                    dropout_probability=values.router_halting_stack_dropout_probability,
                    bias_flag=values.router_halting_stack_bias_flag,
                )
            ),
            halting_output_dim=values.router_halting_output_dim,
        )
    )


def _router_memory_defaults(
    values: ControlDefaultValues,
) -> ExpertsDynamicMemoryOptions:
    provided = cast(
        ExpertsDynamicMemoryOptions | None,
        values.router_dynamic_memory_options,
    )
    return provided or _dynamic_memory_options(
        _MemoryValues(
            memory_flag=values.router_memory_flag,
            memory_option=values.router_memory_option,
            memory_position_option=values.router_memory_position_option,
            memory_test_time_training_learning_rate=(
                values.router_memory_test_time_training_learning_rate
            ),
            memory_test_time_training_num_inner_steps=(
                values.router_memory_test_time_training_num_inner_steps
            ),
            memory_stack_source=_stack_source(
                _StackSourceValues(
                    independent_flag=values.router_memory_stack_independent_flag,
                    hidden_dim=values.router_memory_stack_hidden_dim,
                    num_layers=values.router_memory_stack_num_layers,
                    last_layer_bias_option=(
                        values.router_memory_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.router_memory_stack_apply_output_postprocessing_flag
                    ),
                    activation=values.router_memory_stack_activation,
                    layer_norm_position=values.router_memory_stack_layer_norm_position,
                    residual_connection_option=(
                        values.router_memory_stack_residual_connection_option
                    ),
                    residual_model_flag=values.router_memory_stack_residual_model_flag,
                    dropout_probability=values.router_memory_stack_dropout_probability,
                    bias_flag=values.router_memory_stack_bias_flag,
                )
            ),
        )
    )


def _router_recurrent_defaults(
    values: ControlDefaultValues,
) -> ExpertsRecurrentControllerOptions:
    provided = cast(
        ExpertsRecurrentControllerOptions | None,
        values.router_recurrent_controller_options,
    )
    return provided or _recurrent_controller_options(
        _RecurrentControllerValues(
            recurrent_flag=values.router_recurrent_flag,
            recurrent_max_steps=values.router_recurrent_max_steps,
            recurrent_layer_norm_position=values.router_recurrent_layer_norm_position,
            recurrent_stack_gate_flag=values.router_recurrent_stack_gate_flag,
            recurrent_gate_option=values.router_recurrent_gate_option,
            recurrent_gate_activation=values.router_recurrent_gate_activation,
            recurrent_gate_stack_source=_stack_source(
                _StackSourceValues(
                    independent_flag=(
                        values.router_recurrent_gate_stack_independent_flag
                    ),
                    hidden_dim=values.router_recurrent_gate_stack_hidden_dim,
                    num_layers=values.router_recurrent_gate_stack_num_layers,
                    last_layer_bias_option=(
                        values.router_recurrent_gate_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.router_recurrent_gate_stack_apply_output_postprocessing_flag
                    ),
                    activation=values.router_recurrent_gate_stack_activation,
                    layer_norm_position=(
                        values.router_recurrent_gate_stack_layer_norm_position
                    ),
                    residual_connection_option=(
                        values.router_recurrent_gate_stack_residual_connection_option
                    ),
                    residual_model_flag=(
                        values.router_recurrent_gate_stack_residual_model_flag
                    ),
                    dropout_probability=(
                        values.router_recurrent_gate_stack_dropout_probability
                    ),
                    bias_flag=values.router_recurrent_gate_stack_bias_flag,
                )
            ),
            recurrent_stack_halting_flag=values.router_recurrent_stack_halting_flag,
            recurrent_halting_option=values.router_recurrent_halting_option,
            recurrent_halting_threshold=values.router_recurrent_halting_threshold,
            recurrent_halting_dropout=values.router_recurrent_halting_dropout,
            recurrent_halting_hidden_state_mode=(
                values.router_recurrent_halting_hidden_state_mode
            ),
            recurrent_halting_stack_source=_stack_source(
                _StackSourceValues(
                    independent_flag=(
                        values.router_recurrent_halting_stack_independent_flag
                    ),
                    hidden_dim=values.router_recurrent_halting_stack_hidden_dim,
                    num_layers=values.router_recurrent_halting_stack_num_layers,
                    last_layer_bias_option=(
                        values.router_recurrent_halting_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.router_recurrent_halting_stack_apply_output_postprocessing_flag
                    ),
                    activation=values.router_recurrent_halting_stack_activation,
                    layer_norm_position=(
                        values.router_recurrent_halting_stack_layer_norm_position
                    ),
                    residual_connection_option=(
                        values.router_recurrent_halting_stack_residual_connection_option
                    ),
                    residual_model_flag=(
                        values.router_recurrent_halting_stack_residual_model_flag
                    ),
                    dropout_probability=(
                        values.router_recurrent_halting_stack_dropout_probability
                    ),
                    bias_flag=values.router_recurrent_halting_stack_bias_flag,
                )
            ),
        )
    )


def _layer_control_defaults(values: ControlDefaultValues) -> _LayerControlDefaults:
    layer_controller_options = cast(
        ExpertsLayerControllerOptions | None, values.layer_controller_options
    )
    dynamic_memory_options = cast(
        ExpertsDynamicMemoryOptions | None, values.dynamic_memory_options
    )
    layer_controller_options = (
        layer_controller_options
        or ExpertsLayerControllerOptions(
            stack_gate_flag=values.stack_gate_flag,
            gate_option=values.gate_option,
            gate_activation=values.gate_activation,
            gate_stack_source=ExpertsSubmoduleStackSource(
                independent_flag=values.gate_stack_independent_flag,
                hidden_dim=values.gate_stack_hidden_dim,
                num_layers=values.gate_stack_num_layers,
                last_layer_bias_option=values.gate_stack_last_layer_bias_option,
                apply_output_postprocessing_flag=values.gate_stack_apply_output_postprocessing_flag,
                activation=values.gate_stack_activation,
                layer_norm_position=values.gate_stack_layer_norm_position,
                residual_connection_option=values.gate_stack_residual_connection_option,
                residual_model_flag=values.gate_stack_residual_model_flag,
                dropout_probability=values.gate_stack_dropout_probability,
                bias_flag=values.gate_stack_bias_flag,
            ),
            stack_halting_flag=values.stack_halting_flag,
            halting_option=values.halting_option,
            halting_threshold=values.halting_threshold,
            halting_dropout=values.halting_dropout,
            halting_hidden_state_mode=values.halting_hidden_state_mode,
            halting_stack_source=ExpertsSubmoduleStackSource(
                independent_flag=values.halting_stack_independent_flag,
                hidden_dim=values.halting_stack_hidden_dim,
                num_layers=values.halting_stack_num_layers,
                last_layer_bias_option=values.halting_stack_last_layer_bias_option,
                apply_output_postprocessing_flag=(
                    values.halting_stack_apply_output_postprocessing_flag
                ),
                activation=values.halting_stack_activation,
                layer_norm_position=values.halting_stack_layer_norm_position,
                residual_connection_option=(
                    values.halting_stack_residual_connection_option
                ),
                residual_model_flag=values.halting_stack_residual_model_flag,
                dropout_probability=values.halting_stack_dropout_probability,
                bias_flag=values.halting_stack_bias_flag,
            ),
            halting_output_dim=values.halting_output_dim,
            shared_gate_config=values.shared_gate_config,
        )
    )
    dynamic_memory_options = dynamic_memory_options or ExpertsDynamicMemoryOptions(
        memory_flag=values.memory_flag,
        memory_option=values.memory_option,
        memory_position_option=values.memory_position_option,
        memory_test_time_training_learning_rate=(
            values.memory_test_time_training_learning_rate
        ),
        memory_test_time_training_num_inner_steps=(
            values.memory_test_time_training_num_inner_steps
        ),
        memory_stack_source=ExpertsSubmoduleStackSource(
            independent_flag=values.memory_stack_independent_flag,
            hidden_dim=values.memory_stack_hidden_dim,
            num_layers=values.memory_stack_num_layers,
            last_layer_bias_option=values.memory_stack_last_layer_bias_option,
            apply_output_postprocessing_flag=(
                values.memory_stack_apply_output_postprocessing_flag
            ),
            activation=values.memory_stack_activation,
            layer_norm_position=values.memory_stack_layer_norm_position,
            residual_connection_option=(values.memory_stack_residual_connection_option),
            residual_model_flag=values.memory_stack_residual_model_flag,
            dropout_probability=values.memory_stack_dropout_probability,
            bias_flag=values.memory_stack_bias_flag,
        ),
    )
    return _LayerControlDefaults(
        layer=layer_controller_options,
        memory=dynamic_memory_options,
    )


def _recurrent_defaults(
    values: ControlDefaultValues,
    submodule_stack_options: ExpertsSubmoduleStackOptions,
) -> _RecurrentDefaults:
    recurrent_controller_options = cast(
        ExpertsRecurrentControllerOptions | None,
        values.recurrent_controller_options,
    )
    recurrent_controller_options = (
        recurrent_controller_options
        or ExpertsRecurrentControllerOptions(
            recurrent_flag=values.recurrent_flag,
            recurrent_max_steps=values.recurrent_max_steps,
            recurrent_initial_iterations=values.recurrent_initial_iterations,
            recurrent_gradient_transition_count=values.recurrent_gradient_transition_count,
            recurrent_no_gradient_transition_count=values.recurrent_no_gradient_transition_count,
            recurrent_iteration_increment=values.recurrent_iteration_increment,
            recurrent_forward_calls_before_iteration_increment=(
                values.recurrent_forward_calls_before_iteration_increment
            ),
            recurrent_smooth_iteration_growth_flag=(
                values.recurrent_smooth_iteration_growth_flag
            ),
            recurrent_layer_norm_position=values.recurrent_layer_norm_position,
            recurrent_stack_gate_flag=values.recurrent_stack_gate_flag,
            recurrent_gate_option=values.recurrent_gate_option,
            recurrent_gate_activation=values.recurrent_gate_activation,
            recurrent_gate_stack_source=ExpertsSubmoduleStackSource(
                independent_flag=values.recurrent_gate_stack_independent_flag,
                hidden_dim=values.recurrent_gate_stack_hidden_dim,
                num_layers=values.recurrent_gate_stack_num_layers,
                last_layer_bias_option=(
                    values.recurrent_gate_stack_last_layer_bias_option
                ),
                apply_output_postprocessing_flag=(
                    values.recurrent_gate_stack_apply_output_postprocessing_flag
                ),
                activation=values.recurrent_gate_stack_activation,
                layer_norm_position=values.recurrent_gate_stack_layer_norm_position,
                residual_connection_option=(
                    values.recurrent_gate_stack_residual_connection_option
                ),
                residual_model_flag=values.recurrent_gate_stack_residual_model_flag,
                dropout_probability=values.recurrent_gate_stack_dropout_probability,
                bias_flag=values.recurrent_gate_stack_bias_flag,
            ),
            recurrent_stack_halting_flag=values.recurrent_stack_halting_flag,
            recurrent_halting_option=values.recurrent_halting_option,
            recurrent_halting_threshold=values.recurrent_halting_threshold,
            recurrent_halting_dropout=values.recurrent_halting_dropout,
            recurrent_halting_hidden_state_mode=(
                values.recurrent_halting_hidden_state_mode
            ),
            recurrent_halting_stack_source=ExpertsSubmoduleStackSource(
                independent_flag=values.recurrent_halting_stack_independent_flag,
                hidden_dim=values.recurrent_halting_stack_hidden_dim,
                num_layers=values.recurrent_halting_stack_num_layers,
                last_layer_bias_option=(
                    values.recurrent_halting_stack_last_layer_bias_option
                ),
                apply_output_postprocessing_flag=(
                    values.recurrent_halting_stack_apply_output_postprocessing_flag
                ),
                activation=values.recurrent_halting_stack_activation,
                layer_norm_position=(
                    values.recurrent_halting_stack_layer_norm_position
                ),
                residual_connection_option=(
                    values.recurrent_halting_stack_residual_connection_option
                ),
                residual_model_flag=values.recurrent_halting_stack_residual_model_flag,
                dropout_probability=(
                    values.recurrent_halting_stack_dropout_probability
                ),
                bias_flag=values.recurrent_halting_stack_bias_flag,
            ),
        )
    )
    residual_stack_options = resolve_residual_stack_options(
        ResidualStackSource(
            independent_flag=values.residual_stack_independent_flag,
            hidden_dim=values.residual_stack_hidden_dim,
            num_layers=values.residual_stack_num_layers,
            activation=values.residual_stack_activation,
            layer_norm_position=values.residual_stack_layer_norm_position,
            residual_connection_option=(
                values.residual_stack_residual_connection_option
            ),
            residual_model_flag=values.residual_stack_residual_model_flag,
            dropout_probability=values.residual_stack_dropout_probability,
            last_layer_bias_option=values.residual_stack_last_layer_bias_option,
            apply_output_postprocessing_flag=(
                values.residual_stack_apply_output_postprocessing_flag
            ),
            bias_flag=values.residual_stack_bias_flag,
        ),
        submodule_stack_options,
    )
    return _RecurrentDefaults(
        recurrent=recurrent_controller_options,
        residual_stack=residual_stack_options,
    )


def resolve_control_defaults(
    values: ControlDefaultValues,
    submodule_stack_options: ExpertsSubmoduleStackOptions,
) -> ControlDefaults:
    expert = _expert_control_defaults(values)
    router = _router_control_defaults(values)
    layer = _layer_control_defaults(values)
    recurrent = _recurrent_defaults(values, submodule_stack_options)
    return ControlDefaults(
        expert_layer_controller_options=expert.layer,
        expert_dynamic_memory_options=expert.memory,
        expert_recurrent_controller_options=expert.recurrent,
        router_options=router.router,
        router_stack_options=router.stack,
        router_layer_controller_options=router.layer,
        router_dynamic_memory_options=router.memory,
        router_recurrent_controller_options=router.recurrent,
        layer_controller_options=layer.layer,
        dynamic_memory_options=layer.memory,
        recurrent_controller_options=recurrent.recurrent,
        residual_stack_options=recurrent.residual_stack,
    )


__all__ = ["ControlDefaultValues", "ControlDefaults", "resolve_control_defaults"]

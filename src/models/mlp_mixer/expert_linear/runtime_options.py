from __future__ import annotations

from dataclasses import dataclass

from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConnectionOptions,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    batch_size: int
    learning_rate: float
    input_dim: int
    hidden_dim: int
    output_dim: int
    output_bias_flag: bool
    image_patch_size: int
    input_channels: int
    image_height: int
    patch_dropout_probability: float
    patch_bias_flag: bool
    stack_num_layers: int
    layer_norm_position: LayerNormPositionOptions
    stack_activation: ActivationOptions
    stack_dropout_probability: float
    mixer_residual_connection_option: ResidualConnectionOptions
    stack_residual_connection_option: ResidualConnectionOptions | None
    stack_last_layer_bias_option: LastLayerBiasOptions
    stack_apply_output_pipeline_flag: bool
    stack_bias_flag: bool
    submodule_stack_hidden_dim: int
    submodule_stack_layer_norm_position: LayerNormPositionOptions
    submodule_stack_num_layers: int
    submodule_stack_activation: ActivationOptions
    submodule_stack_residual_connection_option: ResidualConnectionOptions | None
    submodule_stack_dropout_probability: float
    submodule_stack_last_layer_bias_option: LastLayerBiasOptions
    submodule_stack_apply_output_pipeline_flag: bool
    submodule_stack_bias_flag: bool
    controller_stack_hidden_dim: int
    controller_stack_layer_norm_position: LayerNormPositionOptions
    controller_stack_num_layers: int
    controller_stack_activation: ActivationOptions
    controller_stack_residual_connection_option: ResidualConnectionOptions | None
    controller_stack_dropout_probability: float
    controller_stack_last_layer_bias_option: LastLayerBiasOptions
    controller_stack_apply_output_pipeline_flag: bool
    controller_stack_bias_flag: bool
    token_mixer_stack_hidden_dim: int
    token_mixer_num_layers: int
    token_mixer_stack_activation: ActivationOptions
    token_mixer_stack_dropout_probability: float
    token_mixer_stack_layer_norm_position: LayerNormPositionOptions
    token_mixer_stack_residual_connection_option: ResidualConnectionOptions | None
    token_mixer_stack_last_layer_bias_option: LastLayerBiasOptions
    token_mixer_stack_apply_output_pipeline_flag: bool
    channel_mixer_stack_hidden_dim: int
    channel_mixer_num_layers: int
    channel_mixer_stack_activation: ActivationOptions
    channel_mixer_stack_dropout_probability: float
    channel_mixer_stack_layer_norm_position: LayerNormPositionOptions
    channel_mixer_stack_residual_connection_option: ResidualConnectionOptions | None
    channel_mixer_stack_last_layer_bias_option: LastLayerBiasOptions
    channel_mixer_stack_apply_output_pipeline_flag: bool
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    stack_halting_flag: bool
    halting_option: type[HaltingConfig]
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_test_time_training_learning_rate: float | None
    memory_test_time_training_num_inner_steps: int | None
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_residual_connection_option: ResidualConnectionOptions | None
    recurrent_stack_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_stack_halting_flag: bool
    recurrent_halting_option: type[HaltingConfig]
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    recurrent_memory_flag: bool
    gate_stack_independent_flag: bool
    gate_stack_hidden_dim: int | None
    gate_stack_layer_norm_position: LayerNormPositionOptions | None
    gate_stack_num_layers: int | None
    gate_stack_activation: ActivationOptions | None
    gate_stack_residual_connection_option: ResidualConnectionOptions | None
    gate_stack_dropout_probability: float | None
    gate_stack_last_layer_bias_option: LastLayerBiasOptions | None
    gate_stack_apply_output_pipeline_flag: bool | None
    gate_stack_bias_flag: bool | None
    halting_stack_independent_flag: bool
    halting_stack_hidden_dim: int | None
    halting_stack_layer_norm_position: LayerNormPositionOptions | None
    halting_stack_num_layers: int | None
    halting_stack_activation: ActivationOptions | None
    halting_stack_residual_connection_option: ResidualConnectionOptions | None
    halting_stack_dropout_probability: float | None
    halting_stack_last_layer_bias_option: LastLayerBiasOptions | None
    halting_stack_apply_output_pipeline_flag: bool | None
    halting_stack_bias_flag: bool | None
    memory_stack_independent_flag: bool
    memory_stack_hidden_dim: int | None
    memory_stack_layer_norm_position: LayerNormPositionOptions | None
    memory_stack_num_layers: int | None
    memory_stack_activation: ActivationOptions | None
    memory_stack_residual_connection_option: ResidualConnectionOptions | None
    memory_stack_dropout_probability: float | None
    memory_stack_last_layer_bias_option: LastLayerBiasOptions | None
    memory_stack_apply_output_pipeline_flag: bool | None
    memory_stack_bias_flag: bool | None
    recurrent_gate_stack_independent_flag: bool
    recurrent_gate_stack_hidden_dim: int | None
    recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions | None
    recurrent_gate_stack_num_layers: int | None
    recurrent_gate_stack_activation: ActivationOptions | None
    recurrent_gate_stack_residual_connection_option: ResidualConnectionOptions | None
    recurrent_gate_stack_dropout_probability: float | None
    recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None
    recurrent_gate_stack_apply_output_pipeline_flag: bool | None
    recurrent_gate_stack_bias_flag: bool | None
    recurrent_halting_stack_independent_flag: bool
    recurrent_halting_stack_hidden_dim: int | None
    recurrent_halting_stack_layer_norm_position: LayerNormPositionOptions | None
    recurrent_halting_stack_num_layers: int | None
    recurrent_halting_stack_activation: ActivationOptions | None
    recurrent_halting_stack_residual_connection_option: ResidualConnectionOptions | None
    recurrent_halting_stack_dropout_probability: float | None
    recurrent_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None
    recurrent_halting_stack_apply_output_pipeline_flag: bool | None
    recurrent_halting_stack_bias_flag: bool | None
    token_mixer_stack_gate_flag: bool
    token_mixer_gate_option: LayerGateOptions | None
    token_mixer_gate_activation: ActivationOptions | None
    token_mixer_gate_stack_independent_flag: bool
    token_mixer_gate_stack_hidden_dim: int | None
    token_mixer_gate_stack_layer_norm_position: LayerNormPositionOptions | None
    token_mixer_gate_stack_num_layers: int | None
    token_mixer_gate_stack_activation: ActivationOptions | None
    token_mixer_gate_stack_residual_connection_option: ResidualConnectionOptions | None
    token_mixer_gate_stack_dropout_probability: float | None
    token_mixer_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None
    token_mixer_gate_stack_apply_output_pipeline_flag: bool | None
    token_mixer_gate_stack_bias_flag: bool | None
    token_mixer_stack_halting_flag: bool
    token_mixer_halting_option: type[HaltingConfig]
    token_mixer_halting_threshold: float
    token_mixer_halting_dropout: float
    token_mixer_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    token_mixer_halting_stack_independent_flag: bool
    token_mixer_halting_stack_hidden_dim: int | None
    token_mixer_halting_stack_layer_norm_position: LayerNormPositionOptions | None
    token_mixer_halting_stack_num_layers: int | None
    token_mixer_halting_stack_activation: ActivationOptions | None
    token_mixer_halting_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    token_mixer_halting_stack_dropout_probability: float | None
    token_mixer_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None
    token_mixer_halting_stack_apply_output_pipeline_flag: bool | None
    token_mixer_halting_stack_bias_flag: bool | None
    token_mixer_memory_flag: bool
    token_mixer_memory_option: type[DynamicMemoryConfig]
    token_mixer_memory_position_option: MemoryPositionOptions
    token_mixer_memory_test_time_training_learning_rate: float | None
    token_mixer_memory_test_time_training_num_inner_steps: int | None
    token_mixer_memory_stack_independent_flag: bool
    token_mixer_memory_stack_hidden_dim: int | None
    token_mixer_memory_stack_layer_norm_position: LayerNormPositionOptions | None
    token_mixer_memory_stack_num_layers: int | None
    token_mixer_memory_stack_activation: ActivationOptions | None
    token_mixer_memory_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    token_mixer_memory_stack_dropout_probability: float | None
    token_mixer_memory_stack_last_layer_bias_option: LastLayerBiasOptions | None
    token_mixer_memory_stack_apply_output_pipeline_flag: bool | None
    token_mixer_memory_stack_bias_flag: bool | None
    token_mixer_recurrent_flag: bool
    token_mixer_recurrent_max_steps: int
    token_mixer_recurrent_layer_norm_position: LayerNormPositionOptions
    token_mixer_recurrent_residual_connection_option: ResidualConnectionOptions | None
    token_mixer_recurrent_stack_gate_flag: bool
    token_mixer_recurrent_gate_option: LayerGateOptions | None
    token_mixer_recurrent_gate_activation: ActivationOptions | None
    token_mixer_recurrent_gate_stack_independent_flag: bool
    token_mixer_recurrent_gate_stack_hidden_dim: int | None
    token_mixer_recurrent_gate_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    )
    token_mixer_recurrent_gate_stack_num_layers: int | None
    token_mixer_recurrent_gate_stack_activation: ActivationOptions | None
    token_mixer_recurrent_gate_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    token_mixer_recurrent_gate_stack_dropout_probability: float | None
    token_mixer_recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None
    token_mixer_recurrent_gate_stack_apply_output_pipeline_flag: bool | None
    token_mixer_recurrent_gate_stack_bias_flag: bool | None
    token_mixer_recurrent_stack_halting_flag: bool
    token_mixer_recurrent_halting_option: type[HaltingConfig]
    token_mixer_recurrent_halting_threshold: float
    token_mixer_recurrent_halting_dropout: float
    token_mixer_recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    token_mixer_recurrent_halting_stack_independent_flag: bool
    token_mixer_recurrent_halting_stack_hidden_dim: int | None
    token_mixer_recurrent_halting_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    )
    token_mixer_recurrent_halting_stack_num_layers: int | None
    token_mixer_recurrent_halting_stack_activation: ActivationOptions | None
    token_mixer_recurrent_halting_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    token_mixer_recurrent_halting_stack_dropout_probability: float | None
    token_mixer_recurrent_halting_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    )
    token_mixer_recurrent_halting_stack_apply_output_pipeline_flag: bool | None
    token_mixer_recurrent_halting_stack_bias_flag: bool | None
    channel_mixer_stack_gate_flag: bool
    channel_mixer_gate_option: LayerGateOptions | None
    channel_mixer_gate_activation: ActivationOptions | None
    channel_mixer_gate_stack_independent_flag: bool
    channel_mixer_gate_stack_hidden_dim: int | None
    channel_mixer_gate_stack_layer_norm_position: LayerNormPositionOptions | None
    channel_mixer_gate_stack_num_layers: int | None
    channel_mixer_gate_stack_activation: ActivationOptions | None
    channel_mixer_gate_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    channel_mixer_gate_stack_dropout_probability: float | None
    channel_mixer_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None
    channel_mixer_gate_stack_apply_output_pipeline_flag: bool | None
    channel_mixer_gate_stack_bias_flag: bool | None
    channel_mixer_stack_halting_flag: bool
    channel_mixer_halting_option: type[HaltingConfig]
    channel_mixer_halting_threshold: float
    channel_mixer_halting_dropout: float
    channel_mixer_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    channel_mixer_halting_stack_independent_flag: bool
    channel_mixer_halting_stack_hidden_dim: int | None
    channel_mixer_halting_stack_layer_norm_position: LayerNormPositionOptions | None
    channel_mixer_halting_stack_num_layers: int | None
    channel_mixer_halting_stack_activation: ActivationOptions | None
    channel_mixer_halting_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    channel_mixer_halting_stack_dropout_probability: float | None
    channel_mixer_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None
    channel_mixer_halting_stack_apply_output_pipeline_flag: bool | None
    channel_mixer_halting_stack_bias_flag: bool | None
    channel_mixer_memory_flag: bool
    channel_mixer_memory_option: type[DynamicMemoryConfig]
    channel_mixer_memory_position_option: MemoryPositionOptions
    channel_mixer_memory_test_time_training_learning_rate: float | None
    channel_mixer_memory_test_time_training_num_inner_steps: int | None
    channel_mixer_memory_stack_independent_flag: bool
    channel_mixer_memory_stack_hidden_dim: int | None
    channel_mixer_memory_stack_layer_norm_position: LayerNormPositionOptions | None
    channel_mixer_memory_stack_num_layers: int | None
    channel_mixer_memory_stack_activation: ActivationOptions | None
    channel_mixer_memory_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    channel_mixer_memory_stack_dropout_probability: float | None
    channel_mixer_memory_stack_last_layer_bias_option: LastLayerBiasOptions | None
    channel_mixer_memory_stack_apply_output_pipeline_flag: bool | None
    channel_mixer_memory_stack_bias_flag: bool | None
    channel_mixer_recurrent_flag: bool
    channel_mixer_recurrent_max_steps: int
    channel_mixer_recurrent_layer_norm_position: LayerNormPositionOptions
    channel_mixer_recurrent_residual_connection_option: ResidualConnectionOptions | None
    channel_mixer_recurrent_stack_gate_flag: bool
    channel_mixer_recurrent_gate_option: LayerGateOptions | None
    channel_mixer_recurrent_gate_activation: ActivationOptions | None
    channel_mixer_recurrent_gate_stack_independent_flag: bool
    channel_mixer_recurrent_gate_stack_hidden_dim: int | None
    channel_mixer_recurrent_gate_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    )
    channel_mixer_recurrent_gate_stack_num_layers: int | None
    channel_mixer_recurrent_gate_stack_activation: ActivationOptions | None
    channel_mixer_recurrent_gate_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    channel_mixer_recurrent_gate_stack_dropout_probability: float | None
    channel_mixer_recurrent_gate_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    )
    channel_mixer_recurrent_gate_stack_apply_output_pipeline_flag: bool | None
    channel_mixer_recurrent_gate_stack_bias_flag: bool | None
    channel_mixer_recurrent_stack_halting_flag: bool
    channel_mixer_recurrent_halting_option: type[HaltingConfig]
    channel_mixer_recurrent_halting_threshold: float
    channel_mixer_recurrent_halting_dropout: float
    channel_mixer_recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    channel_mixer_recurrent_halting_stack_independent_flag: bool
    channel_mixer_recurrent_halting_stack_hidden_dim: int | None
    channel_mixer_recurrent_halting_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    )
    channel_mixer_recurrent_halting_stack_num_layers: int | None
    channel_mixer_recurrent_halting_stack_activation: ActivationOptions | None
    channel_mixer_recurrent_halting_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    channel_mixer_recurrent_halting_stack_dropout_probability: float | None
    channel_mixer_recurrent_halting_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    )
    channel_mixer_recurrent_halting_stack_apply_output_pipeline_flag: bool | None
    channel_mixer_recurrent_halting_stack_bias_flag: bool | None
    top_k: int
    num_experts: int
    capacity_factor: float
    dropped_token_behavior: DroppedTokenOptions
    weighted_parameters_flag: bool
    compute_expert_mixture_flag: bool
    weighting_position_option: ExpertWeightingPositionOptions
    routing_initialization_mode: RoutingInitializationMode
    sampler_threshold: float
    sampler_filter_above_threshold: bool
    sampler_num_topk_samples: int
    sampler_normalize_probabilities_flag: bool
    sampler_noisy_topk_flag: bool
    sampler_coefficient_of_variation_loss_weight: float
    sampler_switch_loss_weight: float
    sampler_zero_centred_loss_weight: float
    sampler_mutual_information_loss_weight: float
    router_noisy_topk_flag: bool
    router_stack_hidden_dim: int
    router_stack_num_layers: int
    router_stack_activation: ActivationOptions
    router_stack_residual_connection_option: ResidualConnectionOptions | None
    router_stack_dropout_probability: float
    router_stack_layer_norm_position: LayerNormPositionOptions
    router_stack_last_layer_bias_option: LastLayerBiasOptions
    router_stack_apply_output_pipeline_flag: bool
    router_bias_flag: bool
    expert_stack_hidden_dim: int
    expert_stack_num_layers: int
    expert_stack_activation: ActivationOptions
    expert_stack_dropout_probability: float
    expert_stack_layer_norm_position: LayerNormPositionOptions
    expert_stack_residual_connection_option: ResidualConnectionOptions | None
    expert_stack_last_layer_bias_option: LastLayerBiasOptions
    expert_stack_apply_output_pipeline_flag: bool
    expert_bias_flag: bool
    expert_stack_gate_flag: bool
    expert_gate_option: LayerGateOptions | None
    expert_gate_activation: ActivationOptions | None
    expert_gate_stack_independent_flag: bool
    expert_gate_stack_hidden_dim: int | None
    expert_gate_stack_layer_norm_position: LayerNormPositionOptions | None
    expert_gate_stack_num_layers: int | None
    expert_gate_stack_activation: ActivationOptions | None
    expert_gate_stack_residual_connection_option: ResidualConnectionOptions | None
    expert_gate_stack_dropout_probability: float | None
    expert_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None
    expert_gate_stack_apply_output_pipeline_flag: bool | None
    expert_gate_stack_bias_flag: bool | None
    expert_stack_halting_flag: bool
    expert_halting_option: type[HaltingConfig]
    expert_halting_threshold: float
    expert_halting_dropout: float
    expert_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    expert_halting_stack_independent_flag: bool
    expert_halting_stack_hidden_dim: int | None
    expert_halting_stack_layer_norm_position: LayerNormPositionOptions | None
    expert_halting_stack_num_layers: int | None
    expert_halting_stack_activation: ActivationOptions | None
    expert_halting_stack_residual_connection_option: ResidualConnectionOptions | None
    expert_halting_stack_dropout_probability: float | None
    expert_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None
    expert_halting_stack_apply_output_pipeline_flag: bool | None
    expert_halting_stack_bias_flag: bool | None
    expert_memory_flag: bool
    expert_memory_option: type[DynamicMemoryConfig]
    expert_memory_position_option: MemoryPositionOptions
    expert_memory_test_time_training_learning_rate: float | None
    expert_memory_test_time_training_num_inner_steps: int | None
    expert_memory_stack_independent_flag: bool
    expert_memory_stack_hidden_dim: int | None
    expert_memory_stack_layer_norm_position: LayerNormPositionOptions | None
    expert_memory_stack_num_layers: int | None
    expert_memory_stack_activation: ActivationOptions | None
    expert_memory_stack_residual_connection_option: ResidualConnectionOptions | None
    expert_memory_stack_dropout_probability: float | None
    expert_memory_stack_last_layer_bias_option: LastLayerBiasOptions | None
    expert_memory_stack_apply_output_pipeline_flag: bool | None
    expert_memory_stack_bias_flag: bool | None
    expert_recurrent_flag: bool
    expert_recurrent_max_steps: int
    expert_recurrent_layer_norm_position: LayerNormPositionOptions
    expert_recurrent_residual_connection_option: ResidualConnectionOptions | None
    expert_recurrent_stack_gate_flag: bool
    expert_recurrent_gate_option: LayerGateOptions | None
    expert_recurrent_gate_activation: ActivationOptions | None
    expert_recurrent_gate_stack_independent_flag: bool
    expert_recurrent_gate_stack_hidden_dim: int | None
    expert_recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions | None
    expert_recurrent_gate_stack_num_layers: int | None
    expert_recurrent_gate_stack_activation: ActivationOptions | None
    expert_recurrent_gate_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    expert_recurrent_gate_stack_dropout_probability: float | None
    expert_recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None
    expert_recurrent_gate_stack_apply_output_pipeline_flag: bool | None
    expert_recurrent_gate_stack_bias_flag: bool | None
    expert_recurrent_stack_halting_flag: bool
    expert_recurrent_halting_option: type[HaltingConfig]
    expert_recurrent_halting_threshold: float
    expert_recurrent_halting_dropout: float
    expert_recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    expert_recurrent_halting_stack_independent_flag: bool
    expert_recurrent_halting_stack_hidden_dim: int | None
    expert_recurrent_halting_stack_layer_norm_position: LayerNormPositionOptions | None
    expert_recurrent_halting_stack_num_layers: int | None
    expert_recurrent_halting_stack_activation: ActivationOptions | None
    expert_recurrent_halting_stack_residual_connection_option: (
        ResidualConnectionOptions | None
    )
    expert_recurrent_halting_stack_dropout_probability: float | None
    expert_recurrent_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None
    expert_recurrent_halting_stack_apply_output_pipeline_flag: bool | None
    expert_recurrent_halting_stack_bias_flag: bool | None


__all__ = ["RuntimeOptions"]

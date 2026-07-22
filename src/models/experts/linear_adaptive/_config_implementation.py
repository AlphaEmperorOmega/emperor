from dataclasses import replace
from typing import TYPE_CHECKING

import models.experts.linear_adaptive.config as config
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
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting import HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConnectionOptions,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.experts.linear_adaptive._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.experts.linear_adaptive._projection_config_factory import (
    AdaptiveBoundaryModelOptions,
    BoundaryModelConfigDependencies,
    BoundaryModelConfigFactory,
)
from models.experts.linear_adaptive.experiment_config import ExperimentConfig
from models.experts.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    resolve_experts_controller_stack_options,
    resolve_experts_submodule_stack_options,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class _RuntimeDefaultsResolver:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        stack_bias_flag: bool = config.STACK_BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_connection_option: ResidualConnectionOptions = config.STACK_RESIDUAL_CONNECTION_OPTION,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        stack_last_layer_bias_option: LastLayerBiasOptions = config.STACK_LAST_LAYER_BIAS_OPTION,
        stack_apply_output_pipeline_flag: bool = config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        submodule_stack_hidden_dim: int = config.SUBMODULE_STACK_HIDDEN_DIM,
        submodule_stack_num_layers: int = config.SUBMODULE_STACK_NUM_LAYERS,
        submodule_stack_activation: ActivationOptions = config.SUBMODULE_STACK_ACTIVATION,
        submodule_stack_residual_connection_option: ResidualConnectionOptions = config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
        submodule_stack_dropout_probability: float = config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
        submodule_stack_layer_norm_position: LayerNormPositionOptions = config.SUBMODULE_STACK_LAYER_NORM_POSITION,
        submodule_stack_last_layer_bias_option: LastLayerBiasOptions = config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
        submodule_stack_apply_output_pipeline_flag: bool = config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        submodule_stack_bias_flag: bool = config.SUBMODULE_STACK_BIAS_FLAG,
        top_k: int = config.TOP_K,
        num_experts: int = config.NUM_EXPERTS,
        capacity_factor: float = config.CAPACITY_FACTOR,
        dropped_token_behavior: DroppedTokenOptions = config.DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag: bool = config.COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag: bool = config.WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option: ExpertWeightingPositionOptions = config.WEIGHTING_POSITION_OPTION,
        routing_initialization_mode: RoutingInitializationMode = config.ROUTING_INITIALIZATION_MODE,
        expert_stack_hidden_dim: int | None = None,
        expert_stack_num_layers: int | None = None,
        expert_stack_activation: ActivationOptions | None = None,
        expert_stack_residual_connection_option: ResidualConnectionOptions
        | None = None,
        expert_stack_dropout_probability: float | None = None,
        expert_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.EXPERT_STACK_LAYER_NORM_POSITION,
        expert_stack_last_layer_bias_option: LastLayerBiasOptions | None = None,
        expert_stack_apply_output_pipeline_flag: bool
        | None = config.EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        expert_bias_flag: bool | None = None,
        expert_stack_gate_flag: bool = config.EXPERT_STACK_GATE_FLAG,
        expert_gate_option: LayerGateOptions | None = config.EXPERT_GATE_OPTION,
        expert_gate_activation: ActivationOptions
        | None = config.EXPERT_GATE_ACTIVATION,
        expert_gate_stack_independent_flag: bool = config.EXPERT_GATE_STACK_INDEPENDENT_FLAG,
        expert_gate_stack_hidden_dim: int | None = config.EXPERT_GATE_STACK_HIDDEN_DIM,
        expert_gate_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.EXPERT_GATE_STACK_LAYER_NORM_POSITION,
        expert_gate_stack_num_layers: int | None = config.EXPERT_GATE_STACK_NUM_LAYERS,
        expert_gate_stack_activation: ActivationOptions
        | None = config.EXPERT_GATE_STACK_ACTIVATION,
        expert_gate_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        expert_gate_stack_dropout_probability: float
        | None = config.EXPERT_GATE_STACK_DROPOUT_PROBABILITY,
        expert_gate_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        expert_gate_stack_apply_output_pipeline_flag: bool
        | None = config.EXPERT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        expert_gate_stack_bias_flag: bool | None = config.EXPERT_GATE_STACK_BIAS_FLAG,
        expert_stack_halting_flag: bool = config.EXPERT_STACK_HALTING_FLAG,
        expert_halting_threshold: float = config.EXPERT_HALTING_THRESHOLD,
        expert_halting_dropout: float = config.EXPERT_HALTING_DROPOUT,
        expert_halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.EXPERT_HALTING_HIDDEN_STATE_MODE,
        expert_halting_output_dim: int = config.EXPERT_HALTING_OUTPUT_DIM,
        expert_halting_stack_independent_flag: bool = config.EXPERT_HALTING_STACK_INDEPENDENT_FLAG,
        expert_halting_stack_hidden_dim: int
        | None = config.EXPERT_HALTING_STACK_HIDDEN_DIM,
        expert_halting_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.EXPERT_HALTING_STACK_LAYER_NORM_POSITION,
        expert_halting_stack_num_layers: int
        | None = config.EXPERT_HALTING_STACK_NUM_LAYERS,
        expert_halting_stack_activation: ActivationOptions
        | None = config.EXPERT_HALTING_STACK_ACTIVATION,
        expert_halting_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        expert_halting_stack_dropout_probability: float
        | None = config.EXPERT_HALTING_STACK_DROPOUT_PROBABILITY,
        expert_halting_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        expert_halting_stack_apply_output_pipeline_flag: bool
        | None = config.EXPERT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        expert_halting_stack_bias_flag: bool
        | None = config.EXPERT_HALTING_STACK_BIAS_FLAG,
        expert_memory_flag: bool = config.EXPERT_MEMORY_FLAG,
        expert_memory_option: type[DynamicMemoryConfig] = config.EXPERT_MEMORY_OPTION,
        expert_memory_position_option: MemoryPositionOptions = config.EXPERT_MEMORY_POSITION_OPTION,
        expert_memory_test_time_training_learning_rate: float
        | None = config.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
        expert_memory_test_time_training_num_inner_steps: int
        | None = config.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
        expert_memory_stack_independent_flag: bool = config.EXPERT_MEMORY_STACK_INDEPENDENT_FLAG,
        expert_memory_stack_hidden_dim: int
        | None = config.EXPERT_MEMORY_STACK_HIDDEN_DIM,
        expert_memory_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.EXPERT_MEMORY_STACK_LAYER_NORM_POSITION,
        expert_memory_stack_num_layers: int
        | None = config.EXPERT_MEMORY_STACK_NUM_LAYERS,
        expert_memory_stack_activation: ActivationOptions
        | None = config.EXPERT_MEMORY_STACK_ACTIVATION,
        expert_memory_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
        expert_memory_stack_dropout_probability: float
        | None = config.EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY,
        expert_memory_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        expert_memory_stack_apply_output_pipeline_flag: bool
        | None = config.EXPERT_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        expert_memory_stack_bias_flag: bool
        | None = config.EXPERT_MEMORY_STACK_BIAS_FLAG,
        expert_recurrent_flag: bool = config.EXPERT_RECURRENT_FLAG,
        expert_recurrent_max_steps: int = config.EXPERT_RECURRENT_MAX_STEPS,
        expert_recurrent_layer_norm_position: LayerNormPositionOptions = config.EXPERT_RECURRENT_LAYER_NORM_POSITION,
        expert_recurrent_stack_gate_flag: bool = config.EXPERT_RECURRENT_STACK_GATE_FLAG,
        expert_recurrent_gate_option: LayerGateOptions
        | None = config.EXPERT_RECURRENT_GATE_OPTION,
        expert_recurrent_gate_activation: ActivationOptions
        | None = config.EXPERT_RECURRENT_GATE_ACTIVATION,
        expert_recurrent_gate_stack_independent_flag: bool = config.EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
        expert_recurrent_gate_stack_hidden_dim: int
        | None = config.EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM,
        expert_recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
        expert_recurrent_gate_stack_num_layers: int
        | None = config.EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS,
        expert_recurrent_gate_stack_activation: ActivationOptions
        | None = config.EXPERT_RECURRENT_GATE_STACK_ACTIVATION,
        expert_recurrent_gate_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        expert_recurrent_gate_stack_dropout_probability: float
        | None = config.EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
        expert_recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        expert_recurrent_gate_stack_apply_output_pipeline_flag: bool
        | None = config.EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        expert_recurrent_gate_stack_bias_flag: bool
        | None = config.EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG,
        expert_recurrent_stack_halting_flag: bool = config.EXPERT_RECURRENT_STACK_HALTING_FLAG,
        expert_recurrent_halting_threshold: float = config.EXPERT_RECURRENT_HALTING_THRESHOLD,
        expert_recurrent_halting_dropout: float = config.EXPERT_RECURRENT_HALTING_DROPOUT,
        expert_recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE,
        expert_recurrent_halting_stack_independent_flag: bool = config.EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        expert_recurrent_halting_stack_hidden_dim: int
        | None = config.EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM,
        expert_recurrent_halting_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        expert_recurrent_halting_stack_num_layers: int
        | None = config.EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS,
        expert_recurrent_halting_stack_activation: ActivationOptions
        | None = config.EXPERT_RECURRENT_HALTING_STACK_ACTIVATION,
        expert_recurrent_halting_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        expert_recurrent_halting_stack_dropout_probability: float
        | None = config.EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        expert_recurrent_halting_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        expert_recurrent_halting_stack_apply_output_pipeline_flag: bool
        | None = config.EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        expert_recurrent_halting_stack_bias_flag: bool
        | None = config.EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG,
        sampler_threshold: float = config.SAMPLER_THRESHOLD,
        sampler_filter_above_threshold: bool = config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        sampler_num_topk_samples: int = config.SAMPLER_NUM_TOPK_SAMPLES,
        sampler_normalize_probabilities_flag: bool = config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        sampler_noisy_topk_flag: bool = config.SAMPLER_NOISY_TOPK_FLAG,
        sampler_coefficient_of_variation_loss_weight: float = config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
        sampler_switch_loss_weight: float = config.SAMPLER_SWITCH_LOSS_WEIGHT,
        sampler_zero_centred_loss_weight: float = config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        sampler_mutual_information_loss_weight: float = config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
        router_noisy_topk_flag: bool = config.ROUTER_NOISY_TOPK_FLAG,
        router_stack_hidden_dim: int = config.ROUTER_STACK_HIDDEN_DIM,
        router_stack_num_layers: int = config.ROUTER_STACK_NUM_LAYERS,
        router_stack_activation: ActivationOptions = config.ROUTER_STACK_ACTIVATION,
        router_stack_residual_connection_option: ResidualConnectionOptions = (
            config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        router_stack_dropout_probability: float = (
            config.ROUTER_STACK_DROPOUT_PROBABILITY
        ),
        router_stack_layer_norm_position: LayerNormPositionOptions = (
            config.ROUTER_STACK_LAYER_NORM_POSITION
        ),
        router_stack_last_layer_bias_option: LastLayerBiasOptions = (
            config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION
        ),
        router_stack_apply_output_pipeline_flag: bool = (
            config.ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        router_bias_flag: bool = config.ROUTER_BIAS_FLAG,
        router_stack_gate_flag: bool = config.ROUTER_STACK_GATE_FLAG,
        router_gate_option: LayerGateOptions | None = config.ROUTER_GATE_OPTION,
        router_gate_activation: ActivationOptions
        | None = config.ROUTER_GATE_ACTIVATION,
        router_gate_stack_independent_flag: bool = config.ROUTER_GATE_STACK_INDEPENDENT_FLAG,
        router_gate_stack_hidden_dim: int | None = config.ROUTER_GATE_STACK_HIDDEN_DIM,
        router_gate_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.ROUTER_GATE_STACK_LAYER_NORM_POSITION,
        router_gate_stack_num_layers: int | None = config.ROUTER_GATE_STACK_NUM_LAYERS,
        router_gate_stack_activation: ActivationOptions
        | None = config.ROUTER_GATE_STACK_ACTIVATION,
        router_gate_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.ROUTER_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        router_gate_stack_dropout_probability: float
        | None = config.ROUTER_GATE_STACK_DROPOUT_PROBABILITY,
        router_gate_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.ROUTER_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        router_gate_stack_apply_output_pipeline_flag: bool
        | None = config.ROUTER_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        router_gate_stack_bias_flag: bool | None = config.ROUTER_GATE_STACK_BIAS_FLAG,
        router_stack_halting_flag: bool = config.ROUTER_STACK_HALTING_FLAG,
        router_halting_threshold: float = config.ROUTER_HALTING_THRESHOLD,
        router_halting_dropout: float = config.ROUTER_HALTING_DROPOUT,
        router_halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.ROUTER_HALTING_HIDDEN_STATE_MODE,
        router_halting_output_dim: int = config.ROUTER_HALTING_OUTPUT_DIM,
        router_halting_stack_independent_flag: bool = config.ROUTER_HALTING_STACK_INDEPENDENT_FLAG,
        router_halting_stack_hidden_dim: int
        | None = config.ROUTER_HALTING_STACK_HIDDEN_DIM,
        router_halting_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.ROUTER_HALTING_STACK_LAYER_NORM_POSITION,
        router_halting_stack_num_layers: int
        | None = config.ROUTER_HALTING_STACK_NUM_LAYERS,
        router_halting_stack_activation: ActivationOptions
        | None = config.ROUTER_HALTING_STACK_ACTIVATION,
        router_halting_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.ROUTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        router_halting_stack_dropout_probability: float
        | None = config.ROUTER_HALTING_STACK_DROPOUT_PROBABILITY,
        router_halting_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.ROUTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        router_halting_stack_apply_output_pipeline_flag: bool
        | None = config.ROUTER_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        router_halting_stack_bias_flag: bool
        | None = config.ROUTER_HALTING_STACK_BIAS_FLAG,
        router_memory_flag: bool = config.ROUTER_MEMORY_FLAG,
        router_memory_option: type[DynamicMemoryConfig] = config.ROUTER_MEMORY_OPTION,
        router_memory_position_option: MemoryPositionOptions = config.ROUTER_MEMORY_POSITION_OPTION,
        router_memory_test_time_training_learning_rate: float
        | None = config.ROUTER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
        router_memory_test_time_training_num_inner_steps: int
        | None = config.ROUTER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
        router_memory_stack_independent_flag: bool = config.ROUTER_MEMORY_STACK_INDEPENDENT_FLAG,
        router_memory_stack_hidden_dim: int
        | None = config.ROUTER_MEMORY_STACK_HIDDEN_DIM,
        router_memory_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.ROUTER_MEMORY_STACK_LAYER_NORM_POSITION,
        router_memory_stack_num_layers: int
        | None = config.ROUTER_MEMORY_STACK_NUM_LAYERS,
        router_memory_stack_activation: ActivationOptions
        | None = config.ROUTER_MEMORY_STACK_ACTIVATION,
        router_memory_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.ROUTER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
        router_memory_stack_dropout_probability: float
        | None = config.ROUTER_MEMORY_STACK_DROPOUT_PROBABILITY,
        router_memory_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.ROUTER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        router_memory_stack_apply_output_pipeline_flag: bool
        | None = config.ROUTER_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        router_memory_stack_bias_flag: bool
        | None = config.ROUTER_MEMORY_STACK_BIAS_FLAG,
        router_recurrent_flag: bool = config.ROUTER_RECURRENT_FLAG,
        router_recurrent_max_steps: int = config.ROUTER_RECURRENT_MAX_STEPS,
        router_recurrent_layer_norm_position: LayerNormPositionOptions = config.ROUTER_RECURRENT_LAYER_NORM_POSITION,
        router_recurrent_stack_gate_flag: bool = config.ROUTER_RECURRENT_STACK_GATE_FLAG,
        router_recurrent_gate_option: LayerGateOptions
        | None = config.ROUTER_RECURRENT_GATE_OPTION,
        router_recurrent_gate_activation: ActivationOptions
        | None = config.ROUTER_RECURRENT_GATE_ACTIVATION,
        router_recurrent_gate_stack_independent_flag: bool = config.ROUTER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
        router_recurrent_gate_stack_hidden_dim: int
        | None = config.ROUTER_RECURRENT_GATE_STACK_HIDDEN_DIM,
        router_recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.ROUTER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
        router_recurrent_gate_stack_num_layers: int
        | None = config.ROUTER_RECURRENT_GATE_STACK_NUM_LAYERS,
        router_recurrent_gate_stack_activation: ActivationOptions
        | None = config.ROUTER_RECURRENT_GATE_STACK_ACTIVATION,
        router_recurrent_gate_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        router_recurrent_gate_stack_dropout_probability: float
        | None = config.ROUTER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
        router_recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.ROUTER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        router_recurrent_gate_stack_apply_output_pipeline_flag: bool
        | None = config.ROUTER_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        router_recurrent_gate_stack_bias_flag: bool
        | None = config.ROUTER_RECURRENT_GATE_STACK_BIAS_FLAG,
        router_recurrent_stack_halting_flag: bool = config.ROUTER_RECURRENT_STACK_HALTING_FLAG,
        router_recurrent_halting_threshold: float = config.ROUTER_RECURRENT_HALTING_THRESHOLD,
        router_recurrent_halting_dropout: float = config.ROUTER_RECURRENT_HALTING_DROPOUT,
        router_recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE,
        router_recurrent_halting_stack_independent_flag: bool = config.ROUTER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        router_recurrent_halting_stack_hidden_dim: int
        | None = config.ROUTER_RECURRENT_HALTING_STACK_HIDDEN_DIM,
        router_recurrent_halting_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.ROUTER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        router_recurrent_halting_stack_num_layers: int
        | None = config.ROUTER_RECURRENT_HALTING_STACK_NUM_LAYERS,
        router_recurrent_halting_stack_activation: ActivationOptions
        | None = config.ROUTER_RECURRENT_HALTING_STACK_ACTIVATION,
        router_recurrent_halting_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        router_recurrent_halting_stack_dropout_probability: float
        | None = config.ROUTER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        router_recurrent_halting_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.ROUTER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        router_recurrent_halting_stack_apply_output_pipeline_flag: bool
        | None = config.ROUTER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        router_recurrent_halting_stack_bias_flag: bool
        | None = config.ROUTER_RECURRENT_HALTING_STACK_BIAS_FLAG,
        stack_gate_flag: bool = config.STACK_GATE_FLAG,
        gate_option: LayerGateOptions | None = config.GATE_OPTION,
        gate_activation: ActivationOptions | None = config.GATE_ACTIVATION,
        gate_stack_independent_flag: bool = config.GATE_STACK_INDEPENDENT_FLAG,
        gate_stack_hidden_dim: int | None = config.GATE_STACK_HIDDEN_DIM,
        gate_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.GATE_STACK_LAYER_NORM_POSITION,
        gate_stack_num_layers: int | None = config.GATE_STACK_NUM_LAYERS,
        gate_stack_activation: ActivationOptions | None = config.GATE_STACK_ACTIVATION,
        gate_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        gate_stack_dropout_probability: float
        | None = config.GATE_STACK_DROPOUT_PROBABILITY,
        gate_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        gate_stack_apply_output_pipeline_flag: bool
        | None = config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        gate_stack_bias_flag: bool | None = config.GATE_STACK_BIAS_FLAG,
        stack_halting_flag: bool = config.STACK_HALTING_FLAG,
        halting_threshold: float = config.HALTING_THRESHOLD,
        halting_dropout: float = config.HALTING_DROPOUT,
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.HALTING_HIDDEN_STATE_MODE,
        halting_output_dim: int = config.HALTING_OUTPUT_DIM,
        halting_stack_independent_flag: bool = config.HALTING_STACK_INDEPENDENT_FLAG,
        halting_stack_hidden_dim: int | None = config.HALTING_STACK_HIDDEN_DIM,
        halting_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.HALTING_STACK_LAYER_NORM_POSITION,
        halting_stack_num_layers: int | None = config.HALTING_STACK_NUM_LAYERS,
        halting_stack_activation: ActivationOptions
        | None = config.HALTING_STACK_ACTIVATION,
        halting_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        halting_stack_dropout_probability: float
        | None = config.HALTING_STACK_DROPOUT_PROBABILITY,
        halting_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        halting_stack_apply_output_pipeline_flag: bool
        | None = config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        halting_stack_bias_flag: bool | None = config.HALTING_STACK_BIAS_FLAG,
        memory_flag: bool = config.MEMORY_FLAG,
        memory_option: type[DynamicMemoryConfig] = config.MEMORY_OPTION,
        memory_position_option: MemoryPositionOptions = config.MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate: float
        | None = config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
        memory_test_time_training_num_inner_steps: int
        | None = config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
        memory_stack_independent_flag: bool = config.MEMORY_STACK_INDEPENDENT_FLAG,
        memory_stack_hidden_dim: int | None = config.MEMORY_STACK_HIDDEN_DIM,
        memory_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.MEMORY_STACK_LAYER_NORM_POSITION,
        memory_stack_num_layers: int | None = config.MEMORY_STACK_NUM_LAYERS,
        memory_stack_activation: ActivationOptions
        | None = config.MEMORY_STACK_ACTIVATION,
        memory_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
        memory_stack_dropout_probability: float
        | None = config.MEMORY_STACK_DROPOUT_PROBABILITY,
        memory_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        memory_stack_apply_output_pipeline_flag: bool
        | None = config.MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        memory_stack_bias_flag: bool | None = config.MEMORY_STACK_BIAS_FLAG,
        weight_option_flag: bool | None = None,
        generator_depth: DynamicDepthOptions = config.GENERATOR_DEPTH,
        diagonal_option: type[DynamicDiagonalConfig] | None = config.DIAGONAL_OPTION,
        diagonal_option_flag: bool | None = None,
        bias_option: type[DynamicBiasConfig] | None = config.BIAS_OPTION,
        bias_option_flag: bool | None = None,
        weight_option: type[DynamicWeightConfig] | None = config.WEIGHT_OPTION,
        weight_generator_stack_independent_flag: bool = config.WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
        weight_generator_stack_hidden_dim: int
        | None = config.WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
        weight_generator_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
        weight_generator_stack_num_layers: int
        | None = config.WEIGHT_GENERATOR_STACK_NUM_LAYERS,
        weight_generator_stack_activation: ActivationOptions
        | None = config.WEIGHT_GENERATOR_STACK_ACTIVATION,
        weight_generator_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        weight_generator_stack_dropout_probability: float
        | None = config.WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
        weight_generator_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        weight_generator_stack_apply_output_pipeline_flag: bool
        | None = config.WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        weight_generator_stack_bias_flag: bool
        | None = config.WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        weight_normalization_option: WeightNormalizationOptions = config.WEIGHT_NORMALIZATION_OPTION,
        weight_normalization_position_option: WeightNormalizationPositionOptions = config.WEIGHT_NORMALIZATION_POSITION_OPTION,
        weight_decay_schedule: WeightDecayScheduleOptions = config.WEIGHT_DECAY_SCHEDULE,
        weight_decay_rate: float = config.WEIGHT_DECAY_RATE,
        weight_decay_warmup_batches: int = config.WEIGHT_DECAY_WARMUP_BATCHES,
        weight_bank_expansion_factor: BankExpansionFactorOptions = config.WEIGHT_BANK_EXPANSION_FACTOR,
        bias_decay_schedule: WeightDecayScheduleOptions = config.BIAS_DECAY_SCHEDULE,
        bias_decay_rate: float = config.BIAS_DECAY_RATE,
        bias_decay_warmup_batches: int = config.BIAS_DECAY_WARMUP_BATCHES,
        bias_bank_expansion_factor: BankExpansionFactorOptions = config.BIAS_BANK_EXPANSION_FACTOR,
        bias_generator_stack_independent_flag: bool = config.BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
        bias_generator_stack_hidden_dim: int
        | None = config.BIAS_GENERATOR_STACK_HIDDEN_DIM,
        bias_generator_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
        bias_generator_stack_num_layers: int
        | None = config.BIAS_GENERATOR_STACK_NUM_LAYERS,
        bias_generator_stack_activation: ActivationOptions
        | None = config.BIAS_GENERATOR_STACK_ACTIVATION,
        bias_generator_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        bias_generator_stack_dropout_probability: float
        | None = config.BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
        bias_generator_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        bias_generator_stack_apply_output_pipeline_flag: bool
        | None = config.BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        bias_generator_stack_bias_flag: bool
        | None = config.BIAS_GENERATOR_STACK_BIAS_FLAG,
        diagonal_generator_stack_independent_flag: bool = config.DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
        diagonal_generator_stack_hidden_dim: int
        | None = config.DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
        diagonal_generator_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
        diagonal_generator_stack_num_layers: int
        | None = config.DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
        diagonal_generator_stack_activation: ActivationOptions
        | None = config.DIAGONAL_GENERATOR_STACK_ACTIVATION,
        diagonal_generator_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        diagonal_generator_stack_dropout_probability: float
        | None = config.DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
        diagonal_generator_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        diagonal_generator_stack_apply_output_pipeline_flag: bool
        | None = config.DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        diagonal_generator_stack_bias_flag: bool
        | None = config.DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        row_mask_option: type[AxisMaskConfig] | None = config.ROW_MASK_OPTION,
        mask_option_flag: bool | None = None,
        mask_dimension_option: MaskDimensionOptions = config.MASK_DIMENSION_OPTION,
        mask_threshold: float = config.MASK_THRESHOLD,
        mask_surrogate_scale: float = config.MASK_SURROGATE_SCALE,
        mask_floor: float = config.MASK_FLOOR,
        mask_transition_width: float = config.MASK_TRANSITION_WIDTH,
        mask_generator_stack_independent_flag: bool = config.MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
        mask_generator_stack_hidden_dim: int
        | None = config.MASK_GENERATOR_STACK_HIDDEN_DIM,
        mask_generator_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
        mask_generator_stack_num_layers: int
        | None = config.MASK_GENERATOR_STACK_NUM_LAYERS,
        mask_generator_stack_activation: ActivationOptions
        | None = config.MASK_GENERATOR_STACK_ACTIVATION,
        mask_generator_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        mask_generator_stack_dropout_probability: float
        | None = config.MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
        mask_generator_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        mask_generator_stack_apply_output_pipeline_flag: bool
        | None = config.MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        mask_generator_stack_bias_flag: bool
        | None = config.MASK_GENERATOR_STACK_BIAS_FLAG,
        adaptive_generator_stack_num_layers: int = config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
        adaptive_generator_stack_hidden_dim: int = config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
        adaptive_generator_stack_activation: ActivationOptions = config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
        adaptive_generator_stack_residual_connection_option: ResidualConnectionOptions = config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        adaptive_generator_stack_dropout_probability: float = config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
        adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION,
        adaptive_generator_stack_last_layer_bias_option: LastLayerBiasOptions = config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        adaptive_generator_stack_apply_output_pipeline_flag: bool = config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        adaptive_generator_stack_bias_flag: bool = config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG,
        input_layer_weight_option: type[DynamicWeightConfig]
        | None = config.INPUT_LAYER_WEIGHT_OPTION,
        input_layer_generator_depth: DynamicDepthOptions = config.INPUT_LAYER_GENERATOR_DEPTH,
        input_layer_weight_decay_schedule: WeightDecayScheduleOptions = config.INPUT_LAYER_WEIGHT_DECAY_SCHEDULE,
        input_layer_weight_decay_rate: float = config.INPUT_LAYER_WEIGHT_DECAY_RATE,
        input_layer_weight_decay_warmup_batches: int = config.INPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES,
        input_layer_weight_normalization_option: WeightNormalizationOptions = config.INPUT_LAYER_WEIGHT_NORMALIZATION_OPTION,
        input_layer_weight_normalization_position_option: WeightNormalizationPositionOptions = config.INPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION,
        input_layer_weight_bank_expansion_factor: BankExpansionFactorOptions = config.INPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR,
        input_layer_bias_option: type[DynamicBiasConfig]
        | None = config.INPUT_LAYER_BIAS_OPTION,
        input_layer_bias_decay_schedule: WeightDecayScheduleOptions = config.INPUT_LAYER_BIAS_DECAY_SCHEDULE,
        input_layer_bias_decay_rate: float = config.INPUT_LAYER_BIAS_DECAY_RATE,
        input_layer_bias_decay_warmup_batches: int = config.INPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES,
        input_layer_bias_bank_expansion_factor: BankExpansionFactorOptions = config.INPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR,
        input_layer_diagonal_option: type[DynamicDiagonalConfig]
        | None = config.INPUT_LAYER_DIAGONAL_OPTION,
        input_layer_row_mask_option: type[AxisMaskConfig]
        | None = config.INPUT_LAYER_ROW_MASK_OPTION,
        input_layer_mask_dimension_option: MaskDimensionOptions = config.INPUT_LAYER_MASK_DIMENSION_OPTION,
        input_layer_mask_threshold: float = config.INPUT_LAYER_MASK_THRESHOLD,
        input_layer_mask_surrogate_scale: float = config.INPUT_LAYER_MASK_SURROGATE_SCALE,
        input_layer_mask_floor: float = config.INPUT_LAYER_MASK_FLOOR,
        input_layer_mask_transition_width: float = config.INPUT_LAYER_MASK_TRANSITION_WIDTH,
        output_layer_weight_option: type[DynamicWeightConfig]
        | None = config.OUTPUT_LAYER_WEIGHT_OPTION,
        output_layer_generator_depth: DynamicDepthOptions = config.OUTPUT_LAYER_GENERATOR_DEPTH,
        output_layer_weight_decay_schedule: WeightDecayScheduleOptions = config.OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE,
        output_layer_weight_decay_rate: float = config.OUTPUT_LAYER_WEIGHT_DECAY_RATE,
        output_layer_weight_decay_warmup_batches: int = config.OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES,
        output_layer_weight_normalization_option: WeightNormalizationOptions = config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION,
        output_layer_weight_normalization_position_option: WeightNormalizationPositionOptions = config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION,
        output_layer_weight_bank_expansion_factor: BankExpansionFactorOptions = config.OUTPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR,
        output_layer_bias_option: type[DynamicBiasConfig]
        | None = config.OUTPUT_LAYER_BIAS_OPTION,
        output_layer_bias_decay_schedule: WeightDecayScheduleOptions = config.OUTPUT_LAYER_BIAS_DECAY_SCHEDULE,
        output_layer_bias_decay_rate: float = config.OUTPUT_LAYER_BIAS_DECAY_RATE,
        output_layer_bias_decay_warmup_batches: int = config.OUTPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES,
        output_layer_bias_bank_expansion_factor: BankExpansionFactorOptions = config.OUTPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR,
        output_layer_diagonal_option: type[DynamicDiagonalConfig]
        | None = config.OUTPUT_LAYER_DIAGONAL_OPTION,
        output_layer_row_mask_option: type[AxisMaskConfig]
        | None = config.OUTPUT_LAYER_ROW_MASK_OPTION,
        output_layer_mask_dimension_option: MaskDimensionOptions = config.OUTPUT_LAYER_MASK_DIMENSION_OPTION,
        output_layer_mask_threshold: float = config.OUTPUT_LAYER_MASK_THRESHOLD,
        output_layer_mask_surrogate_scale: float = config.OUTPUT_LAYER_MASK_SURROGATE_SCALE,
        output_layer_mask_floor: float = config.OUTPUT_LAYER_MASK_FLOOR,
        output_layer_mask_transition_width: float = config.OUTPUT_LAYER_MASK_TRANSITION_WIDTH,
        router_weight_option_flag: bool | None = None,
        router_weight_option: type[DynamicWeightConfig]
        | None = config.ROUTER_WEIGHT_OPTION,
        router_generator_depth: DynamicDepthOptions = config.ROUTER_GENERATOR_DEPTH,
        router_weight_normalization_option: WeightNormalizationOptions = config.ROUTER_WEIGHT_NORMALIZATION_OPTION,
        router_weight_normalization_position_option: WeightNormalizationPositionOptions = config.ROUTER_WEIGHT_NORMALIZATION_POSITION_OPTION,
        router_weight_decay_schedule: WeightDecayScheduleOptions = config.ROUTER_WEIGHT_DECAY_SCHEDULE,
        router_weight_decay_rate: float = config.ROUTER_WEIGHT_DECAY_RATE,
        router_weight_decay_warmup_batches: int = config.ROUTER_WEIGHT_DECAY_WARMUP_BATCHES,
        router_weight_bank_expansion_factor: BankExpansionFactorOptions = config.ROUTER_WEIGHT_BANK_EXPANSION_FACTOR,
        router_weight_generator_stack_independent_flag: bool = config.ROUTER_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
        router_weight_generator_stack_hidden_dim: int
        | None = config.ROUTER_WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
        router_weight_generator_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.ROUTER_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
        router_weight_generator_stack_num_layers: int
        | None = config.ROUTER_WEIGHT_GENERATOR_STACK_NUM_LAYERS,
        router_weight_generator_stack_activation: ActivationOptions
        | None = config.ROUTER_WEIGHT_GENERATOR_STACK_ACTIVATION,
        router_weight_generator_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        router_weight_generator_stack_dropout_probability: float
        | None = config.ROUTER_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
        router_weight_generator_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.ROUTER_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        router_weight_generator_stack_apply_output_pipeline_flag: bool
        | None = config.ROUTER_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        router_weight_generator_stack_bias_flag: bool
        | None = config.ROUTER_WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        router_bias_option_flag: bool | None = None,
        router_bias_option: type[DynamicBiasConfig] | None = config.ROUTER_BIAS_OPTION,
        router_bias_decay_schedule: WeightDecayScheduleOptions = config.ROUTER_BIAS_DECAY_SCHEDULE,
        router_bias_decay_rate: float = config.ROUTER_BIAS_DECAY_RATE,
        router_bias_decay_warmup_batches: int = config.ROUTER_BIAS_DECAY_WARMUP_BATCHES,
        router_bias_bank_expansion_factor: BankExpansionFactorOptions = config.ROUTER_BIAS_BANK_EXPANSION_FACTOR,
        router_bias_generator_stack_independent_flag: bool = config.ROUTER_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
        router_bias_generator_stack_hidden_dim: int
        | None = config.ROUTER_BIAS_GENERATOR_STACK_HIDDEN_DIM,
        router_bias_generator_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.ROUTER_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
        router_bias_generator_stack_num_layers: int
        | None = config.ROUTER_BIAS_GENERATOR_STACK_NUM_LAYERS,
        router_bias_generator_stack_activation: ActivationOptions
        | None = config.ROUTER_BIAS_GENERATOR_STACK_ACTIVATION,
        router_bias_generator_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        router_bias_generator_stack_dropout_probability: float
        | None = config.ROUTER_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
        router_bias_generator_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.ROUTER_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        router_bias_generator_stack_apply_output_pipeline_flag: bool
        | None = config.ROUTER_BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        router_bias_generator_stack_bias_flag: bool
        | None = config.ROUTER_BIAS_GENERATOR_STACK_BIAS_FLAG,
        router_diagonal_option_flag: bool | None = None,
        router_diagonal_option: type[DynamicDiagonalConfig]
        | None = config.ROUTER_DIAGONAL_OPTION,
        router_diagonal_generator_stack_independent_flag: bool = config.ROUTER_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
        router_diagonal_generator_stack_hidden_dim: int
        | None = config.ROUTER_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
        router_diagonal_generator_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.ROUTER_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
        router_diagonal_generator_stack_num_layers: int
        | None = config.ROUTER_DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
        router_diagonal_generator_stack_activation: ActivationOptions
        | None = config.ROUTER_DIAGONAL_GENERATOR_STACK_ACTIVATION,
        router_diagonal_generator_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        router_diagonal_generator_stack_dropout_probability: float
        | None = config.ROUTER_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
        router_diagonal_generator_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.ROUTER_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        router_diagonal_generator_stack_apply_output_pipeline_flag: bool
        | None = config.ROUTER_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        router_diagonal_generator_stack_bias_flag: bool
        | None = config.ROUTER_DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        router_mask_option_flag: bool | None = None,
        router_row_mask_option: type[AxisMaskConfig]
        | None = config.ROUTER_ROW_MASK_OPTION,
        router_mask_dimension_option: MaskDimensionOptions = config.ROUTER_MASK_DIMENSION_OPTION,
        router_mask_threshold: float = config.ROUTER_MASK_THRESHOLD,
        router_mask_surrogate_scale: float = config.ROUTER_MASK_SURROGATE_SCALE,
        router_mask_floor: float = config.ROUTER_MASK_FLOOR,
        router_mask_transition_width: float = config.ROUTER_MASK_TRANSITION_WIDTH,
        router_mask_generator_stack_independent_flag: bool = config.ROUTER_MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
        router_mask_generator_stack_hidden_dim: int
        | None = config.ROUTER_MASK_GENERATOR_STACK_HIDDEN_DIM,
        router_mask_generator_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.ROUTER_MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
        router_mask_generator_stack_num_layers: int
        | None = config.ROUTER_MASK_GENERATOR_STACK_NUM_LAYERS,
        router_mask_generator_stack_activation: ActivationOptions
        | None = config.ROUTER_MASK_GENERATOR_STACK_ACTIVATION,
        router_mask_generator_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        router_mask_generator_stack_dropout_probability: float
        | None = config.ROUTER_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
        router_mask_generator_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.ROUTER_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        router_mask_generator_stack_apply_output_pipeline_flag: bool
        | None = config.ROUTER_MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        router_mask_generator_stack_bias_flag: bool
        | None = config.ROUTER_MASK_GENERATOR_STACK_BIAS_FLAG,
        recurrent_flag: bool = config.RECURRENT_FLAG,
        recurrent_max_steps: int = config.RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position: LayerNormPositionOptions = config.RECURRENT_LAYER_NORM_POSITION,
        recurrent_stack_gate_flag: bool = config.RECURRENT_STACK_GATE_FLAG,
        recurrent_gate_option: LayerGateOptions | None = config.RECURRENT_GATE_OPTION,
        recurrent_gate_activation: ActivationOptions
        | None = config.RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_independent_flag: bool = config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
        recurrent_gate_stack_hidden_dim: int
        | None = config.RECURRENT_GATE_STACK_HIDDEN_DIM,
        recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
        recurrent_gate_stack_num_layers: int
        | None = config.RECURRENT_GATE_STACK_NUM_LAYERS,
        recurrent_gate_stack_activation: ActivationOptions
        | None = config.RECURRENT_GATE_STACK_ACTIVATION,
        recurrent_gate_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        recurrent_gate_stack_dropout_probability: float
        | None = config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
        recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        recurrent_gate_stack_apply_output_pipeline_flag: bool
        | None = config.RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        recurrent_gate_stack_bias_flag: bool
        | None = config.RECURRENT_GATE_STACK_BIAS_FLAG,
        recurrent_stack_halting_flag: bool = config.RECURRENT_STACK_HALTING_FLAG,
        recurrent_halting_threshold: float = config.RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout: float = config.RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.RECURRENT_HALTING_HIDDEN_STATE_MODE,
        recurrent_halting_stack_independent_flag: bool = config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        recurrent_halting_stack_hidden_dim: int
        | None = config.RECURRENT_HALTING_STACK_HIDDEN_DIM,
        recurrent_halting_stack_layer_norm_position: LayerNormPositionOptions
        | None = config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        recurrent_halting_stack_num_layers: int
        | None = config.RECURRENT_HALTING_STACK_NUM_LAYERS,
        recurrent_halting_stack_activation: ActivationOptions
        | None = config.RECURRENT_HALTING_STACK_ACTIVATION,
        recurrent_halting_stack_residual_connection_option: ResidualConnectionOptions
        | None = config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        recurrent_halting_stack_dropout_probability: float
        | None = config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        recurrent_halting_stack_last_layer_bias_option: LastLayerBiasOptions
        | None = config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        recurrent_halting_stack_apply_output_pipeline_flag: bool
        | None = config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        recurrent_halting_stack_bias_flag: bool
        | None = config.RECURRENT_HALTING_STACK_BIAS_FLAG,
        shared_gate_config: GateConfig | None = None,
        stack_options: ExpertsStackOptions | None = None,
        submodule_stack_options: ExpertsSubmoduleStackOptions | None = None,
        mixture_options: ExpertsMixtureOptions | None = None,
        expert_stack_options: ExpertsSubmoduleStackOptions | None = None,
        sampler_options: ExpertsSamplerOptions | None = None,
        router_options: ExpertsRouterOptions | None = None,
        router_stack_options: ExpertsSubmoduleStackOptions | None = None,
        router_layer_controller_options: ExpertsLayerControllerOptions | None = None,
        router_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        router_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
        layer_controller_options: ExpertsLayerControllerOptions | None = None,
        dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        expert_layer_controller_options: ExpertsLayerControllerOptions | None = None,
        expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        expert_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
        adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None = None,
        hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None,
        hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None,
        hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None,
        hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None,
        input_boundary_options: AdaptiveBoundaryModelOptions | None = None,
        output_boundary_options: AdaptiveBoundaryModelOptions | None = None,
        router_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None,
        router_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None,
        router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None,
        router_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None,
        recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None,
    ) -> None:
        stack_options = stack_options or ExpertsStackOptions(
            hidden_dim=hidden_dim,
            bias_flag=stack_bias_flag,
            layer_norm_position=layer_norm_position,
            num_layers=stack_num_layers,
            activation=stack_activation,
            residual_connection_option=stack_residual_connection_option,
            dropout_probability=stack_dropout_probability,
            last_layer_bias_option=stack_last_layer_bias_option,
            apply_output_pipeline_flag=stack_apply_output_pipeline_flag,
        )
        submodule_stack_options = (
            submodule_stack_options
            or ExpertsSubmoduleStackOptions(
                hidden_dim=submodule_stack_hidden_dim,
                num_layers=submodule_stack_num_layers,
                last_layer_bias_option=submodule_stack_last_layer_bias_option,
                apply_output_pipeline_flag=submodule_stack_apply_output_pipeline_flag,
                activation=submodule_stack_activation,
                layer_norm_position=submodule_stack_layer_norm_position,
                residual_connection_option=(submodule_stack_residual_connection_option),
                dropout_probability=submodule_stack_dropout_probability,
                bias_flag=submodule_stack_bias_flag,
            )
        )
        mixture_options = mixture_options or ExpertsMixtureOptions(
            top_k=top_k,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            dropped_token_behavior=dropped_token_behavior,
            compute_expert_mixture_flag=compute_expert_mixture_flag,
            weighted_parameters_flag=weighted_parameters_flag,
            weighting_position_option=weighting_position_option,
            routing_initialization_mode=routing_initialization_mode,
        )
        expert_stack_options = (
            expert_stack_options
            or resolve_experts_submodule_stack_options(
                submodule_stack_options,
                hidden_dim=expert_stack_hidden_dim,
                num_layers=expert_stack_num_layers,
                last_layer_bias_option=expert_stack_last_layer_bias_option,
                apply_output_pipeline_flag=expert_stack_apply_output_pipeline_flag,
                activation=expert_stack_activation,
                layer_norm_position=expert_stack_layer_norm_position,
                residual_connection_option=expert_stack_residual_connection_option,
                dropout_probability=expert_stack_dropout_probability,
                bias_flag=expert_bias_flag,
            )
        )
        expert_layer_controller_options = (
            expert_layer_controller_options
            or ExpertsLayerControllerOptions(
                stack_gate_flag=expert_stack_gate_flag,
                gate_option=expert_gate_option,
                gate_activation=expert_gate_activation,
                gate_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=expert_gate_stack_independent_flag,
                    hidden_dim=expert_gate_stack_hidden_dim,
                    num_layers=expert_gate_stack_num_layers,
                    last_layer_bias_option=(expert_gate_stack_last_layer_bias_option),
                    apply_output_pipeline_flag=(
                        expert_gate_stack_apply_output_pipeline_flag
                    ),
                    activation=expert_gate_stack_activation,
                    layer_norm_position=expert_gate_stack_layer_norm_position,
                    residual_connection_option=(
                        expert_gate_stack_residual_connection_option
                    ),
                    dropout_probability=expert_gate_stack_dropout_probability,
                    bias_flag=expert_gate_stack_bias_flag,
                ),
                stack_halting_flag=expert_stack_halting_flag,
                halting_threshold=expert_halting_threshold,
                halting_dropout=expert_halting_dropout,
                halting_hidden_state_mode=expert_halting_hidden_state_mode,
                halting_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=expert_halting_stack_independent_flag,
                    hidden_dim=expert_halting_stack_hidden_dim,
                    num_layers=expert_halting_stack_num_layers,
                    last_layer_bias_option=(
                        expert_halting_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        expert_halting_stack_apply_output_pipeline_flag
                    ),
                    activation=expert_halting_stack_activation,
                    layer_norm_position=expert_halting_stack_layer_norm_position,
                    residual_connection_option=(
                        expert_halting_stack_residual_connection_option
                    ),
                    dropout_probability=expert_halting_stack_dropout_probability,
                    bias_flag=expert_halting_stack_bias_flag,
                ),
                halting_output_dim=expert_halting_output_dim,
            )
        )
        expert_dynamic_memory_options = (
            expert_dynamic_memory_options
            or ExpertsDynamicMemoryOptions(
                memory_flag=expert_memory_flag,
                memory_option=expert_memory_option,
                memory_position_option=expert_memory_position_option,
                memory_test_time_training_learning_rate=(
                    expert_memory_test_time_training_learning_rate
                ),
                memory_test_time_training_num_inner_steps=(
                    expert_memory_test_time_training_num_inner_steps
                ),
                memory_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=expert_memory_stack_independent_flag,
                    hidden_dim=expert_memory_stack_hidden_dim,
                    num_layers=expert_memory_stack_num_layers,
                    last_layer_bias_option=(expert_memory_stack_last_layer_bias_option),
                    apply_output_pipeline_flag=(
                        expert_memory_stack_apply_output_pipeline_flag
                    ),
                    activation=expert_memory_stack_activation,
                    layer_norm_position=expert_memory_stack_layer_norm_position,
                    residual_connection_option=(
                        expert_memory_stack_residual_connection_option
                    ),
                    dropout_probability=expert_memory_stack_dropout_probability,
                    bias_flag=expert_memory_stack_bias_flag,
                ),
            )
        )
        expert_recurrent_controller_options = (
            expert_recurrent_controller_options
            or ExpertsRecurrentControllerOptions(
                recurrent_flag=expert_recurrent_flag,
                recurrent_max_steps=expert_recurrent_max_steps,
                recurrent_layer_norm_position=(expert_recurrent_layer_norm_position),
                recurrent_stack_gate_flag=expert_recurrent_stack_gate_flag,
                recurrent_gate_option=expert_recurrent_gate_option,
                recurrent_gate_activation=expert_recurrent_gate_activation,
                recurrent_gate_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=(expert_recurrent_gate_stack_independent_flag),
                    hidden_dim=expert_recurrent_gate_stack_hidden_dim,
                    num_layers=expert_recurrent_gate_stack_num_layers,
                    last_layer_bias_option=(
                        expert_recurrent_gate_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        expert_recurrent_gate_stack_apply_output_pipeline_flag
                    ),
                    activation=expert_recurrent_gate_stack_activation,
                    layer_norm_position=(
                        expert_recurrent_gate_stack_layer_norm_position
                    ),
                    residual_connection_option=(
                        expert_recurrent_gate_stack_residual_connection_option
                    ),
                    dropout_probability=(
                        expert_recurrent_gate_stack_dropout_probability
                    ),
                    bias_flag=expert_recurrent_gate_stack_bias_flag,
                ),
                recurrent_stack_halting_flag=expert_recurrent_stack_halting_flag,
                recurrent_halting_threshold=(expert_recurrent_halting_threshold),
                recurrent_halting_dropout=expert_recurrent_halting_dropout,
                recurrent_halting_hidden_state_mode=(
                    expert_recurrent_halting_hidden_state_mode
                ),
                recurrent_halting_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=(expert_recurrent_halting_stack_independent_flag),
                    hidden_dim=expert_recurrent_halting_stack_hidden_dim,
                    num_layers=expert_recurrent_halting_stack_num_layers,
                    last_layer_bias_option=(
                        expert_recurrent_halting_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        expert_recurrent_halting_stack_apply_output_pipeline_flag
                    ),
                    activation=expert_recurrent_halting_stack_activation,
                    layer_norm_position=(
                        expert_recurrent_halting_stack_layer_norm_position
                    ),
                    residual_connection_option=(
                        expert_recurrent_halting_stack_residual_connection_option
                    ),
                    dropout_probability=(
                        expert_recurrent_halting_stack_dropout_probability
                    ),
                    bias_flag=expert_recurrent_halting_stack_bias_flag,
                ),
            )
        )
        sampler_options = sampler_options or ExpertsSamplerOptions(
            threshold=sampler_threshold,
            filter_above_threshold=sampler_filter_above_threshold,
            num_topk_samples=sampler_num_topk_samples,
            normalize_probabilities_flag=sampler_normalize_probabilities_flag,
            noisy_topk_flag=sampler_noisy_topk_flag,
            coefficient_of_variation_loss_weight=(
                sampler_coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=sampler_switch_loss_weight,
            zero_centred_loss_weight=sampler_zero_centred_loss_weight,
            mutual_information_loss_weight=sampler_mutual_information_loss_weight,
        )
        router_options = router_options or ExpertsRouterOptions(
            noisy_topk_flag=router_noisy_topk_flag,
        )
        router_stack_options = router_stack_options or ExpertsSubmoduleStackOptions(
            hidden_dim=router_stack_hidden_dim,
            num_layers=router_stack_num_layers,
            last_layer_bias_option=router_stack_last_layer_bias_option,
            apply_output_pipeline_flag=router_stack_apply_output_pipeline_flag,
            activation=router_stack_activation,
            layer_norm_position=router_stack_layer_norm_position,
            residual_connection_option=router_stack_residual_connection_option,
            dropout_probability=router_stack_dropout_probability,
            bias_flag=router_bias_flag,
        )
        router_layer_controller_options = (
            router_layer_controller_options
            or ExpertsLayerControllerOptions(
                stack_gate_flag=router_stack_gate_flag,
                gate_option=router_gate_option,
                gate_activation=router_gate_activation,
                gate_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=router_gate_stack_independent_flag,
                    hidden_dim=router_gate_stack_hidden_dim,
                    num_layers=router_gate_stack_num_layers,
                    last_layer_bias_option=(router_gate_stack_last_layer_bias_option),
                    apply_output_pipeline_flag=(
                        router_gate_stack_apply_output_pipeline_flag
                    ),
                    activation=router_gate_stack_activation,
                    layer_norm_position=router_gate_stack_layer_norm_position,
                    residual_connection_option=(
                        router_gate_stack_residual_connection_option
                    ),
                    dropout_probability=router_gate_stack_dropout_probability,
                    bias_flag=router_gate_stack_bias_flag,
                ),
                stack_halting_flag=router_stack_halting_flag,
                halting_threshold=router_halting_threshold,
                halting_dropout=router_halting_dropout,
                halting_hidden_state_mode=router_halting_hidden_state_mode,
                halting_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=router_halting_stack_independent_flag,
                    hidden_dim=router_halting_stack_hidden_dim,
                    num_layers=router_halting_stack_num_layers,
                    last_layer_bias_option=(
                        router_halting_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        router_halting_stack_apply_output_pipeline_flag
                    ),
                    activation=router_halting_stack_activation,
                    layer_norm_position=router_halting_stack_layer_norm_position,
                    residual_connection_option=(
                        router_halting_stack_residual_connection_option
                    ),
                    dropout_probability=router_halting_stack_dropout_probability,
                    bias_flag=router_halting_stack_bias_flag,
                ),
                halting_output_dim=router_halting_output_dim,
            )
        )
        router_dynamic_memory_options = (
            router_dynamic_memory_options
            or ExpertsDynamicMemoryOptions(
                memory_flag=router_memory_flag,
                memory_option=router_memory_option,
                memory_position_option=router_memory_position_option,
                memory_test_time_training_learning_rate=(
                    router_memory_test_time_training_learning_rate
                ),
                memory_test_time_training_num_inner_steps=(
                    router_memory_test_time_training_num_inner_steps
                ),
                memory_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=router_memory_stack_independent_flag,
                    hidden_dim=router_memory_stack_hidden_dim,
                    num_layers=router_memory_stack_num_layers,
                    last_layer_bias_option=(router_memory_stack_last_layer_bias_option),
                    apply_output_pipeline_flag=(
                        router_memory_stack_apply_output_pipeline_flag
                    ),
                    activation=router_memory_stack_activation,
                    layer_norm_position=router_memory_stack_layer_norm_position,
                    residual_connection_option=(
                        router_memory_stack_residual_connection_option
                    ),
                    dropout_probability=router_memory_stack_dropout_probability,
                    bias_flag=router_memory_stack_bias_flag,
                ),
            )
        )
        router_recurrent_controller_options = (
            router_recurrent_controller_options
            or ExpertsRecurrentControllerOptions(
                recurrent_flag=router_recurrent_flag,
                recurrent_max_steps=router_recurrent_max_steps,
                recurrent_layer_norm_position=(router_recurrent_layer_norm_position),
                recurrent_stack_gate_flag=router_recurrent_stack_gate_flag,
                recurrent_gate_option=router_recurrent_gate_option,
                recurrent_gate_activation=router_recurrent_gate_activation,
                recurrent_gate_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=(router_recurrent_gate_stack_independent_flag),
                    hidden_dim=router_recurrent_gate_stack_hidden_dim,
                    num_layers=router_recurrent_gate_stack_num_layers,
                    last_layer_bias_option=(
                        router_recurrent_gate_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        router_recurrent_gate_stack_apply_output_pipeline_flag
                    ),
                    activation=router_recurrent_gate_stack_activation,
                    layer_norm_position=(
                        router_recurrent_gate_stack_layer_norm_position
                    ),
                    residual_connection_option=(
                        router_recurrent_gate_stack_residual_connection_option
                    ),
                    dropout_probability=(
                        router_recurrent_gate_stack_dropout_probability
                    ),
                    bias_flag=router_recurrent_gate_stack_bias_flag,
                ),
                recurrent_stack_halting_flag=router_recurrent_stack_halting_flag,
                recurrent_halting_threshold=(router_recurrent_halting_threshold),
                recurrent_halting_dropout=router_recurrent_halting_dropout,
                recurrent_halting_hidden_state_mode=(
                    router_recurrent_halting_hidden_state_mode
                ),
                recurrent_halting_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=(router_recurrent_halting_stack_independent_flag),
                    hidden_dim=router_recurrent_halting_stack_hidden_dim,
                    num_layers=router_recurrent_halting_stack_num_layers,
                    last_layer_bias_option=(
                        router_recurrent_halting_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        router_recurrent_halting_stack_apply_output_pipeline_flag
                    ),
                    activation=router_recurrent_halting_stack_activation,
                    layer_norm_position=(
                        router_recurrent_halting_stack_layer_norm_position
                    ),
                    residual_connection_option=(
                        router_recurrent_halting_stack_residual_connection_option
                    ),
                    dropout_probability=(
                        router_recurrent_halting_stack_dropout_probability
                    ),
                    bias_flag=router_recurrent_halting_stack_bias_flag,
                ),
            )
        )
        layer_controller_options = (
            layer_controller_options
            or ExpertsLayerControllerOptions(
                stack_gate_flag=stack_gate_flag,
                gate_option=gate_option,
                gate_activation=gate_activation,
                gate_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=gate_stack_independent_flag,
                    hidden_dim=gate_stack_hidden_dim,
                    num_layers=gate_stack_num_layers,
                    last_layer_bias_option=gate_stack_last_layer_bias_option,
                    apply_output_pipeline_flag=gate_stack_apply_output_pipeline_flag,
                    activation=gate_stack_activation,
                    layer_norm_position=gate_stack_layer_norm_position,
                    residual_connection_option=gate_stack_residual_connection_option,
                    dropout_probability=gate_stack_dropout_probability,
                    bias_flag=gate_stack_bias_flag,
                ),
                stack_halting_flag=stack_halting_flag,
                halting_threshold=halting_threshold,
                halting_dropout=halting_dropout,
                halting_hidden_state_mode=halting_hidden_state_mode,
                halting_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=halting_stack_independent_flag,
                    hidden_dim=halting_stack_hidden_dim,
                    num_layers=halting_stack_num_layers,
                    last_layer_bias_option=halting_stack_last_layer_bias_option,
                    apply_output_pipeline_flag=(
                        halting_stack_apply_output_pipeline_flag
                    ),
                    activation=halting_stack_activation,
                    layer_norm_position=halting_stack_layer_norm_position,
                    residual_connection_option=(
                        halting_stack_residual_connection_option
                    ),
                    dropout_probability=halting_stack_dropout_probability,
                    bias_flag=halting_stack_bias_flag,
                ),
                halting_output_dim=halting_output_dim,
                shared_gate_config=shared_gate_config,
            )
        )
        dynamic_memory_options = dynamic_memory_options or ExpertsDynamicMemoryOptions(
            memory_flag=memory_flag,
            memory_option=memory_option,
            memory_position_option=memory_position_option,
            memory_test_time_training_learning_rate=(
                memory_test_time_training_learning_rate
            ),
            memory_test_time_training_num_inner_steps=(
                memory_test_time_training_num_inner_steps
            ),
            memory_stack_source=ExpertsSubmoduleStackSource(
                independent_flag=memory_stack_independent_flag,
                hidden_dim=memory_stack_hidden_dim,
                num_layers=memory_stack_num_layers,
                last_layer_bias_option=memory_stack_last_layer_bias_option,
                apply_output_pipeline_flag=(memory_stack_apply_output_pipeline_flag),
                activation=memory_stack_activation,
                layer_norm_position=memory_stack_layer_norm_position,
                residual_connection_option=(memory_stack_residual_connection_option),
                dropout_probability=memory_stack_dropout_probability,
                bias_flag=memory_stack_bias_flag,
            ),
        )
        adaptive_generator_stack_options = (
            adaptive_generator_stack_options
            or AdaptiveGeneratorStackOptions(
                hidden_dim=adaptive_generator_stack_hidden_dim,
                layer_norm_position=adaptive_generator_stack_layer_norm_position,
                num_layers=adaptive_generator_stack_num_layers,
                activation=adaptive_generator_stack_activation,
                residual_connection_option=(
                    adaptive_generator_stack_residual_connection_option
                ),
                dropout_probability=adaptive_generator_stack_dropout_probability,
                last_layer_bias_option=(
                    adaptive_generator_stack_last_layer_bias_option
                ),
                apply_output_pipeline_flag=(
                    adaptive_generator_stack_apply_output_pipeline_flag
                ),
                bias_flag=adaptive_generator_stack_bias_flag,
            )
        )
        if not hasattr(adaptive_generator_stack_options, "bias_flag"):
            adaptive_generator_stack_options = AdaptiveGeneratorStackOptions(
                hidden_dim=adaptive_generator_stack_options.hidden_dim,
                layer_norm_position=(
                    adaptive_generator_stack_options.layer_norm_position
                ),
                num_layers=adaptive_generator_stack_options.num_layers,
                activation=adaptive_generator_stack_options.activation,
                residual_connection_option=(
                    adaptive_generator_stack_options.residual_connection_option
                ),
                dropout_probability=(
                    adaptive_generator_stack_options.dropout_probability
                ),
                last_layer_bias_option=(
                    adaptive_generator_stack_options.last_layer_bias_option
                ),
                apply_output_pipeline_flag=(
                    adaptive_generator_stack_options.apply_output_pipeline_flag
                ),
                bias_flag=adaptive_generator_stack_bias_flag,
            )
        hidden_adaptive_weight_options = (
            hidden_adaptive_weight_options
            or HiddenAdaptiveWeightOptions(
                generator_depth=generator_depth,
                option_flag=self.__adaptive_option_flag(
                    weight_option_flag,
                    weight_option,
                    config.WEIGHT_OPTION_FLAG,
                ),
                option=weight_option,
                normalization_option=weight_normalization_option,
                normalization_position_option=(weight_normalization_position_option),
                decay_schedule=weight_decay_schedule,
                decay_rate=weight_decay_rate,
                decay_warmup_batches=weight_decay_warmup_batches,
                bank_expansion_factor=weight_bank_expansion_factor,
                generator_stack_source=AdaptiveGeneratorStackSource(
                    independent_flag=weight_generator_stack_independent_flag,
                    hidden_dim=weight_generator_stack_hidden_dim,
                    layer_norm_position=(weight_generator_stack_layer_norm_position),
                    num_layers=weight_generator_stack_num_layers,
                    activation=weight_generator_stack_activation,
                    residual_connection_option=(
                        weight_generator_stack_residual_connection_option
                    ),
                    dropout_probability=(weight_generator_stack_dropout_probability),
                    last_layer_bias_option=(
                        weight_generator_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        weight_generator_stack_apply_output_pipeline_flag
                    ),
                    bias_flag=weight_generator_stack_bias_flag,
                ),
            )
        )
        hidden_adaptive_bias_options = (
            hidden_adaptive_bias_options
            or HiddenAdaptiveBiasOptions(
                option_flag=self.__adaptive_option_flag(
                    bias_option_flag,
                    bias_option,
                    config.BIAS_OPTION_FLAG,
                ),
                option=bias_option,
                decay_schedule=bias_decay_schedule,
                decay_rate=bias_decay_rate,
                decay_warmup_batches=bias_decay_warmup_batches,
                bank_expansion_factor=bias_bank_expansion_factor,
                generator_stack_source=AdaptiveGeneratorStackSource(
                    independent_flag=bias_generator_stack_independent_flag,
                    hidden_dim=bias_generator_stack_hidden_dim,
                    layer_norm_position=bias_generator_stack_layer_norm_position,
                    num_layers=bias_generator_stack_num_layers,
                    activation=bias_generator_stack_activation,
                    residual_connection_option=(
                        bias_generator_stack_residual_connection_option
                    ),
                    dropout_probability=bias_generator_stack_dropout_probability,
                    last_layer_bias_option=(
                        bias_generator_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        bias_generator_stack_apply_output_pipeline_flag
                    ),
                    bias_flag=bias_generator_stack_bias_flag,
                ),
            )
        )
        hidden_adaptive_diagonal_options = (
            hidden_adaptive_diagonal_options
            or HiddenAdaptiveDiagonalOptions(
                option_flag=self.__adaptive_option_flag(
                    diagonal_option_flag,
                    diagonal_option,
                    config.DIAGONAL_OPTION_FLAG,
                ),
                option=diagonal_option,
                generator_stack_source=AdaptiveGeneratorStackSource(
                    independent_flag=diagonal_generator_stack_independent_flag,
                    hidden_dim=diagonal_generator_stack_hidden_dim,
                    layer_norm_position=(diagonal_generator_stack_layer_norm_position),
                    num_layers=diagonal_generator_stack_num_layers,
                    activation=diagonal_generator_stack_activation,
                    residual_connection_option=(
                        diagonal_generator_stack_residual_connection_option
                    ),
                    dropout_probability=(diagonal_generator_stack_dropout_probability),
                    last_layer_bias_option=(
                        diagonal_generator_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        diagonal_generator_stack_apply_output_pipeline_flag
                    ),
                    bias_flag=diagonal_generator_stack_bias_flag,
                ),
            )
        )
        hidden_adaptive_mask_options = (
            hidden_adaptive_mask_options
            or HiddenAdaptiveMaskOptions(
                option_flag=self.__adaptive_option_flag(
                    mask_option_flag,
                    row_mask_option,
                    config.MASK_OPTION_FLAG,
                ),
                row_mask_option=row_mask_option,
                mask_dimension_option=mask_dimension_option,
                mask_threshold=mask_threshold,
                mask_surrogate_scale=mask_surrogate_scale,
                mask_floor=mask_floor,
                mask_transition_width=mask_transition_width,
                generator_stack_source=AdaptiveGeneratorStackSource(
                    independent_flag=mask_generator_stack_independent_flag,
                    hidden_dim=mask_generator_stack_hidden_dim,
                    layer_norm_position=mask_generator_stack_layer_norm_position,
                    num_layers=mask_generator_stack_num_layers,
                    activation=mask_generator_stack_activation,
                    residual_connection_option=(
                        mask_generator_stack_residual_connection_option
                    ),
                    dropout_probability=mask_generator_stack_dropout_probability,
                    last_layer_bias_option=(
                        mask_generator_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        mask_generator_stack_apply_output_pipeline_flag
                    ),
                    bias_flag=mask_generator_stack_bias_flag,
                ),
            )
        )
        input_boundary_options = input_boundary_options or AdaptiveBoundaryModelOptions(
            weight_option=input_layer_weight_option,
            generator_depth=input_layer_generator_depth,
            weight_decay_schedule=input_layer_weight_decay_schedule,
            weight_decay_rate=input_layer_weight_decay_rate,
            weight_decay_warmup_batches=input_layer_weight_decay_warmup_batches,
            weight_normalization_option=input_layer_weight_normalization_option,
            weight_normalization_position_option=(
                input_layer_weight_normalization_position_option
            ),
            weight_bank_expansion_factor=input_layer_weight_bank_expansion_factor,
            bias_option=input_layer_bias_option,
            bias_decay_schedule=input_layer_bias_decay_schedule,
            bias_decay_rate=input_layer_bias_decay_rate,
            bias_decay_warmup_batches=input_layer_bias_decay_warmup_batches,
            bias_bank_expansion_factor=input_layer_bias_bank_expansion_factor,
            diagonal_option=input_layer_diagonal_option,
            row_mask_option=input_layer_row_mask_option,
            mask_dimension_option=input_layer_mask_dimension_option,
            mask_threshold=input_layer_mask_threshold,
            mask_surrogate_scale=input_layer_mask_surrogate_scale,
            mask_floor=input_layer_mask_floor,
            mask_transition_width=input_layer_mask_transition_width,
        )
        output_boundary_options = (
            output_boundary_options
            or AdaptiveBoundaryModelOptions(
                weight_option=output_layer_weight_option,
                generator_depth=output_layer_generator_depth,
                weight_decay_schedule=output_layer_weight_decay_schedule,
                weight_decay_rate=output_layer_weight_decay_rate,
                weight_decay_warmup_batches=output_layer_weight_decay_warmup_batches,
                weight_normalization_option=output_layer_weight_normalization_option,
                weight_normalization_position_option=(
                    output_layer_weight_normalization_position_option
                ),
                weight_bank_expansion_factor=output_layer_weight_bank_expansion_factor,
                bias_option=output_layer_bias_option,
                bias_decay_schedule=output_layer_bias_decay_schedule,
                bias_decay_rate=output_layer_bias_decay_rate,
                bias_decay_warmup_batches=output_layer_bias_decay_warmup_batches,
                bias_bank_expansion_factor=output_layer_bias_bank_expansion_factor,
                diagonal_option=output_layer_diagonal_option,
                row_mask_option=output_layer_row_mask_option,
                mask_dimension_option=output_layer_mask_dimension_option,
                mask_threshold=output_layer_mask_threshold,
                mask_surrogate_scale=output_layer_mask_surrogate_scale,
                mask_floor=output_layer_mask_floor,
                mask_transition_width=output_layer_mask_transition_width,
            )
        )
        router_adaptive_weight_options = (
            router_adaptive_weight_options
            or HiddenAdaptiveWeightOptions(
                generator_depth=router_generator_depth,
                option_flag=self.__adaptive_option_flag(
                    router_weight_option_flag,
                    router_weight_option,
                    config.ROUTER_WEIGHT_OPTION_FLAG,
                ),
                option=router_weight_option,
                normalization_option=router_weight_normalization_option,
                normalization_position_option=(
                    router_weight_normalization_position_option
                ),
                decay_schedule=router_weight_decay_schedule,
                decay_rate=router_weight_decay_rate,
                decay_warmup_batches=router_weight_decay_warmup_batches,
                bank_expansion_factor=router_weight_bank_expansion_factor,
                generator_stack_source=AdaptiveGeneratorStackSource(
                    independent_flag=(router_weight_generator_stack_independent_flag),
                    hidden_dim=router_weight_generator_stack_hidden_dim,
                    layer_norm_position=(
                        router_weight_generator_stack_layer_norm_position
                    ),
                    num_layers=router_weight_generator_stack_num_layers,
                    activation=router_weight_generator_stack_activation,
                    residual_connection_option=(
                        router_weight_generator_stack_residual_connection_option
                    ),
                    dropout_probability=(
                        router_weight_generator_stack_dropout_probability
                    ),
                    last_layer_bias_option=(
                        router_weight_generator_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        router_weight_generator_stack_apply_output_pipeline_flag
                    ),
                    bias_flag=router_weight_generator_stack_bias_flag,
                ),
            )
        )
        router_adaptive_bias_options = (
            router_adaptive_bias_options
            or HiddenAdaptiveBiasOptions(
                option_flag=self.__adaptive_option_flag(
                    router_bias_option_flag,
                    router_bias_option,
                    config.ROUTER_BIAS_OPTION_FLAG,
                ),
                option=router_bias_option,
                decay_schedule=router_bias_decay_schedule,
                decay_rate=router_bias_decay_rate,
                decay_warmup_batches=router_bias_decay_warmup_batches,
                bank_expansion_factor=router_bias_bank_expansion_factor,
                generator_stack_source=AdaptiveGeneratorStackSource(
                    independent_flag=(router_bias_generator_stack_independent_flag),
                    hidden_dim=router_bias_generator_stack_hidden_dim,
                    layer_norm_position=(
                        router_bias_generator_stack_layer_norm_position
                    ),
                    num_layers=router_bias_generator_stack_num_layers,
                    activation=router_bias_generator_stack_activation,
                    residual_connection_option=(
                        router_bias_generator_stack_residual_connection_option
                    ),
                    dropout_probability=(
                        router_bias_generator_stack_dropout_probability
                    ),
                    last_layer_bias_option=(
                        router_bias_generator_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        router_bias_generator_stack_apply_output_pipeline_flag
                    ),
                    bias_flag=router_bias_generator_stack_bias_flag,
                ),
            )
        )
        router_adaptive_diagonal_options = (
            router_adaptive_diagonal_options
            or HiddenAdaptiveDiagonalOptions(
                option_flag=self.__adaptive_option_flag(
                    router_diagonal_option_flag,
                    router_diagonal_option,
                    config.ROUTER_DIAGONAL_OPTION_FLAG,
                ),
                option=router_diagonal_option,
                generator_stack_source=AdaptiveGeneratorStackSource(
                    independent_flag=(router_diagonal_generator_stack_independent_flag),
                    hidden_dim=router_diagonal_generator_stack_hidden_dim,
                    layer_norm_position=(
                        router_diagonal_generator_stack_layer_norm_position
                    ),
                    num_layers=router_diagonal_generator_stack_num_layers,
                    activation=router_diagonal_generator_stack_activation,
                    residual_connection_option=(
                        router_diagonal_generator_stack_residual_connection_option
                    ),
                    dropout_probability=(
                        router_diagonal_generator_stack_dropout_probability
                    ),
                    last_layer_bias_option=(
                        router_diagonal_generator_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        router_diagonal_generator_stack_apply_output_pipeline_flag
                    ),
                    bias_flag=router_diagonal_generator_stack_bias_flag,
                ),
            )
        )
        router_adaptive_mask_options = (
            router_adaptive_mask_options
            or HiddenAdaptiveMaskOptions(
                option_flag=self.__adaptive_option_flag(
                    router_mask_option_flag,
                    router_row_mask_option,
                    config.ROUTER_MASK_OPTION_FLAG,
                ),
                row_mask_option=router_row_mask_option,
                mask_dimension_option=router_mask_dimension_option,
                mask_threshold=router_mask_threshold,
                mask_surrogate_scale=router_mask_surrogate_scale,
                mask_floor=router_mask_floor,
                mask_transition_width=router_mask_transition_width,
                generator_stack_source=AdaptiveGeneratorStackSource(
                    independent_flag=router_mask_generator_stack_independent_flag,
                    hidden_dim=router_mask_generator_stack_hidden_dim,
                    layer_norm_position=(
                        router_mask_generator_stack_layer_norm_position
                    ),
                    num_layers=router_mask_generator_stack_num_layers,
                    activation=router_mask_generator_stack_activation,
                    residual_connection_option=(
                        router_mask_generator_stack_residual_connection_option
                    ),
                    dropout_probability=(
                        router_mask_generator_stack_dropout_probability
                    ),
                    last_layer_bias_option=(
                        router_mask_generator_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        router_mask_generator_stack_apply_output_pipeline_flag
                    ),
                    bias_flag=router_mask_generator_stack_bias_flag,
                ),
            )
        )
        recurrent_controller_options = (
            recurrent_controller_options
            or ExpertsRecurrentControllerOptions(
                recurrent_flag=recurrent_flag,
                recurrent_max_steps=recurrent_max_steps,
                recurrent_layer_norm_position=recurrent_layer_norm_position,
                recurrent_stack_gate_flag=recurrent_stack_gate_flag,
                recurrent_gate_option=recurrent_gate_option,
                recurrent_gate_activation=recurrent_gate_activation,
                recurrent_gate_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=recurrent_gate_stack_independent_flag,
                    hidden_dim=recurrent_gate_stack_hidden_dim,
                    num_layers=recurrent_gate_stack_num_layers,
                    last_layer_bias_option=(
                        recurrent_gate_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        recurrent_gate_stack_apply_output_pipeline_flag
                    ),
                    activation=recurrent_gate_stack_activation,
                    layer_norm_position=recurrent_gate_stack_layer_norm_position,
                    residual_connection_option=(
                        recurrent_gate_stack_residual_connection_option
                    ),
                    dropout_probability=recurrent_gate_stack_dropout_probability,
                    bias_flag=recurrent_gate_stack_bias_flag,
                ),
                recurrent_stack_halting_flag=recurrent_stack_halting_flag,
                recurrent_halting_threshold=recurrent_halting_threshold,
                recurrent_halting_dropout=recurrent_halting_dropout,
                recurrent_halting_hidden_state_mode=(
                    recurrent_halting_hidden_state_mode
                ),
                recurrent_halting_stack_source=ExpertsSubmoduleStackSource(
                    independent_flag=recurrent_halting_stack_independent_flag,
                    hidden_dim=recurrent_halting_stack_hidden_dim,
                    num_layers=recurrent_halting_stack_num_layers,
                    last_layer_bias_option=(
                        recurrent_halting_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        recurrent_halting_stack_apply_output_pipeline_flag
                    ),
                    activation=recurrent_halting_stack_activation,
                    layer_norm_position=(recurrent_halting_stack_layer_norm_position),
                    residual_connection_option=(
                        recurrent_halting_stack_residual_connection_option
                    ),
                    dropout_probability=(recurrent_halting_stack_dropout_probability),
                    bias_flag=recurrent_halting_stack_bias_flag,
                ),
            )
        )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.stack_options = stack_options
        self.hidden_dim = stack_options.hidden_dim
        self.output_dim = output_dim
        self.bias_flag = stack_options.bias_flag
        self.layer_norm_position = stack_options.layer_norm_position
        self.stack_num_layers = stack_options.num_layers
        self.stack_activation = stack_options.activation
        self.stack_residual_connection_option = stack_options.residual_connection_option
        self.stack_dropout_probability = stack_options.dropout_probability
        self.stack_last_layer_bias_option = stack_options.last_layer_bias_option
        self.stack_apply_output_pipeline_flag = stack_options.apply_output_pipeline_flag
        self.submodule_stack_options = submodule_stack_options
        self.submodule_stack_hidden_dim = submodule_stack_options.hidden_dim
        self.submodule_stack_num_layers = submodule_stack_options.num_layers
        self.submodule_stack_activation = submodule_stack_options.activation
        self.submodule_stack_residual_connection_option = (
            submodule_stack_options.residual_connection_option
        )
        self.submodule_stack_dropout_probability = (
            submodule_stack_options.dropout_probability
        )
        self.submodule_stack_layer_norm_position = (
            submodule_stack_options.layer_norm_position
        )
        self.submodule_stack_last_layer_bias_option = (
            submodule_stack_options.last_layer_bias_option
        )
        self.submodule_stack_apply_output_pipeline_flag = (
            submodule_stack_options.apply_output_pipeline_flag
        )
        self.submodule_stack_bias_flag = submodule_stack_options.bias_flag
        self.mixture_options = mixture_options
        self.top_k = mixture_options.top_k
        self.num_experts = mixture_options.num_experts
        self.capacity_factor = mixture_options.capacity_factor
        self.dropped_token_behavior = mixture_options.dropped_token_behavior
        self.compute_expert_mixture_flag = mixture_options.compute_expert_mixture_flag
        self.weighted_parameters_flag = mixture_options.weighted_parameters_flag
        self.weighting_position_option = mixture_options.weighting_position_option
        self.routing_initialization_mode = mixture_options.routing_initialization_mode
        self.expert_stack_options = expert_stack_options
        self.expert_stack_num_layers = expert_stack_options.num_layers
        self.expert_stack_activation = expert_stack_options.activation
        self.expert_stack_residual_connection_option = (
            expert_stack_options.residual_connection_option
        )
        self.expert_stack_dropout_probability = expert_stack_options.dropout_probability
        self.expert_stack_layer_norm_position = expert_stack_options.layer_norm_position
        self.expert_stack_last_layer_bias_option = (
            expert_stack_options.last_layer_bias_option
        )
        self.expert_stack_apply_output_pipeline_flag = (
            expert_stack_options.apply_output_pipeline_flag
        )
        self.expert_bias_flag = expert_stack_options.bias_flag
        self.expert_layer_controller_options = expert_layer_controller_options
        self.expert_stack_gate_flag = expert_layer_controller_options.stack_gate_flag
        self.expert_gate_option = expert_layer_controller_options.gate_option
        self.expert_gate_activation = expert_layer_controller_options.gate_activation
        self.expert_gate_stack_source = (
            expert_layer_controller_options.gate_stack_source
        )
        self.expert_gate_stack_options = resolve_experts_controller_stack_options(
            self.expert_gate_stack_source,
            self.expert_stack_options,
        )
        self.expert_stack_halting_flag = (
            expert_layer_controller_options.stack_halting_flag
        )
        self.expert_halting_threshold = (
            expert_layer_controller_options.halting_threshold
        )
        self.expert_halting_dropout = expert_layer_controller_options.halting_dropout
        self.expert_halting_hidden_state_mode = (
            expert_layer_controller_options.halting_hidden_state_mode
        )
        self.expert_halting_output_dim = (
            expert_layer_controller_options.halting_output_dim
        )
        self.expert_halting_stack_source = (
            expert_layer_controller_options.halting_stack_source
        )
        self.expert_halting_stack_options = resolve_experts_controller_stack_options(
            self.expert_halting_stack_source,
            self.expert_stack_options,
        )
        self.expert_dynamic_memory_options = expert_dynamic_memory_options
        self.expert_memory_flag = expert_dynamic_memory_options.memory_flag
        self.expert_memory_option = expert_dynamic_memory_options.memory_option
        self.expert_memory_position_option = (
            expert_dynamic_memory_options.memory_position_option
        )
        self.expert_memory_stack_source = (
            expert_dynamic_memory_options.memory_stack_source
        )
        self.expert_memory_stack_options = resolve_experts_controller_stack_options(
            self.expert_memory_stack_source,
            self.expert_stack_options,
        )
        self.expert_recurrent_controller_options = expert_recurrent_controller_options
        self.expert_recurrent_flag = expert_recurrent_controller_options.recurrent_flag
        self.expert_recurrent_max_steps = (
            expert_recurrent_controller_options.recurrent_max_steps
        )
        self.expert_recurrent_layer_norm_position = (
            expert_recurrent_controller_options.recurrent_layer_norm_position
        )
        self.expert_recurrent_stack_gate_flag = (
            expert_recurrent_controller_options.recurrent_stack_gate_flag
        )
        self.expert_recurrent_gate_stack_source = (
            expert_recurrent_controller_options.recurrent_gate_stack_source
        )
        self.expert_recurrent_gate_stack_options = (
            resolve_experts_controller_stack_options(
                self.expert_recurrent_gate_stack_source,
                self.expert_stack_options,
            )
        )
        self.expert_recurrent_stack_halting_flag = (
            expert_recurrent_controller_options.recurrent_stack_halting_flag
        )
        self.expert_recurrent_halting_stack_source = (
            expert_recurrent_controller_options.recurrent_halting_stack_source
        )
        self.expert_recurrent_halting_stack_options = (
            resolve_experts_controller_stack_options(
                self.expert_recurrent_halting_stack_source,
                self.expert_stack_options,
            )
        )
        self.sampler_options = sampler_options
        self.sampler_threshold = sampler_options.threshold
        self.sampler_filter_above_threshold = sampler_options.filter_above_threshold
        self.sampler_num_topk_samples = sampler_options.num_topk_samples
        self.sampler_normalize_probabilities_flag = (
            sampler_options.normalize_probabilities_flag
        )
        self.sampler_noisy_topk_flag = sampler_options.noisy_topk_flag
        self.sampler_coefficient_of_variation_loss_weight = (
            sampler_options.coefficient_of_variation_loss_weight
        )
        self.sampler_switch_loss_weight = sampler_options.switch_loss_weight
        self.sampler_zero_centred_loss_weight = sampler_options.zero_centred_loss_weight
        self.sampler_mutual_information_loss_weight = (
            sampler_options.mutual_information_loss_weight
        )
        self.router_options = router_options
        self.router_noisy_topk_flag = router_options.noisy_topk_flag
        self.router_stack_options = router_stack_options
        self.router_stack_hidden_dim = router_stack_options.hidden_dim
        self.router_stack_num_layers = router_stack_options.num_layers
        self.router_stack_activation = router_stack_options.activation
        self.router_stack_residual_connection_option = (
            router_stack_options.residual_connection_option
        )
        self.router_stack_dropout_probability = router_stack_options.dropout_probability
        self.router_stack_layer_norm_position = router_stack_options.layer_norm_position
        self.router_stack_last_layer_bias_option = (
            router_stack_options.last_layer_bias_option
        )
        self.router_stack_apply_output_pipeline_flag = (
            router_stack_options.apply_output_pipeline_flag
        )
        self.router_bias_flag = router_stack_options.bias_flag
        self.router_layer_controller_options = router_layer_controller_options
        self.router_stack_gate_flag = router_layer_controller_options.stack_gate_flag
        self.router_gate_option = router_layer_controller_options.gate_option
        self.router_gate_activation = router_layer_controller_options.gate_activation
        self.router_gate_stack_source = (
            router_layer_controller_options.gate_stack_source
        )
        self.router_gate_stack_options = resolve_experts_controller_stack_options(
            self.router_gate_stack_source,
            self.submodule_stack_options,
        )
        self.router_stack_halting_flag = (
            router_layer_controller_options.stack_halting_flag
        )
        self.router_halting_threshold = (
            router_layer_controller_options.halting_threshold
        )
        self.router_halting_dropout = router_layer_controller_options.halting_dropout
        self.router_halting_hidden_state_mode = (
            router_layer_controller_options.halting_hidden_state_mode
        )
        self.router_halting_output_dim = (
            router_layer_controller_options.halting_output_dim
        )
        self.router_halting_stack_source = (
            router_layer_controller_options.halting_stack_source
        )
        self.router_halting_stack_options = resolve_experts_controller_stack_options(
            self.router_halting_stack_source,
            replace(
                self.submodule_stack_options,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            ),
        )
        self.router_dynamic_memory_options = router_dynamic_memory_options
        self.router_memory_flag = router_dynamic_memory_options.memory_flag
        self.router_memory_option = router_dynamic_memory_options.memory_option
        self.router_memory_position_option = (
            router_dynamic_memory_options.memory_position_option
        )
        self.router_memory_stack_source = (
            router_dynamic_memory_options.memory_stack_source
        )
        self.router_memory_stack_options = resolve_experts_controller_stack_options(
            self.router_memory_stack_source,
            self.submodule_stack_options,
        )
        self.router_recurrent_controller_options = router_recurrent_controller_options
        self.router_recurrent_flag = router_recurrent_controller_options.recurrent_flag
        self.router_recurrent_max_steps = (
            router_recurrent_controller_options.recurrent_max_steps
        )
        self.router_recurrent_layer_norm_position = (
            router_recurrent_controller_options.recurrent_layer_norm_position
        )
        self.router_recurrent_stack_gate_flag = (
            router_recurrent_controller_options.recurrent_stack_gate_flag
        )
        self.router_recurrent_gate_stack_source = (
            router_recurrent_controller_options.recurrent_gate_stack_source
        )
        self.router_recurrent_gate_stack_options = (
            resolve_experts_controller_stack_options(
                self.router_recurrent_gate_stack_source,
                self.submodule_stack_options,
            )
        )
        self.router_recurrent_stack_halting_flag = (
            router_recurrent_controller_options.recurrent_stack_halting_flag
        )
        self.router_recurrent_halting_stack_source = (
            router_recurrent_controller_options.recurrent_halting_stack_source
        )
        self.router_recurrent_halting_stack_options = (
            resolve_experts_controller_stack_options(
                self.router_recurrent_halting_stack_source,
                replace(
                    self.submodule_stack_options,
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                ),
            )
        )
        self.layer_controller_options = layer_controller_options
        self.stack_gate_flag = layer_controller_options.stack_gate_flag
        self.gate_option = layer_controller_options.gate_option
        self.gate_activation = layer_controller_options.gate_activation
        self.gate_stack_source = layer_controller_options.gate_stack_source
        self.gate_stack_options = resolve_experts_controller_stack_options(
            self.gate_stack_source,
            self.submodule_stack_options,
        )
        self.gate_stack_hidden_dim = self.gate_stack_options.hidden_dim
        self.gate_stack_layer_norm_position = (
            self.gate_stack_options.layer_norm_position
        )
        self.gate_stack_num_layers = self.gate_stack_options.num_layers
        self.gate_stack_activation = self.gate_stack_options.activation
        self.gate_stack_residual_connection_option = (
            self.gate_stack_options.residual_connection_option
        )
        self.gate_stack_dropout_probability = (
            self.gate_stack_options.dropout_probability
        )
        self.gate_stack_last_layer_bias_option = (
            self.gate_stack_options.last_layer_bias_option
        )
        self.gate_stack_apply_output_pipeline_flag = (
            self.gate_stack_options.apply_output_pipeline_flag
        )
        self.gate_stack_bias_flag = self.gate_stack_options.bias_flag
        self.shared_gate_config = layer_controller_options.shared_gate_config
        self.stack_halting_flag = layer_controller_options.stack_halting_flag
        self.halting_threshold = layer_controller_options.halting_threshold
        self.halting_dropout = layer_controller_options.halting_dropout
        self.halting_hidden_state_mode = (
            layer_controller_options.halting_hidden_state_mode
        )
        self.halting_stack_source = layer_controller_options.halting_stack_source
        self.halting_stack_options = resolve_experts_controller_stack_options(
            self.halting_stack_source,
            replace(
                self.submodule_stack_options,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            ),
        )
        self.halting_stack_hidden_dim = self.halting_stack_options.hidden_dim
        self.halting_output_dim = layer_controller_options.halting_output_dim
        self.halting_stack_layer_norm_position = (
            self.halting_stack_options.layer_norm_position
        )
        self.halting_stack_num_layers = self.halting_stack_options.num_layers
        self.halting_stack_activation = self.halting_stack_options.activation
        self.halting_stack_residual_connection_option = (
            self.halting_stack_options.residual_connection_option
        )
        self.halting_stack_dropout_probability = (
            self.halting_stack_options.dropout_probability
        )
        self.halting_stack_last_layer_bias_option = (
            self.halting_stack_options.last_layer_bias_option
        )
        self.halting_stack_apply_output_pipeline_flag = (
            self.halting_stack_options.apply_output_pipeline_flag
        )
        self.halting_stack_bias_flag = self.halting_stack_options.bias_flag
        self.dynamic_memory_options = dynamic_memory_options
        self.memory_flag = dynamic_memory_options.memory_flag
        self.memory_option = dynamic_memory_options.memory_option
        self.memory_position_option = dynamic_memory_options.memory_position_option
        self.memory_test_time_training_learning_rate = (
            dynamic_memory_options.memory_test_time_training_learning_rate
        )
        self.memory_test_time_training_num_inner_steps = (
            dynamic_memory_options.memory_test_time_training_num_inner_steps
        )
        self.memory_stack_source = dynamic_memory_options.memory_stack_source
        self.memory_stack_options = resolve_experts_controller_stack_options(
            self.memory_stack_source,
            self.submodule_stack_options,
        )
        self.memory_stack_hidden_dim = self.memory_stack_options.hidden_dim
        self.memory_stack_layer_norm_position = (
            self.memory_stack_options.layer_norm_position
        )
        self.memory_stack_num_layers = self.memory_stack_options.num_layers
        self.memory_stack_activation = self.memory_stack_options.activation
        self.memory_stack_residual_connection_option = (
            self.memory_stack_options.residual_connection_option
        )
        self.memory_stack_dropout_probability = (
            self.memory_stack_options.dropout_probability
        )
        self.memory_stack_last_layer_bias_option = (
            self.memory_stack_options.last_layer_bias_option
        )
        self.memory_stack_apply_output_pipeline_flag = (
            self.memory_stack_options.apply_output_pipeline_flag
        )
        self.memory_stack_bias_flag = self.memory_stack_options.bias_flag
        self.hidden_adaptive_weight_options = hidden_adaptive_weight_options
        self.hidden_adaptive_bias_options = hidden_adaptive_bias_options
        self.hidden_adaptive_diagonal_options = hidden_adaptive_diagonal_options
        self.hidden_adaptive_mask_options = hidden_adaptive_mask_options
        self.input_boundary_options = input_boundary_options
        self.output_boundary_options = output_boundary_options
        self.router_adaptive_weight_options = router_adaptive_weight_options
        self.router_adaptive_bias_options = router_adaptive_bias_options
        self.router_adaptive_diagonal_options = router_adaptive_diagonal_options
        self.router_adaptive_mask_options = router_adaptive_mask_options
        self.weight_option_flag = hidden_adaptive_weight_options.option_flag
        self.generator_depth = hidden_adaptive_weight_options.generator_depth
        self.diagonal_option_flag = hidden_adaptive_diagonal_options.option_flag
        self.diagonal_option = hidden_adaptive_diagonal_options.option
        self.bias_option_flag = hidden_adaptive_bias_options.option_flag
        self.bias_option = hidden_adaptive_bias_options.option
        self.weight_option = hidden_adaptive_weight_options.option
        self.weight_normalization_option = (
            hidden_adaptive_weight_options.normalization_option
        )
        self.weight_normalization_position_option = (
            hidden_adaptive_weight_options.normalization_position_option
        )
        self.weight_decay_schedule = hidden_adaptive_weight_options.decay_schedule
        self.weight_decay_rate = hidden_adaptive_weight_options.decay_rate
        self.weight_decay_warmup_batches = (
            hidden_adaptive_weight_options.decay_warmup_batches
        )
        self.weight_bank_expansion_factor = (
            hidden_adaptive_weight_options.bank_expansion_factor
        )
        self.bias_decay_schedule = hidden_adaptive_bias_options.decay_schedule
        self.bias_decay_rate = hidden_adaptive_bias_options.decay_rate
        self.bias_decay_warmup_batches = (
            hidden_adaptive_bias_options.decay_warmup_batches
        )
        self.bias_bank_expansion_factor = (
            hidden_adaptive_bias_options.bank_expansion_factor
        )
        self.mask_option_flag = hidden_adaptive_mask_options.option_flag
        self.row_mask_option = hidden_adaptive_mask_options.row_mask_option
        self.mask_dimension_option = hidden_adaptive_mask_options.mask_dimension_option
        self.mask_threshold = hidden_adaptive_mask_options.mask_threshold
        self.mask_surrogate_scale = hidden_adaptive_mask_options.mask_surrogate_scale
        self.mask_floor = hidden_adaptive_mask_options.mask_floor
        self.mask_transition_width = hidden_adaptive_mask_options.mask_transition_width
        self.adaptive_generator_stack_options = adaptive_generator_stack_options
        self.adaptive_generator_stack_num_layers = (
            adaptive_generator_stack_options.num_layers
        )
        self.adaptive_generator_stack_hidden_dim = (
            adaptive_generator_stack_options.hidden_dim
        )
        self.adaptive_generator_stack_activation = (
            adaptive_generator_stack_options.activation
        )
        self.adaptive_generator_stack_residual_connection_option = (
            adaptive_generator_stack_options.residual_connection_option
        )
        self.adaptive_generator_stack_dropout_probability = (
            adaptive_generator_stack_options.dropout_probability
        )
        self.adaptive_generator_stack_layer_norm_position = (
            adaptive_generator_stack_options.layer_norm_position
        )
        self.adaptive_generator_stack_last_layer_bias_option = (
            adaptive_generator_stack_options.last_layer_bias_option
        )
        self.adaptive_generator_stack_apply_output_pipeline_flag = (
            adaptive_generator_stack_options.apply_output_pipeline_flag
        )
        self.adaptive_generator_stack_bias_flag = (
            adaptive_generator_stack_options.bias_flag
        )
        self.recurrent_controller_options = recurrent_controller_options
        self.recurrent_flag = recurrent_controller_options.recurrent_flag
        self.recurrent_max_steps = recurrent_controller_options.recurrent_max_steps
        self.recurrent_layer_norm_position = (
            recurrent_controller_options.recurrent_layer_norm_position
        )
        self.recurrent_stack_gate_flag = (
            recurrent_controller_options.recurrent_stack_gate_flag
        )
        self.recurrent_gate_option = recurrent_controller_options.recurrent_gate_option
        self.recurrent_gate_activation = (
            recurrent_controller_options.recurrent_gate_activation
        )
        self.recurrent_gate_stack_source = (
            recurrent_controller_options.recurrent_gate_stack_source
        )
        self.recurrent_gate_stack_options = resolve_experts_controller_stack_options(
            self.recurrent_gate_stack_source,
            self.gate_stack_options,
        )
        self.recurrent_stack_halting_flag = (
            recurrent_controller_options.recurrent_stack_halting_flag
        )
        self.recurrent_halting_threshold = (
            recurrent_controller_options.recurrent_halting_threshold
        )
        self.recurrent_halting_dropout = (
            recurrent_controller_options.recurrent_halting_dropout
        )
        self.recurrent_halting_hidden_state_mode = (
            recurrent_controller_options.recurrent_halting_hidden_state_mode
        )
        self.recurrent_halting_stack_source = (
            recurrent_controller_options.recurrent_halting_stack_source
        )
        self.recurrent_halting_stack_options = resolve_experts_controller_stack_options(
            self.recurrent_halting_stack_source,
            self.halting_stack_options,
        )

    @staticmethod
    def __adaptive_option_flag(
        explicit_flag: bool | None,
        option: type | None,
        default_flag: bool,
    ) -> bool:
        if explicit_flag is not None:
            return explicit_flag
        if option is not None:
            return True
        return default_flag

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=self.__input_model_config(),
                model_config=self.__model_config(),
                output_model_config=self.__output_model_config(),
            ),
        )

    def __input_model_config(self):
        boundary_model_config_dependencies = self.__boundary_model_config_dependencies()
        boundary_model_config_factory = BoundaryModelConfigFactory(
            boundary_model_config_dependencies
        )
        return boundary_model_config_factory.build_input_model_config()

    def __model_config(self):
        control_dependencies = self.__control_config_dependencies()
        control_factory = ControlConfigFactory(control_dependencies)
        return control_factory.build()

    def __output_model_config(self):
        boundary_model_config_dependencies = self.__boundary_model_config_dependencies()
        boundary_model_config_factory = BoundaryModelConfigFactory(
            boundary_model_config_dependencies
        )
        return boundary_model_config_factory.build_output_model_config()

    def __boundary_model_config_dependencies(
        self,
    ) -> BoundaryModelConfigDependencies:
        return BoundaryModelConfigDependencies(
            stack_options=self.stack_options,
            input_boundary_options=self.input_boundary_options,
            output_boundary_options=self.output_boundary_options,
            adaptive_generator_stack_options=(self.adaptive_generator_stack_options),
        )

    def __control_config_dependencies(self) -> ControlConfigDependencies:
        return ControlConfigDependencies(
            stack_options=self.stack_options,
            submodule_stack_options=self.submodule_stack_options,
            mixture_options=self.mixture_options,
            expert_stack_options=self.expert_stack_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            router_stack_options=self.router_stack_options,
            router_layer_controller_options=self.router_layer_controller_options,
            router_dynamic_memory_options=self.router_dynamic_memory_options,
            router_recurrent_controller_options=(
                self.router_recurrent_controller_options
            ),
            layer_controller_options=self.layer_controller_options,
            dynamic_memory_options=self.dynamic_memory_options,
            recurrent_controller_options=self.recurrent_controller_options,
            expert_layer_controller_options=self.expert_layer_controller_options,
            expert_dynamic_memory_options=self.expert_dynamic_memory_options,
            expert_recurrent_controller_options=(
                self.expert_recurrent_controller_options
            ),
            adaptive_generator_stack_options=(self.adaptive_generator_stack_options),
            hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=(self.hidden_adaptive_diagonal_options),
            hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
            router_adaptive_weight_options=self.router_adaptive_weight_options,
            router_adaptive_bias_options=self.router_adaptive_bias_options,
            router_adaptive_diagonal_options=(self.router_adaptive_diagonal_options),
            router_adaptive_mask_options=self.router_adaptive_mask_options,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )

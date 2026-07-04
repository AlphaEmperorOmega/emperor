from dataclasses import replace

from emperor.base.layer.residual import ResidualConnectionOptions
import models.experts.experts_linear_adaptive.config as config

from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer.config import (
    LayerConfig,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.memory.config import DynamicMemoryConfig
from emperor.memory.options import MemoryPositionOptions
from emperor.augmentations.adaptive_parameters.core.bias import (
    DynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    AxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from models.experts._builder_options import (
    ExpertsAdaptiveGeneratorStackOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    resolve_experts_controller_stack_options,
    resolve_experts_submodule_stack_options,
)
from models.experts.experts_linear_adaptive._control_config_factory import (
    AdaptiveAugmentationDependencies,
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.experts.experts_linear_adaptive.experiment_config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExpertsLinearAdaptiveConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        stack_hidden_dim: int = config.STACK_HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        stack_bias_flag: bool = config.STACK_BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.STACK_LAYER_NORM_POSITION,
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
        top_k: int = config.EXPERT_TOP_K,
        num_experts: int = config.EXPERT_NUM_EXPERTS,
        capacity_factor: float = config.EXPERT_CAPACITY_FACTOR,
        dropped_token_behavior: DroppedTokenOptions = config.EXPERT_DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag: bool = config.EXPERT_COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag: bool = config.EXPERT_WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option: ExpertWeightingPositionOptions = config.EXPERT_WEIGHTING_POSITION_OPTION,
        routing_initialization_mode: RoutingInitializationMode = config.EXPERT_ROUTING_INITIALIZATION_MODE,
        expert_stack_hidden_dim: int | None = None,
        expert_stack_num_layers: int | None = None,
        expert_stack_activation: ActivationOptions | None = None,
        expert_stack_residual_connection_option: ResidualConnectionOptions | None = None,
        expert_stack_dropout_probability: float | None = None,
        expert_stack_layer_norm_position: LayerNormPositionOptions | None = config.EXPERT_STACK_LAYER_NORM_POSITION,
        expert_stack_last_layer_bias_option: LastLayerBiasOptions | None = None,
        expert_stack_apply_output_pipeline_flag: bool | None = config.EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        expert_bias_flag: bool | None = None,
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
        sampler_stack_hidden_dim: int | None = None,
        sampler_stack_num_layers: int | None = None,
        sampler_stack_activation: ActivationOptions | None = None,
        sampler_stack_residual_connection_option: ResidualConnectionOptions | None = None,
        sampler_stack_dropout_probability: float | None = None,
        sampler_stack_layer_norm_position: LayerNormPositionOptions | None = config.SAMPLER_STACK_LAYER_NORM_POSITION,
        sampler_stack_last_layer_bias_option: LastLayerBiasOptions | None = None,
        sampler_stack_apply_output_pipeline_flag: bool | None = config.SAMPLER_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        sampler_bias_flag: bool | None = None,
        stack_gate_flag: bool = config.GATE_FLAG,
        gate_option: LayerGateOptions | None = config.GATE_OPTION,
        gate_activation: ActivationOptions | None = config.GATE_ACTIVATION,
        gate_stack_independent_flag: bool = config.GATE_STACK_INDEPENDENT_FLAG,
        gate_stack_hidden_dim: int | None = config.GATE_STACK_HIDDEN_DIM,
        gate_stack_layer_norm_position: LayerNormPositionOptions | None = config.GATE_STACK_LAYER_NORM_POSITION,
        gate_stack_num_layers: int | None = config.GATE_STACK_NUM_LAYERS,
        gate_stack_activation: ActivationOptions | None = config.GATE_STACK_ACTIVATION,
        gate_stack_residual_connection_option: ResidualConnectionOptions | None = config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        gate_stack_dropout_probability: float | None = config.GATE_STACK_DROPOUT_PROBABILITY,
        gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        gate_stack_apply_output_pipeline_flag: bool | None = config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        gate_stack_bias_flag: bool | None = config.GATE_STACK_BIAS_FLAG,
        stack_halting_flag: bool = config.HALTING_FLAG,
        halting_threshold: float = config.HALTING_THRESHOLD,
        halting_dropout: float = config.HALTING_DROPOUT,
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.HALTING_HIDDEN_STATE_MODE,
        halting_output_dim: int = config.HALTING_OUTPUT_DIM,
        halting_stack_independent_flag: bool = config.HALTING_STACK_INDEPENDENT_FLAG,
        halting_stack_hidden_dim: int | None = config.HALTING_STACK_HIDDEN_DIM,
        halting_stack_layer_norm_position: LayerNormPositionOptions | None = config.HALTING_STACK_LAYER_NORM_POSITION,
        halting_stack_num_layers: int | None = config.HALTING_STACK_NUM_LAYERS,
        halting_stack_activation: ActivationOptions | None = config.HALTING_STACK_ACTIVATION,
        halting_stack_residual_connection_option: ResidualConnectionOptions | None = config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        halting_stack_dropout_probability: float | None = config.HALTING_STACK_DROPOUT_PROBABILITY,
        halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        halting_stack_apply_output_pipeline_flag: bool | None = config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        halting_stack_bias_flag: bool | None = config.HALTING_STACK_BIAS_FLAG,
        memory_flag: bool = config.MEMORY_FLAG,
        memory_option: type[DynamicMemoryConfig] = config.MEMORY_OPTION,
        memory_position_option: MemoryPositionOptions = config.MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate: float | None = config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
        memory_test_time_training_num_inner_steps: int | None = config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
        memory_stack_independent_flag: bool = config.MEMORY_STACK_INDEPENDENT_FLAG,
        memory_stack_hidden_dim: int | None = config.MEMORY_STACK_HIDDEN_DIM,
        memory_stack_layer_norm_position: LayerNormPositionOptions | None = config.MEMORY_STACK_LAYER_NORM_POSITION,
        memory_stack_num_layers: int | None = config.MEMORY_STACK_NUM_LAYERS,
        memory_stack_activation: ActivationOptions | None = config.MEMORY_STACK_ACTIVATION,
        memory_stack_residual_connection_option: ResidualConnectionOptions | None = config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
        memory_stack_dropout_probability: float | None = config.MEMORY_STACK_DROPOUT_PROBABILITY,
        memory_stack_last_layer_bias_option: LastLayerBiasOptions | None = config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        memory_stack_apply_output_pipeline_flag: bool | None = config.MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        memory_stack_bias_flag: bool | None = config.MEMORY_STACK_BIAS_FLAG,
        generator_depth: DynamicDepthOptions = config.WEIGHT_GENERATOR_DEPTH,
        diagonal_option: type[DynamicDiagonalConfig] | None = config.DIAGONAL_OPTION,
        bias_option: type[DynamicBiasConfig] | None = config.BIAS_OPTION,
        weight_option: type[DynamicWeightConfig] | None = config.WEIGHT_OPTION,
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
        row_mask_option: type[AxisMaskConfig] | None = config.ROW_MASK_OPTION,
        mask_dimension_option: MaskDimensionOptions = config.MASK_DIMENSION_OPTION,
        mask_threshold: float = config.MASK_THRESHOLD,
        mask_surrogate_scale: float = config.MASK_SURROGATE_SCALE,
        mask_floor: float = config.MASK_FLOOR,
        mask_transition_width: float = config.MASK_TRANSITION_WIDTH,
        adaptive_generator_stack_num_layers: int = config.ADAPTIVE_STACK_NUM_LAYERS,
        adaptive_generator_stack_hidden_dim: int = config.ADAPTIVE_STACK_HIDDEN_DIM,
        adaptive_generator_stack_activation: ActivationOptions = config.ADAPTIVE_STACK_ACTIVATION,
        adaptive_generator_stack_residual_connection_option: ResidualConnectionOptions = config.ADAPTIVE_STACK_RESIDUAL_CONNECTION_OPTION,
        adaptive_generator_stack_dropout_probability: float = config.ADAPTIVE_STACK_DROPOUT_PROBABILITY,
        adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = config.ADAPTIVE_STACK_LAYER_NORM_POSITION,
        adaptive_generator_stack_last_layer_bias_option: LastLayerBiasOptions = config.ADAPTIVE_STACK_LAST_LAYER_BIAS_OPTION,
        adaptive_generator_stack_apply_output_pipeline_flag: bool = config.ADAPTIVE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        recurrent_flag: bool = config.RECURRENT_FLAG,
        recurrent_max_steps: int = config.RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position: LayerNormPositionOptions = config.RECURRENT_LAYER_NORM_POSITION,
        recurrent_gate_flag: bool = config.RECURRENT_GATE_FLAG,
        recurrent_gate_option: LayerGateOptions | None = config.RECURRENT_GATE_OPTION,
        recurrent_gate_activation: ActivationOptions | None = config.RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_independent_flag: bool = config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
        recurrent_gate_stack_hidden_dim: int | None = config.RECURRENT_GATE_STACK_HIDDEN_DIM,
        recurrent_gate_stack_layer_norm_position: LayerNormPositionOptions | None = config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
        recurrent_gate_stack_num_layers: int | None = config.RECURRENT_GATE_STACK_NUM_LAYERS,
        recurrent_gate_stack_activation: ActivationOptions | None = config.RECURRENT_GATE_STACK_ACTIVATION,
        recurrent_gate_stack_residual_connection_option: ResidualConnectionOptions | None = config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        recurrent_gate_stack_dropout_probability: float | None = config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
        recurrent_gate_stack_last_layer_bias_option: LastLayerBiasOptions | None = config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        recurrent_gate_stack_apply_output_pipeline_flag: bool | None = config.RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        recurrent_gate_stack_bias_flag: bool | None = config.RECURRENT_GATE_STACK_BIAS_FLAG,
        recurrent_halting_flag: bool = config.RECURRENT_HALTING_FLAG,
        recurrent_halting_threshold: float = config.RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout: float = config.RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.RECURRENT_HALTING_HIDDEN_STATE_MODE,
        recurrent_halting_stack_independent_flag: bool = config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        recurrent_halting_stack_hidden_dim: int | None = config.RECURRENT_HALTING_STACK_HIDDEN_DIM,
        recurrent_halting_stack_layer_norm_position: LayerNormPositionOptions | None = config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        recurrent_halting_stack_num_layers: int | None = config.RECURRENT_HALTING_STACK_NUM_LAYERS,
        recurrent_halting_stack_activation: ActivationOptions | None = config.RECURRENT_HALTING_STACK_ACTIVATION,
        recurrent_halting_stack_residual_connection_option: ResidualConnectionOptions | None = config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        recurrent_halting_stack_dropout_probability: float | None = config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        recurrent_halting_stack_last_layer_bias_option: LastLayerBiasOptions | None = config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        recurrent_halting_stack_apply_output_pipeline_flag: bool | None = config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        recurrent_halting_stack_bias_flag: bool | None = config.RECURRENT_HALTING_STACK_BIAS_FLAG,
        shared_gate_config: GateConfig | None = None,
        stack_options: ExpertsStackOptions | None = None,
        submodule_stack_options: ExpertsSubmoduleStackOptions | None = None,
        mixture_options: ExpertsMixtureOptions | None = None,
        expert_stack_options: ExpertsSubmoduleStackOptions | None = None,
        sampler_options: ExpertsSamplerOptions | None = None,
        router_options: ExpertsRouterOptions | None = None,
        sampler_stack_options: ExpertsSubmoduleStackOptions | None = None,
        layer_controller_options: ExpertsLayerControllerOptions | None = None,
        dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        adaptive_generator_stack_options: (
            ExpertsAdaptiveGeneratorStackOptions | None
        ) = None,
        recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None,
    ) -> None:
        stack_options = stack_options or ExpertsStackOptions(
            hidden_dim=stack_hidden_dim,
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
                residual_connection_option=(
                    submodule_stack_residual_connection_option
                ),
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
        sampler_stack_options = (
            sampler_stack_options
            or resolve_experts_submodule_stack_options(
                submodule_stack_options,
                hidden_dim=sampler_stack_hidden_dim,
                num_layers=sampler_stack_num_layers,
                last_layer_bias_option=sampler_stack_last_layer_bias_option,
                apply_output_pipeline_flag=sampler_stack_apply_output_pipeline_flag,
                activation=sampler_stack_activation,
                layer_norm_position=sampler_stack_layer_norm_position,
                residual_connection_option=sampler_stack_residual_connection_option,
                dropout_probability=sampler_stack_dropout_probability,
                bias_flag=sampler_bias_flag,
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
        dynamic_memory_options = (
            dynamic_memory_options
            or ExpertsDynamicMemoryOptions(
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
                    apply_output_pipeline_flag=(
                        memory_stack_apply_output_pipeline_flag
                    ),
                    activation=memory_stack_activation,
                    layer_norm_position=memory_stack_layer_norm_position,
                    residual_connection_option=(
                        memory_stack_residual_connection_option
                    ),
                    dropout_probability=memory_stack_dropout_probability,
                    bias_flag=memory_stack_bias_flag,
                ),
            )
        )
        adaptive_generator_stack_options = (
            adaptive_generator_stack_options
            or ExpertsAdaptiveGeneratorStackOptions(
                hidden_dim=adaptive_generator_stack_hidden_dim,
                num_layers=adaptive_generator_stack_num_layers,
                last_layer_bias_option=(
                    adaptive_generator_stack_last_layer_bias_option
                ),
                apply_output_pipeline_flag=(
                    adaptive_generator_stack_apply_output_pipeline_flag
                ),
                activation=adaptive_generator_stack_activation,
                layer_norm_position=adaptive_generator_stack_layer_norm_position,
                residual_connection_option=(
                    adaptive_generator_stack_residual_connection_option
                ),
                dropout_probability=adaptive_generator_stack_dropout_probability,
            )
        )
        recurrent_controller_options = (
            recurrent_controller_options
            or ExpertsRecurrentControllerOptions(
                recurrent_flag=recurrent_flag,
                recurrent_max_steps=recurrent_max_steps,
                recurrent_layer_norm_position=recurrent_layer_norm_position,
                recurrent_gate_flag=recurrent_gate_flag,
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
                recurrent_halting_flag=recurrent_halting_flag,
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
                    layer_norm_position=(
                        recurrent_halting_stack_layer_norm_position
                    ),
                    residual_connection_option=(
                        recurrent_halting_stack_residual_connection_option
                    ),
                    dropout_probability=(
                        recurrent_halting_stack_dropout_probability
                    ),
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
        self.stack_residual_connection_option = (
            stack_options.residual_connection_option
        )
        self.stack_dropout_probability = stack_options.dropout_probability
        self.stack_last_layer_bias_option = stack_options.last_layer_bias_option
        self.stack_apply_output_pipeline_flag = (
            stack_options.apply_output_pipeline_flag
        )
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
        self.compute_expert_mixture_flag = (
            mixture_options.compute_expert_mixture_flag
        )
        self.weighted_parameters_flag = mixture_options.weighted_parameters_flag
        self.weighting_position_option = mixture_options.weighting_position_option
        self.routing_initialization_mode = (
            mixture_options.routing_initialization_mode
        )
        self.expert_stack_options = expert_stack_options
        self.expert_stack_num_layers = expert_stack_options.num_layers
        self.expert_stack_activation = expert_stack_options.activation
        self.expert_stack_residual_connection_option = (
            expert_stack_options.residual_connection_option
        )
        self.expert_stack_dropout_probability = (
            expert_stack_options.dropout_probability
        )
        self.expert_stack_layer_norm_position = (
            expert_stack_options.layer_norm_position
        )
        self.expert_stack_last_layer_bias_option = (
            expert_stack_options.last_layer_bias_option
        )
        self.expert_stack_apply_output_pipeline_flag = (
            expert_stack_options.apply_output_pipeline_flag
        )
        self.expert_bias_flag = expert_stack_options.bias_flag
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
        self.sampler_zero_centred_loss_weight = (
            sampler_options.zero_centred_loss_weight
        )
        self.sampler_mutual_information_loss_weight = (
            sampler_options.mutual_information_loss_weight
        )
        self.router_options = router_options
        self.router_noisy_topk_flag = router_options.noisy_topk_flag
        self.sampler_stack_options = sampler_stack_options
        self.sampler_stack_num_layers = sampler_stack_options.num_layers
        self.sampler_stack_activation = sampler_stack_options.activation
        self.sampler_stack_residual_connection_option = (
            sampler_stack_options.residual_connection_option
        )
        self.sampler_stack_dropout_probability = (
            sampler_stack_options.dropout_probability
        )
        self.sampler_stack_layer_norm_position = (
            sampler_stack_options.layer_norm_position
        )
        self.sampler_stack_last_layer_bias_option = (
            sampler_stack_options.last_layer_bias_option
        )
        self.sampler_stack_apply_output_pipeline_flag = (
            sampler_stack_options.apply_output_pipeline_flag
        )
        self.sampler_bias_flag = sampler_stack_options.bias_flag
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
        self.generator_depth = generator_depth
        self.diagonal_option = diagonal_option
        self.bias_option = bias_option
        self.weight_option = weight_option
        self.weight_normalization_option = weight_normalization_option
        self.weight_normalization_position_option = weight_normalization_position_option
        self.weight_decay_schedule = weight_decay_schedule
        self.weight_decay_rate = weight_decay_rate
        self.weight_decay_warmup_batches = weight_decay_warmup_batches
        self.weight_bank_expansion_factor = weight_bank_expansion_factor
        self.bias_decay_schedule = bias_decay_schedule
        self.bias_decay_rate = bias_decay_rate
        self.bias_decay_warmup_batches = bias_decay_warmup_batches
        self.bias_bank_expansion_factor = bias_bank_expansion_factor
        self.row_mask_option = row_mask_option
        self.mask_dimension_option = mask_dimension_option
        self.mask_threshold = mask_threshold
        self.mask_surrogate_scale = mask_surrogate_scale
        self.mask_floor = mask_floor
        self.mask_transition_width = mask_transition_width
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
        self.recurrent_controller_options = recurrent_controller_options
        self.recurrent_flag = recurrent_controller_options.recurrent_flag
        self.recurrent_max_steps = recurrent_controller_options.recurrent_max_steps
        self.recurrent_layer_norm_position = (
            recurrent_controller_options.recurrent_layer_norm_position
        )
        self.recurrent_gate_flag = recurrent_controller_options.recurrent_gate_flag
        self.recurrent_gate_option = (
            recurrent_controller_options.recurrent_gate_option
        )
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
        self.recurrent_halting_flag = (
            recurrent_controller_options.recurrent_halting_flag
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
        self.recurrent_halting_stack_options = (
            resolve_experts_controller_stack_options(
                self.recurrent_halting_stack_source,
                self.halting_stack_options,
            )
        )

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        adaptive_dependencies = self.__adaptive_augmentation_dependencies()
        control_dependencies = self.__control_config_dependencies(
            adaptive_dependencies
        )
        control_factory = ControlConfigFactory(control_dependencies)

        input_model_config = LayerConfig(
            activation=self.stack_activation,
            layer_norm_position=self.layer_norm_position,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=self.stack_dropout_probability,
            gate_config=None,
            halting_config=None,
            layer_model_config=control_factory.build_adaptive_linear_layer_config(
                self.bias_flag
            ),
        )

        model_config = control_factory.build()

        output_model_config = LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=control_factory.build_adaptive_linear_layer_config(
                self.bias_flag
            ),
        )

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=input_model_config,
                model_config=model_config,
                output_model_config=output_model_config,
            ),
        )

    def __control_config_dependencies(
        self,
        adaptive_dependencies: AdaptiveAugmentationDependencies,
    ) -> ControlConfigDependencies:
        return ControlConfigDependencies(
            stack_options=self.stack_options,
            submodule_stack_options=self.submodule_stack_options,
            mixture_options=self.mixture_options,
            expert_stack_options=self.expert_stack_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            sampler_stack_options=self.sampler_stack_options,
            layer_controller_options=self.layer_controller_options,
            dynamic_memory_options=self.dynamic_memory_options,
            recurrent_controller_options=self.recurrent_controller_options,
            adaptive_generator_stack_options=(
                self.adaptive_generator_stack_options
            ),
            adaptive_augmentation_options=adaptive_dependencies,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )

    def __adaptive_augmentation_dependencies(
        self,
    ) -> AdaptiveAugmentationDependencies:
        return AdaptiveAugmentationDependencies(
            generator_depth=self.generator_depth,
            diagonal_option=self.diagonal_option,
            bias_option=self.bias_option,
            weight_option=self.weight_option,
            weight_normalization_option=self.weight_normalization_option,
            weight_normalization_position_option=(
                self.weight_normalization_position_option
            ),
            weight_decay_schedule=self.weight_decay_schedule,
            weight_decay_rate=self.weight_decay_rate,
            weight_decay_warmup_batches=self.weight_decay_warmup_batches,
            weight_bank_expansion_factor=self.weight_bank_expansion_factor,
            bias_decay_schedule=self.bias_decay_schedule,
            bias_decay_rate=self.bias_decay_rate,
            bias_decay_warmup_batches=self.bias_decay_warmup_batches,
            bias_bank_expansion_factor=self.bias_bank_expansion_factor,
            row_mask_option=self.row_mask_option,
            mask_dimension_option=self.mask_dimension_option,
            mask_threshold=self.mask_threshold,
            mask_surrogate_scale=self.mask_surrogate_scale,
            mask_floor=self.mask_floor,
            mask_transition_width=self.mask_transition_width,
        )

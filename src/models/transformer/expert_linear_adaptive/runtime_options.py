from __future__ import annotations

from dataclasses import dataclass, field

from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GroupingConfig,
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.embedding.absolute import (
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
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
    NormalizationOptions,
    ResidualConfig,
)
from emperor.memory import (
    DynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
)


@dataclass(frozen=True)
class TransformerStackOptions:
    num_layers: int = 3
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    stack_gate_flag: bool = False
    stack_halting_flag: bool = False
    halting_option: type[HaltingConfig] = StickBreakingConfig
    halting_threshold: float | None = None
    memory_flag: bool = False
    recurrent_flag: bool = False
    recurrent_stack_gate_flag: bool = False
    recurrent_stack_halting_flag: bool = False
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig
    recurrent_halting_threshold: float | None = None
    recurrent_max_steps: int = 2
    recurrent_initial_iterations: int = 2
    recurrent_gradient_transition_count: int | None = None
    recurrent_no_gradient_transition_count: int | None = None
    recurrent_iteration_increment: int = 1
    recurrent_forward_calls_before_iteration_increment: int = 1
    recurrent_smooth_iteration_growth_flag: bool = False
    stack_residual_connection_option: type[ResidualConfig] | None = None
    stack_residual_model_flag: bool = field(default=False, kw_only=True)
    stack_residual_block_size: int | None = field(default=None, kw_only=True)
    stack_residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    recurrent_residual_connection_option: type[ResidualConfig] | None = None
    recurrent_residual_model_flag: bool = field(default=False, kw_only=True)
    recurrent_residual_block_size: int | None = field(default=None, kw_only=True)
    recurrent_residual_rms_norm_epsilon: float | None = field(
        default=None, kw_only=True
    )


@dataclass(frozen=True)
class SubmoduleStackOptions:
    hidden_dim: int = 128
    num_layers: int = 1
    last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
    apply_output_postprocessing_flag: bool = False
    activation: ActivationOptions = ActivationOptions.DISABLED
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None = None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float = 0.0
    bias_flag: bool = True


@dataclass(frozen=True)
class ControllerStackOptions:
    independent_flag: bool = False
    hidden_dim: int | None = None
    num_layers: int | None = None
    last_layer_bias_option: LastLayerBiasOptions | None = None
    apply_output_postprocessing_flag: bool | None = None
    activation: ActivationOptions | None = None
    layer_norm_position: LayerNormPositionOptions | None = None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    residual_connection_option: type[ResidualConfig] | None = None
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float | None = None
    bias_flag: bool | None = None


def resolve_controller_stack_options(
    source: ControllerStackOptions,
    defaults: SubmoduleStackOptions,
) -> SubmoduleStackOptions:
    if not source.independent_flag:
        return defaults
    return SubmoduleStackOptions(
        hidden_dim=defaults.hidden_dim
        if source.hidden_dim is None
        else source.hidden_dim,
        num_layers=defaults.num_layers
        if source.num_layers is None
        else source.num_layers,
        last_layer_bias_option=(
            defaults.last_layer_bias_option
            if source.last_layer_bias_option is None
            else source.last_layer_bias_option
        ),
        apply_output_postprocessing_flag=(
            defaults.apply_output_postprocessing_flag
            if source.apply_output_postprocessing_flag is None
            else source.apply_output_postprocessing_flag
        ),
        activation=(
            defaults.activation if source.activation is None else source.activation
        ),
        layer_norm_position=(
            defaults.layer_norm_position
            if source.layer_norm_position is None
            else source.layer_norm_position
        ),
        normalization=(
            defaults.normalization
            if source.normalization is None
            else source.normalization
        ),
        residual_connection_option=(
            defaults.residual_connection_option
            if source.residual_connection_option is None
            else source.residual_connection_option
        ),
        residual_block_size=source.residual_block_size,
        residual_rms_norm_epsilon=source.residual_rms_norm_epsilon,
        residual_model_flag=source.residual_model_flag,
        dropout_probability=(
            defaults.dropout_probability
            if source.dropout_probability is None
            else source.dropout_probability
        ),
        bias_flag=defaults.bias_flag if source.bias_flag is None else source.bias_flag,
    )


@dataclass(frozen=True)
class LayerControllerOptions:
    stack_gate_flag: bool = False
    gate_option: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
    gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID
    gate_stack_options: ControllerStackOptions = ControllerStackOptions()
    stack_halting_flag: bool = False
    halting_option: type[HaltingConfig] = StickBreakingConfig
    halting_threshold: float | None = None
    halting_dropout: float = 0.0
    halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        HaltingHiddenStateModeOptions.RAW
    )
    halting_stack_options: ControllerStackOptions = ControllerStackOptions()
    shared_gate_config: GateConfig | None = None


@dataclass(frozen=True)
class DynamicMemoryOptions:
    memory_flag: bool = False
    memory_option: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
    memory_position_option: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
    memory_test_time_training_learning_rate: float | None = None
    memory_test_time_training_num_inner_steps: int | None = None
    memory_stack_options: ControllerStackOptions = ControllerStackOptions()


@dataclass(frozen=True)
class RecurrentControllerOptions:
    recurrent_flag: bool = False
    recurrent_max_steps: int = 2
    recurrent_layer_norm_position: LayerNormPositionOptions = (
        LayerNormPositionOptions.DISABLED
    )
    recurrent_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    recurrent_stack_gate_flag: bool = False
    recurrent_gate_option: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
    recurrent_gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID
    recurrent_gate_stack_options: ControllerStackOptions = ControllerStackOptions()
    recurrent_stack_halting_flag: bool = False
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig
    recurrent_halting_threshold: float | None = None
    recurrent_halting_dropout: float = 0.0
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        HaltingHiddenStateModeOptions.RAW
    )
    recurrent_halting_stack_options: ControllerStackOptions = ControllerStackOptions()


@dataclass(frozen=True)
class TransformerAttentionOptions:
    num_heads: int = 4
    add_key_value_bias_flag: bool = False
    zero_attention_flag: bool = False
    stack_options: SubmoduleStackOptions = SubmoduleStackOptions()
    layer_controller_options: LayerControllerOptions = LayerControllerOptions()
    dynamic_memory_options: DynamicMemoryOptions = DynamicMemoryOptions()
    recurrent_controller_options: RecurrentControllerOptions = (
        RecurrentControllerOptions()
    )


@dataclass(frozen=True)
class TransformerFeedForwardOptions:
    stack_options: SubmoduleStackOptions = SubmoduleStackOptions(
        hidden_dim=512,
        num_layers=2,
        activation=ActivationOptions.RELU,
        dropout_probability=0.1,
    )
    layer_controller_options: LayerControllerOptions = LayerControllerOptions()
    dynamic_memory_options: DynamicMemoryOptions = DynamicMemoryOptions()
    recurrent_controller_options: RecurrentControllerOptions = (
        RecurrentControllerOptions()
    )


@dataclass(frozen=True)
class ExpertOptions:
    use_kv_expert_models_flag: bool | None = None
    num_experts: int = 4
    top_k: int = 2
    dropped_token_behavior: DroppedTokenOptions = DroppedTokenOptions.ZEROS
    compute_expert_mixture_flag: bool = True
    weighted_parameters_flag: bool = False
    weighting_position_option: ExpertWeightingPositionOptions = (
        ExpertWeightingPositionOptions.BEFORE_EXPERTS
    )
    routing_initialization_mode: RoutingInitializationMode = (
        RoutingInitializationMode.LAYER
    )
    sampler_threshold: float = 0.0
    sampler_filter_above_threshold: bool = False
    sampler_num_topk_samples: int = 0
    normalize_probabilities_flag: bool = True
    sampler_noisy_topk_flag: bool = False
    coefficient_of_variation_loss_weight: float = 0.0
    switch_loss_weight: float = 0.0
    zero_centred_loss_weight: float = 0.0
    mutual_information_loss_weight: float = 0.0
    capacity_factor: float = 0.0
    router_noisy_topk_flag: bool = False
    router_path_options: TransformerFeedForwardOptions = TransformerFeedForwardOptions(
        stack_options=SubmoduleStackOptions(
            hidden_dim=128,
            num_layers=2,
            activation=ActivationOptions.GELU,
        )
    )
    expert_path_options: TransformerFeedForwardOptions = TransformerFeedForwardOptions(
        stack_options=SubmoduleStackOptions(
            hidden_dim=128,
            num_layers=1,
            activation=ActivationOptions.RELU,
        )
    )


@dataclass(frozen=True)
class AdaptiveParameterOptions:
    weight_input_factor_source: LowRankFactorSourceOptions | None = None
    weight_output_factor_source: LowRankFactorSourceOptions | None = None
    weight_mixture_num_experts: int | None = None
    bias_mixture_num_experts: int | None = None
    weight_mixture_top_k: int | None = None
    bias_mixture_top_k: int | None = None
    weight_mixture_normalize_probabilities_flag: bool | None = None
    bias_mixture_normalize_probabilities_flag: bool | None = None
    weight_input_factor_generator_stack_options: ControllerStackOptions = (
        ControllerStackOptions()
    )
    weight_output_factor_generator_stack_options: ControllerStackOptions = (
        ControllerStackOptions()
    )
    weight_coefficient_generator_stack_options: ControllerStackOptions = (
        ControllerStackOptions()
    )
    weight_mixture_router_generator_stack_options: ControllerStackOptions = (
        ControllerStackOptions()
    )
    bias_mixture_router_generator_stack_options: ControllerStackOptions = (
        ControllerStackOptions()
    )
    grouping_config: GroupingConfig | None = None
    weight_option_flag: bool = True
    weight_option: type[DynamicWeightConfig] | None = None
    generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_ONE
    weight_decay_schedule: WeightDecayScheduleOptions = (
        WeightDecayScheduleOptions.DISABLED
    )
    weight_decay_rate: float = 0.0
    weight_decay_warmup_batches: int = 0
    weight_normalization_option: WeightNormalizationOptions = (
        WeightNormalizationOptions.DISABLED
    )
    weight_normalization_position_option: WeightNormalizationPositionOptions = (
        WeightNormalizationPositionOptions.DISABLED
    )
    weight_bank_expansion_factor: BankExpansionFactorOptions = (
        BankExpansionFactorOptions.FACTOR_OF_ONE
    )
    bias_option_flag: bool = True
    bias_option: type[DynamicBiasConfig] | None = None
    bias_decay_schedule: WeightDecayScheduleOptions = (
        WeightDecayScheduleOptions.DISABLED
    )
    bias_decay_rate: float = 0.0
    bias_decay_warmup_batches: int = 0
    bias_bank_expansion_factor: BankExpansionFactorOptions = (
        BankExpansionFactorOptions.FACTOR_OF_ONE
    )
    diagonal_option_flag: bool = True
    diagonal_option: type[DynamicDiagonalConfig] | None = None
    mask_option_flag: bool = True
    row_mask_option: type[AxisMaskConfig] | None = None
    mask_threshold: float = 0.5
    mask_surrogate_scale: float = 1.0
    mask_floor: float = 0.0
    mask_dimension_option: MaskDimensionOptions = MaskDimensionOptions.ROW
    mask_transition_width: float = 0.1
    generator_stack_options: SubmoduleStackOptions = SubmoduleStackOptions(
        hidden_dim=64,
        num_layers=1,
        activation=ActivationOptions.RELU,
    )
    weight_generator_stack_options: ControllerStackOptions = ControllerStackOptions()
    bias_generator_stack_options: ControllerStackOptions = ControllerStackOptions()
    diagonal_generator_stack_options: ControllerStackOptions = ControllerStackOptions()
    mask_generator_stack_options: ControllerStackOptions = ControllerStackOptions()


@dataclass(frozen=True)
class RuntimeOptions:
    encoder_output_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    decoder_output_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    batch_size: int = 64
    learning_rate: float = 1.0
    vocab_size: int = 8192
    model_dim: int = 128
    source_sequence_length: int = 64
    target_sequence_length: int = 64
    dropout_probability: float = 0.1
    residual_stack_independent_flag: bool = False
    residual_stack_hidden_dim: int | None = None
    residual_stack_layer_norm_position: LayerNormPositionOptions | None = None
    residual_stack_normalization: NormalizationOptions | None = field(
        default=None, kw_only=True
    )
    residual_stack_num_layers: int | None = None
    residual_stack_activation: ActivationOptions | None = None
    residual_stack_residual_connection_option: type[ResidualConfig] | None = None
    residual_stack_residual_model_flag: bool = False
    residual_stack_residual_block_size: int | None = field(default=None, kw_only=True)
    residual_stack_residual_rms_norm_epsilon: float | None = field(
        default=None, kw_only=True
    )
    residual_stack_dropout_probability: float | None = None
    residual_stack_last_layer_bias_option: LastLayerBiasOptions | None = None
    residual_stack_apply_output_postprocessing_flag: bool | None = None
    residual_stack_bias_flag: bool | None = None
    positional_embedding_option: type = TextSinusoidalPositionalEmbeddingConfig
    encoder_options: TransformerStackOptions = TransformerStackOptions()
    decoder_options: TransformerStackOptions = TransformerStackOptions()
    encoder_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    decoder_self_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    decoder_cross_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    encoder_feed_forward_options: TransformerFeedForwardOptions = (
        TransformerFeedForwardOptions()
    )
    decoder_feed_forward_options: TransformerFeedForwardOptions = (
        TransformerFeedForwardOptions()
    )
    attention_expert_options: ExpertOptions = ExpertOptions()
    feed_forward_expert_options: ExpertOptions = ExpertOptions()
    attention_projection_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    attention_expert_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    encoder_attention_expert_adaptive_options: AdaptiveParameterOptions | None = None
    decoder_self_attention_expert_adaptive_options: AdaptiveParameterOptions | None = (
        None
    )
    decoder_cross_attention_expert_adaptive_options: AdaptiveParameterOptions | None = (
        None
    )
    router_adaptive_options: AdaptiveParameterOptions = AdaptiveParameterOptions()
    feed_forward_adaptive_options: AdaptiveParameterOptions = AdaptiveParameterOptions()
    encoder_attention_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    decoder_self_attention_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    decoder_cross_attention_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    encoder_feed_forward_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    decoder_feed_forward_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )


__all__ = [
    "AdaptiveParameterOptions",
    "ControllerStackOptions",
    "DynamicMemoryOptions",
    "ExpertOptions",
    "LayerControllerOptions",
    "RecurrentControllerOptions",
    "RuntimeOptions",
    "SubmoduleStackOptions",
    "TransformerAttentionOptions",
    "TransformerFeedForwardOptions",
    "TransformerStackOptions",
]

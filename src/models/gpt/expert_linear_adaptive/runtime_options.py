from dataclasses import dataclass, field
from types import ModuleType
from typing import cast

from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GroupingConfig,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
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
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from model_runtime.packages.runtime_values import ResolvedRuntimeOptions
from models.gpt.expert_linear_adaptive._generation import (
    BiasGenerationOptions,
    WeightGenerationOptions,
)
from models.gpt.expert_linear_adaptive._residual import ResidualStackOptions


@dataclass(frozen=True, slots=True)
class ExpertsStackOptions:
    hidden_dim: int
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig]
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool


@dataclass(frozen=True, slots=True)
class ExpertsSubmoduleStackOptions:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: type[ResidualConfig]
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class ExpertsSubmoduleStackSource:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    dropout_probability: float | None
    bias_flag: bool | None


def resolve_experts_submodule_stack_options(
    defaults: ExpertsSubmoduleStackOptions,
    *,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    last_layer_bias_option: LastLayerBiasOptions | None = None,
    apply_output_postprocessing_flag: bool | None = None,
    activation: ActivationOptions | None = None,
    layer_norm_position: LayerNormPositionOptions | None = None,
    residual_connection_option: type[ResidualConfig] | None = None,
    residual_model_flag: bool | None = None,
    dropout_probability: float | None = None,
    bias_flag: bool | None = None,
) -> ExpertsSubmoduleStackOptions:
    return ExpertsSubmoduleStackOptions(
        hidden_dim=defaults.hidden_dim if hidden_dim is None else hidden_dim,
        num_layers=defaults.num_layers if num_layers is None else num_layers,
        last_layer_bias_option=(
            defaults.last_layer_bias_option
            if last_layer_bias_option is None
            else last_layer_bias_option
        ),
        apply_output_postprocessing_flag=(
            defaults.apply_output_postprocessing_flag
            if apply_output_postprocessing_flag is None
            else apply_output_postprocessing_flag
        ),
        activation=defaults.activation if activation is None else activation,
        layer_norm_position=(
            defaults.layer_norm_position
            if layer_norm_position is None
            else layer_norm_position
        ),
        residual_connection_option=(
            defaults.residual_connection_option
            if residual_connection_option is None
            else residual_connection_option
        ),
        residual_model_flag=(
            defaults.residual_model_flag
            if residual_model_flag is None
            else residual_model_flag
        ),
        residual_stack_options=defaults.residual_stack_options,
        dropout_probability=(
            defaults.dropout_probability
            if dropout_probability is None
            else dropout_probability
        ),
        bias_flag=defaults.bias_flag if bias_flag is None else bias_flag,
    )


def resolve_experts_controller_stack_options(
    source: ExpertsSubmoduleStackSource,
    defaults: ExpertsSubmoduleStackOptions,
) -> ExpertsSubmoduleStackOptions:
    if not source.independent_flag:
        return defaults
    return resolve_experts_submodule_stack_options(
        defaults,
        hidden_dim=source.hidden_dim,
        num_layers=source.num_layers,
        last_layer_bias_option=source.last_layer_bias_option,
        apply_output_postprocessing_flag=source.apply_output_postprocessing_flag,
        activation=source.activation,
        layer_norm_position=source.layer_norm_position,
        residual_connection_option=source.residual_connection_option,
        residual_model_flag=source.residual_model_flag,
        dropout_probability=source.dropout_probability,
        bias_flag=source.bias_flag,
    )


@dataclass(frozen=True, slots=True)
class ExpertsMixtureOptions:
    top_k: int
    num_experts: int
    capacity_factor: float
    dropped_token_behavior: DroppedTokenOptions
    compute_expert_mixture_flag: bool
    weighted_parameters_flag: bool
    weighting_position_option: ExpertWeightingPositionOptions
    routing_initialization_mode: RoutingInitializationMode


@dataclass(frozen=True, slots=True)
class ExpertsSamplerOptions:
    threshold: float
    filter_above_threshold: bool
    num_topk_samples: int
    normalize_probabilities_flag: bool
    noisy_topk_flag: bool
    coefficient_of_variation_loss_weight: float
    switch_loss_weight: float
    zero_centred_loss_weight: float
    mutual_information_loss_weight: float


@dataclass(frozen=True, slots=True)
class ExpertsRouterOptions:
    noisy_topk_flag: bool


@dataclass(frozen=True, slots=True)
class ExpertsLayerControllerOptions:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    gate_stack_source: ExpertsSubmoduleStackSource
    stack_halting_flag: bool
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    halting_stack_source: ExpertsSubmoduleStackSource
    halting_output_dim: int
    shared_gate_config: GateConfig | None = None
    halting_option: type[HaltingConfig] = StickBreakingConfig


@dataclass(frozen=True, slots=True)
class ExpertsDynamicMemoryOptions:
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_test_time_training_learning_rate: float | None
    memory_test_time_training_num_inner_steps: int | None
    memory_stack_source: ExpertsSubmoduleStackSource


@dataclass(frozen=True, slots=True)
class ExpertsRecurrentControllerOptions:
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_initial_iterations: int = field(default=2, kw_only=True)
    recurrent_gradient_transition_count: int | None = field(default=None, kw_only=True)
    recurrent_no_gradient_transition_count: int | None = field(
        default=None, kw_only=True
    )
    recurrent_iteration_increment: int = field(default=1, kw_only=True)
    recurrent_forward_calls_before_iteration_increment: int = field(
        default=1,
        kw_only=True,
    )
    recurrent_smooth_iteration_growth_flag: bool = field(
        default=False,
        kw_only=True,
    )
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_residual_connection_option: type[ResidualConfig] | None = field(
        default=None,
        kw_only=True,
    )
    recurrent_residual_model_flag: bool = field(default=False, kw_only=True)
    residual_stack_options: ResidualStackOptions | None = field(
        default=None,
        kw_only=True,
    )
    recurrent_stack_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_gate_stack_source: ExpertsSubmoduleStackSource
    recurrent_min_steps: int = field(default=1, kw_only=True)
    recurrent_stack_halting_flag: bool
    recurrent_halting_threshold: float
    recurrent_ponder_cost_weight: float = field(default=1.0, kw_only=True)
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    recurrent_halting_stack_source: ExpertsSubmoduleStackSource
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig


@dataclass(frozen=True, slots=True)
class ExpertsAdaptiveGeneratorStackOptions:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: type[ResidualConfig]
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float


@dataclass(frozen=True, slots=True)
class AdaptiveGeneratorStackSource:
    independent_flag: bool
    hidden_dim: int | None
    layer_norm_position: LayerNormPositionOptions | None
    num_layers: int | None
    activation: ActivationOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    dropout_probability: float | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    bias_flag: bool | None


@dataclass(frozen=True, slots=True)
class AdaptiveGeneratorStackOptions:
    hidden_dim: int
    layer_norm_position: LayerNormPositionOptions
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig]
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class HiddenAdaptiveWeightOptions:
    generation: WeightGenerationOptions = field(
        default_factory=WeightGenerationOptions, kw_only=True
    )
    generator_depth: DynamicDepthOptions
    option_flag: bool
    option: type[DynamicWeightConfig] | None
    normalization_option: WeightNormalizationOptions
    normalization_position_option: WeightNormalizationPositionOptions
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack_source: AdaptiveGeneratorStackSource


@dataclass(frozen=True, slots=True)
class HiddenAdaptiveBiasOptions:
    generation: BiasGenerationOptions = field(
        default_factory=BiasGenerationOptions, kw_only=True
    )
    option_flag: bool
    option: type[DynamicBiasConfig] | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack_source: AdaptiveGeneratorStackSource


@dataclass(frozen=True, slots=True)
class HiddenAdaptiveDiagonalOptions:
    option_flag: bool
    option: type[DynamicDiagonalConfig] | None
    generator_stack_source: AdaptiveGeneratorStackSource


@dataclass(frozen=True, slots=True)
class HiddenAdaptiveMaskOptions:
    option_flag: bool
    row_mask_option: type[AxisMaskConfig] | None
    mask_dimension_option: MaskDimensionOptions
    mask_threshold: float
    mask_surrogate_scale: float
    mask_floor: float
    mask_transition_width: float
    generator_stack_source: AdaptiveGeneratorStackSource


@dataclass(frozen=True, slots=True)
class SubmoduleStackSource:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool = field(default=False, kw_only=True)
    dropout_probability: float | None
    bias_flag: bool | None


@dataclass(frozen=True, slots=True)
class SubmoduleStackOptions:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: type[ResidualConfig]
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class MainLayerStackOptions:
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig]
    residual_model_flag: bool = field(default=False, kw_only=True)
    residual_stack_options: ResidualStackOptions | None = field(
        default=None, kw_only=True
    )
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool


@dataclass(frozen=True, slots=True)
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
    halting_option: type[HaltingConfig] = StickBreakingConfig


@dataclass(frozen=True, slots=True)
class DynamicMemoryOptions:
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_test_time_training_learning_rate: float | None
    memory_test_time_training_num_inner_steps: int | None
    memory_stack_source: SubmoduleStackSource


@dataclass(frozen=True, slots=True)
class RecurrentControllerOptions:
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_initial_iterations: int = field(default=2, kw_only=True)
    recurrent_gradient_transition_count: int | None = field(default=None, kw_only=True)
    recurrent_no_gradient_transition_count: int | None = field(
        default=None, kw_only=True
    )
    recurrent_iteration_increment: int = field(default=1, kw_only=True)
    recurrent_forward_calls_before_iteration_increment: int = field(
        default=1,
        kw_only=True,
    )
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_stack_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_gate_stack_source: SubmoduleStackSource
    recurrent_stack_halting_flag: bool
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions
    recurrent_halting_stack_source: SubmoduleStackSource
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig


@dataclass(frozen=True, slots=True)
class TransformerDecoderOptions:
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    dropout_probability: float
    layer_norm_position: LayerNormPositionOptions


@dataclass(frozen=True, slots=True)
class TransformerPositionalEmbeddingOptions:
    option: type[AbsolutePositionalEmbeddingConfig]
    padding_idx: int | None
    auto_expand_flag: bool


@dataclass(frozen=True, slots=True)
class TransformerAttentionOptions:
    num_heads: int
    num_layers: int
    bias_flag: bool
    add_key_value_bias_flag: bool


@dataclass(frozen=True, slots=True)
class TransformerFeedForwardOptions:
    num_layers: int
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class GptEmbeddingOptions:
    layer_norm_flag: bool
    dropout_probability: float


@dataclass(frozen=True, slots=True)
class GptLmHeadOptions:
    weight_tying_flag: bool
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class _ConstructionOptions:
    attention_grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    feed_forward_grouping_config: GroupingConfig | None = field(
        default=None, kw_only=True
    )
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    router_grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    batch_size: int
    learning_rate: float
    input_dim: int
    output_dim: int
    sequence_length: int
    embedding_options: GptEmbeddingOptions | None
    lm_head_options: GptLmHeadOptions | None
    decoder_options: TransformerDecoderOptions | None
    positional_embedding_options: TransformerPositionalEmbeddingOptions | None
    attention_options: TransformerAttentionOptions | None
    feed_forward_options: TransformerFeedForwardOptions | None
    attention_projection_stack_options: ExpertsSubmoduleStackOptions | None
    attention_projection_layer_controller_options: ExpertsLayerControllerOptions | None
    attention_projection_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    attention_projection_recurrent_controller_options: (
        ExpertsRecurrentControllerOptions | None
    )
    feed_forward_stack_options: ExpertsSubmoduleStackOptions | None
    feed_forward_layer_controller_options: ExpertsLayerControllerOptions | None
    feed_forward_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    feed_forward_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    submodule_stack_options: ExpertsSubmoduleStackOptions | None
    layer_controller_options: ExpertsLayerControllerOptions | None
    dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    mixture_options: ExpertsMixtureOptions | None
    mixture_submodule_stack_options: ExpertsSubmoduleStackOptions | None
    mixture_layer_controller_options: ExpertsLayerControllerOptions | None
    mixture_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    mixture_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_stack_options: ExpertsSubmoduleStackOptions | None
    sampler_options: ExpertsSamplerOptions | None
    router_options: ExpertsRouterOptions | None
    router_stack_options: ExpertsSubmoduleStackOptions | None
    router_layer_controller_options: ExpertsLayerControllerOptions | None
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    expert_attention_use_kv_expert_models_flag: bool


@dataclass(frozen=True, slots=True)
class RuntimeOptions(ResolvedRuntimeOptions):
    def _construction_options(
        self,
        config_module: ModuleType,
    ) -> _ConstructionOptions:
        values = self._values
        return _ConstructionOptions(
            attention_grouping_config=cast(
                GroupingConfig | None, values.get("attention_grouping_config")
            ),
            feed_forward_grouping_config=cast(
                GroupingConfig | None, values.get("feed_forward_grouping_config")
            ),
            batch_size=cast(int, values.get("batch_size", config_module.BATCH_SIZE)),
            learning_rate=cast(
                float,
                values.get("learning_rate", config_module.LEARNING_RATE),
            ),
            input_dim=cast(int, values.get("input_dim", config_module.INPUT_DIM)),
            output_dim=cast(int, values.get("output_dim", config_module.OUTPUT_DIM)),
            sequence_length=cast(
                int,
                values.get("sequence_length", config_module.SEQUENCE_LENGTH),
            ),
            embedding_options=cast(
                GptEmbeddingOptions | None,
                values.get("embedding_options"),
            ),
            lm_head_options=cast(
                GptLmHeadOptions | None,
                values.get("lm_head_options"),
            ),
            decoder_options=cast(
                TransformerDecoderOptions | None,
                values.get("decoder_options"),
            ),
            positional_embedding_options=cast(
                TransformerPositionalEmbeddingOptions | None,
                values.get("positional_embedding_options"),
            ),
            attention_options=cast(
                TransformerAttentionOptions | None,
                values.get("attention_options"),
            ),
            feed_forward_options=cast(
                TransformerFeedForwardOptions | None,
                values.get("feed_forward_options"),
            ),
            attention_projection_stack_options=cast(
                ExpertsSubmoduleStackOptions | None,
                values.get("attention_projection_stack_options"),
            ),
            attention_projection_layer_controller_options=cast(
                ExpertsLayerControllerOptions | None,
                values.get("attention_projection_layer_controller_options"),
            ),
            attention_projection_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions | None,
                values.get("attention_projection_dynamic_memory_options"),
            ),
            attention_projection_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions | None,
                values.get("attention_projection_recurrent_controller_options"),
            ),
            feed_forward_stack_options=cast(
                ExpertsSubmoduleStackOptions | None,
                values.get("feed_forward_stack_options"),
            ),
            feed_forward_layer_controller_options=cast(
                ExpertsLayerControllerOptions | None,
                values.get("feed_forward_layer_controller_options"),
            ),
            feed_forward_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions | None,
                values.get("feed_forward_dynamic_memory_options"),
            ),
            feed_forward_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions | None,
                values.get("feed_forward_recurrent_controller_options"),
            ),
            submodule_stack_options=cast(
                ExpertsSubmoduleStackOptions | None,
                values.get("submodule_stack_options"),
            ),
            layer_controller_options=cast(
                ExpertsLayerControllerOptions | None,
                values.get("layer_controller_options"),
            ),
            dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions | None,
                values.get("dynamic_memory_options"),
            ),
            recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions | None,
                values.get("recurrent_controller_options"),
            ),
            mixture_options=cast(
                ExpertsMixtureOptions | None,
                values.get("mixture_options"),
            ),
            mixture_submodule_stack_options=cast(
                ExpertsSubmoduleStackOptions | None,
                values.get("mixture_submodule_stack_options"),
            ),
            mixture_layer_controller_options=cast(
                ExpertsLayerControllerOptions | None,
                values.get("mixture_layer_controller_options"),
            ),
            mixture_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions | None,
                values.get("mixture_dynamic_memory_options"),
            ),
            mixture_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions | None,
                values.get("mixture_recurrent_controller_options"),
            ),
            expert_stack_options=cast(
                ExpertsSubmoduleStackOptions | None,
                values.get("expert_stack_options"),
            ),
            sampler_options=cast(
                ExpertsSamplerOptions | None,
                values.get("sampler_options"),
            ),
            router_options=cast(
                ExpertsRouterOptions | None,
                values.get("router_options"),
            ),
            router_stack_options=cast(
                ExpertsSubmoduleStackOptions | None,
                values.get("router_stack_options"),
            ),
            router_layer_controller_options=cast(
                ExpertsLayerControllerOptions | None,
                values.get("router_layer_controller_options"),
            ),
            router_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions | None,
                values.get("router_dynamic_memory_options"),
            ),
            router_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions | None,
                values.get("router_recurrent_controller_options"),
            ),
            expert_layer_controller_options=cast(
                ExpertsLayerControllerOptions | None,
                values.get("expert_layer_controller_options"),
            ),
            expert_dynamic_memory_options=cast(
                ExpertsDynamicMemoryOptions | None,
                values.get("expert_dynamic_memory_options"),
            ),
            expert_recurrent_controller_options=cast(
                ExpertsRecurrentControllerOptions | None,
                values.get("expert_recurrent_controller_options"),
            ),
            adaptive_generator_stack_options=cast(
                AdaptiveGeneratorStackOptions | None,
                values.get("adaptive_generator_stack_options"),
            ),
            grouping_config=cast(GroupingConfig | None, values.get("grouping_config")),
            hidden_adaptive_weight_options=cast(
                HiddenAdaptiveWeightOptions | None,
                values.get("hidden_adaptive_weight_options"),
            ),
            hidden_adaptive_bias_options=cast(
                HiddenAdaptiveBiasOptions | None,
                values.get("hidden_adaptive_bias_options"),
            ),
            hidden_adaptive_diagonal_options=cast(
                HiddenAdaptiveDiagonalOptions | None,
                values.get("hidden_adaptive_diagonal_options"),
            ),
            hidden_adaptive_mask_options=cast(
                HiddenAdaptiveMaskOptions | None,
                values.get("hidden_adaptive_mask_options"),
            ),
            router_grouping_config=cast(
                GroupingConfig | None, values.get("router_grouping_config")
            ),
            router_adaptive_weight_options=cast(
                HiddenAdaptiveWeightOptions | None,
                values.get("router_adaptive_weight_options"),
            ),
            router_adaptive_bias_options=cast(
                HiddenAdaptiveBiasOptions | None,
                values.get("router_adaptive_bias_options"),
            ),
            router_adaptive_diagonal_options=cast(
                HiddenAdaptiveDiagonalOptions | None,
                values.get("router_adaptive_diagonal_options"),
            ),
            router_adaptive_mask_options=cast(
                HiddenAdaptiveMaskOptions | None,
                values.get("router_adaptive_mask_options"),
            ),
            expert_attention_use_kv_expert_models_flag=cast(
                bool,
                values.get(
                    "expert_attention_use_kv_expert_models_flag",
                    config_module.EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG,
                ),
            ),
        )

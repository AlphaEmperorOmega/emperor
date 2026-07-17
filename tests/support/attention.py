from emperor.attention import (
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    MultiHeadAttentionConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
)
from emperor.embedding.relative import (
    DynamicPositionalBiasConfig,
    RelativePositionalEmbeddingConfig,
)
from emperor.experts import (
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentLayerConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig

#: The leaf config classes a test can build, in declaration order. Tests that
#: previously iterated ``for attention_option in AttentionOptions`` iterate this
#: tuple instead.
ATTENTION_CONFIG_CLASSES = (
    SelfAttentionConfig,
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
)

RELATIVE_POSITIONAL_EMBEDDING_CASES = (
    ("disabled", None),
    ("dynamic_positional_bias", DynamicPositionalBiasConfig),
)


def make_projection_model_config(
    num_layers: int = 1,
    hidden_dim: int = 16,
    bias_flag: bool = True,
) -> LayerStackConfig:
    """Dimension-free projection stack; input/output dims are injected per
    projection via ``projection_model_config.build(LayerStackConfig(...))``."""
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=bias_flag),
        ),
    )


def make_adaptive_projection_model_config(
    num_layers: int = 1,
    hidden_dim: int = 16,
    bias_flag: bool = True,
) -> LayerStackConfig:
    """Projection stack whose layers are AdaptiveLinearLayers with an all-disabled
    augmentation (a no-op adaptive layer; matches test_linears' adaptive preset)."""
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=AdaptiveLinearLayerConfig(
                bias_flag=bias_flag,
                adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                    weight_config=None,
                    bias_config=None,
                    diagonal_config=None,
                    mask_config=None,
                    model_config=make_projection_model_config(),
                ),
            ),
        ),
    )


def make_recurrent_projection_model_config() -> RecurrentLayerConfig:
    return RecurrentLayerConfig(
        max_steps=2,
        recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
        block_config=make_projection_model_config(),
        gate_config=None,
        residual_connection_option=ResidualConnectionOptions.DISABLED,
        halting_config=None,
        memory_config=None,
    )


def _projection_model_config_for_kind(
    kind: str,
) -> LayerStackConfig | RecurrentLayerConfig:
    if kind == "adaptive":
        return make_adaptive_projection_model_config()
    if kind == "base":
        return make_projection_model_config()
    if kind == "recurrent":
        return make_recurrent_projection_model_config()
    raise ValueError(
        f"Unknown projection kind: {kind!r}. Use 'base', 'adaptive', or 'recurrent'."
    )


def make_router_config(
    input_dim: int,
    num_experts: int,
    bias_flag: bool = False,
    noisy_topk_flag: bool = False,
) -> RouterConfig:
    hidden_dim = max(input_dim, num_experts)
    output_dim = num_experts * 2 if noisy_topk_flag else num_experts
    return RouterConfig(
        input_dim=input_dim,
        num_experts=num_experts,
        noisy_topk_flag=noisy_topk_flag,
        model_config=LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                activation=ActivationOptions.RELU,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=bias_flag),
            ),
        ),
    )


def make_experts_config(
    input_dim: int,
    output_dim: int,
    top_k: int = 3,
    num_experts: int = 6,
    compute_expert_mixture_flag: bool = True,
    routing_initialization_mode: RoutingInitializationMode = (
        RoutingInitializationMode.LAYER
    ),
    weighting_position_option: ExpertWeightingPositionOptions = (
        ExpertWeightingPositionOptions.BEFORE_EXPERTS
    ),
    stack_num_layers: int = 2,
) -> MixtureOfExpertsConfig:
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
        num_experts=num_experts,
        capacity_factor=0.0,
        compute_expert_mixture_flag=compute_expert_mixture_flag,
        weighted_parameters_flag=False,
        weighting_position_option=weighting_position_option,
        routing_initialization_mode=routing_initialization_mode,
        expert_model_config=make_projection_model_config(num_layers=stack_num_layers),
        sampler_config=SamplerConfig(
            top_k=top_k,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
            router_config=make_router_config(input_dim, num_experts),
        ),
    )


def make_mixture_of_experts_model_config(
    input_dim: int,
    output_dim: int,
    top_k: int = 3,
    num_experts: int = 6,
    routing_initialization_mode: RoutingInitializationMode = (
        RoutingInitializationMode.LAYER
    ),
    expert_stack_num_layers: int = 2,
    num_layers: int = 2,
):
    """MixtureOfExpertsModelConfig for use as a feed-forward stack_config. Wraps a
    MixtureOfExpertsConfig leaf (via make_experts_config) in a MixtureOfExpertsLayer
    stack, mirroring test_experts' model_preset without a test-to-test dependency."""
    from emperor.experts import (
        MixtureOfExpertsLayerConfig,
        MixtureOfExpertsModelConfig,
    )

    leaf_config = make_experts_config(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
        num_experts=num_experts,
        routing_initialization_mode=routing_initialization_mode,
        stack_num_layers=expert_stack_num_layers,
    )
    stack_config = LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=MixtureOfExpertsLayerConfig(
            activation=ActivationOptions.RELU,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=leaf_config,
        ),
    )
    return MixtureOfExpertsModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
        routing_initialization_mode=routing_initialization_mode,
        sampler_config=leaf_config.sampler_config,
        stack_config=stack_config,
    )


def make_relative_positional_embedding_config(
    config_cls: type[RelativePositionalEmbeddingConfig] | None,
    num_heads: int,
    embedding_dim: int,
) -> RelativePositionalEmbeddingConfig | None:
    if config_cls is None:
        return None
    return config_cls(
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        num_embeddings=64,
        max_positions=32,
        padding_idx=0,
        init_size=64,
        auto_expand_flag=False,
        text_processing_flag=False,
    )


def build_attention_config(
    config_class: type[MultiHeadAttentionConfig] = IndependentAttentionConfig,
    batch_size: int = 8,
    num_heads: int = 4,
    embedding_dim: int = 12,
    query_key_projection_dim: int = 0,
    value_projection_dim: int = 0,
    target_sequence_length: int = 16,
    source_sequence_length: int = 16,
    dropout_probability: float = 0.0,
    zero_attention_flag: bool = False,
    causal_attention_mask_flag: bool = False,
    add_key_value_bias_flag: bool = False,
    average_attention_weights_flag: bool = False,
    return_attention_weights_flag: bool = False,
    use_kv_expert_models_flag: bool = False,
    projection_kind: str = "base",
    self_attention_projection_strategy: SelfAttentionProjectionStrategy = (
        SelfAttentionProjectionStrategy.FUSED
    ),
    relative_positional_embedding_config_cls: (
        type[RelativePositionalEmbeddingConfig] | None
    ) = None,
    experts_top_k: int = 3,
    experts_num_experts: int = 6,
    experts_compute_expert_mixture_flag: bool = True,
    experts_routing_initialization_mode: RoutingInitializationMode = (
        RoutingInitializationMode.LAYER
    ),
    experts_stack_num_layers: int = 2,
):
    from torch import float32

    relative_positional_embedding_config = make_relative_positional_embedding_config(
        relative_positional_embedding_config_cls,
        num_heads,
        query_key_projection_dim or embedding_dim,
    )
    shared_kwargs = dict(
        batch_size=batch_size,
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        query_key_projection_dim=query_key_projection_dim,
        value_projection_dim=value_projection_dim,
        target_sequence_length=target_sequence_length,
        source_sequence_length=source_sequence_length,
        target_dtype=float32,
        dropout_probability=dropout_probability,
        zero_attention_flag=zero_attention_flag,
        causal_attention_mask_flag=causal_attention_mask_flag,
        add_key_value_bias_flag=add_key_value_bias_flag,
        average_attention_weights_flag=average_attention_weights_flag,
        return_attention_weights_flag=return_attention_weights_flag,
        projection_model_config=_projection_model_config_for_kind(projection_kind),
        relative_positional_embedding_config=relative_positional_embedding_config,
    )

    if config_class is MixtureOfAttentionHeadsConfig:
        return config_class(
            use_kv_expert_models_flag=use_kv_expert_models_flag,
            experts_config=make_experts_config(
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                top_k=experts_top_k,
                num_experts=experts_num_experts,
                compute_expert_mixture_flag=experts_compute_expert_mixture_flag,
                routing_initialization_mode=experts_routing_initialization_mode,
                stack_num_layers=experts_stack_num_layers,
            ),
            **shared_kwargs,
        )

    if config_class is SelfAttentionConfig:
        return config_class(
            projection_strategy=self_attention_projection_strategy,
            **shared_kwargs,
        )

    return config_class(**shared_kwargs)

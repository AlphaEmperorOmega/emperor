"""Shared builders for the attention unit tests.

Replaces the deleted ``MultiHeadAttentionPresets``. ``build_attention_config``
dispatches on the leaf config class to assemble the projection / experts /
positional-embedding sub-configs from the current core APIs (mirroring
``test_linears`` and ``test_experts``). The runtime variant is selected by the
config subclass itself; there is no longer an ``AttentionOptions`` enum.
"""

from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import LinearLayerConfig, AdaptiveLinearLayerConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.sampler.core.config import RouterConfig, SamplerConfig
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core.options import (
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.embedding.options import RelativePositionalEmbeddingOptions
from emperor.embedding.relative.config import RelativePositionalEmbeddingConfig
from emperor.attention.core.config import MultiHeadAttentionConfig
from emperor.attention.self_attention.config import SelfAttentionConfig
from emperor.attention.independent_attention.config import IndependentAttentionConfig
from emperor.attention.mixture_of_attention_heads.config import (
    MixtureOfAttentionHeadsConfig,
)

#: The leaf config classes a test can build, in declaration order. Tests that
#: previously iterated ``for attention_option in AttentionOptions`` iterate this
#: tuple instead.
ATTENTION_CONFIG_CLASSES = (
    SelfAttentionConfig,
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
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
            residual_flag=False,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            shared_halting_flag=False,
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
            residual_flag=False,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            shared_halting_flag=False,
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


def _projection_model_config_for_kind(kind: str) -> LayerStackConfig:
    if kind == "adaptive":
        return make_adaptive_projection_model_config()
    if kind == "base":
        return make_projection_model_config()
    raise ValueError(f"Unknown projection kind: {kind!r}. Use 'base' or 'adaptive'.")


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
                residual_flag=False,
                dropout_probability=0.0,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                shared_halting_flag=False,
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
    routing_initialization_mode: RoutingInitializationMode = RoutingInitializationMode.LAYER,
    weighting_position_option: ExpertWeightingPositionOptions = ExpertWeightingPositionOptions.BEFORE_EXPERTS,
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


def make_relative_positional_embedding_config(
    positional_embedding_option: RelativePositionalEmbeddingOptions,
    num_heads: int,
    embedding_dim: int,
) -> RelativePositionalEmbeddingConfig | None:
    if positional_embedding_option == RelativePositionalEmbeddingOptions.DISABLED:
        return None
    return RelativePositionalEmbeddingConfig(
        positional_embedding_option=positional_embedding_option,
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
    positional_embedding_option: RelativePositionalEmbeddingOptions = RelativePositionalEmbeddingOptions.DISABLED,
    experts_top_k: int = 3,
    experts_num_experts: int = 6,
    experts_compute_expert_mixture_flag: bool = True,
    experts_routing_initialization_mode: RoutingInitializationMode = RoutingInitializationMode.LAYER,
    experts_stack_num_layers: int = 2,
):
    from torch import float32

    relative_positional_embedding_config = make_relative_positional_embedding_config(
        positional_embedding_option, num_heads, embedding_dim
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

    return config_class(**shared_kwargs)

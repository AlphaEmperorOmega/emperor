"""Native configuration fixtures for hierarchical byte embedding tests."""

import torch

from emperor.attention import SelfAttentionConfig, SelfAttentionProjectionStrategy
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdditiveDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig
from emperor.transformer import (
    FeedForwardConfig,
    TransformerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)


def linear_stack(input_dim=8, output_dim=8, *, hidden_dim=16, num_layers=1):
    return LayerStackConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def adaptive_stack(input_dim=8, output_dim=8):
    config = linear_stack(input_dim, output_dim)
    config.layer_config.layer_model_config = AdaptiveLinearLayerConfig(
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            bias_config=AdditiveDynamicBiasConfig(
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
                model_config=linear_stack(None, None),
            )
        ),
    )
    return config


def moe_config(input_dim=8, output_dim=8, *, adaptive=True, capacity=0.0):
    stack = linear_stack(input_dim, output_dim)
    stack.layer_config = MixtureOfExpertsLayerConfig(
        activation=ActivationOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        layer_model_config=MixtureOfExpertsConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            top_k=2,
            num_experts=3,
            capacity_factor=capacity,
            dropped_token_behavior=DroppedTokenOptions.ZEROS,
            compute_expert_mixture_flag=True,
            weighted_parameters_flag=True,
            weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
            routing_initialization_mode=RoutingInitializationMode.LAYER,
            sampler_config=SamplerConfig(
                top_k=2,
                threshold=0.0,
                filter_above_threshold=False,
                num_topk_samples=0,
                normalize_probabilities_flag=True,
                noisy_topk_flag=False,
                num_experts=3,
                coefficient_of_variation_loss_weight=0.1,
                switch_loss_weight=0.1,
                zero_centred_loss_weight=0.0,
                mutual_information_loss_weight=0.0,
                router_config=RouterConfig(
                    input_dim=input_dim,
                    num_experts=3,
                    noisy_topk_flag=False,
                    model_config=linear_stack(input_dim, 3),
                ),
            ),
            expert_model_config=(
                adaptive_stack(None, None) if adaptive else linear_stack(None, None)
            ),
        ),
    )
    return MixtureOfExpertsModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=2,
        routing_initialization_mode=RoutingInitializationMode.LAYER,
        stack_config=stack,
    )


def encoder_layer_config(dim=8, *, max_batch=32, max_length=33, causal=False):
    return TransformerEncoderLayerConfig(
        embedding_dim=dim,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        dropout_probability=0.0,
        residual_config=AdditiveResidualConfig(),
        attention_config=SelfAttentionConfig(
            batch_size=max_batch,
            num_heads=2,
            embedding_dim=dim,
            query_key_projection_dim=0,
            value_projection_dim=0,
            target_sequence_length=max_length,
            source_sequence_length=max_length,
            target_dtype=torch.float32,
            dropout_probability=0.0,
            zero_attention_flag=False,
            causal_attention_mask_flag=causal,
            add_key_value_bias_flag=False,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            batch_first_flag=True,
            projection_model_config=linear_stack(dim, dim),
            projection_strategy=SelfAttentionProjectionStrategy.FUSED,
        ),
        feed_forward_config=FeedForwardConfig(
            input_dim=dim,
            output_dim=dim,
            stack_config=linear_stack(dim, dim, hidden_dim=dim * 2),
        ),
    )


def embedding_config(*, dim=8, output_dim=6, max_bytes=32, max_batch=32):
    layer_config = encoder_layer_config(
        dim, max_batch=max_batch, max_length=max_bytes + 1
    )
    return HierarchicalByteEmbeddingConfig(
        byte_embedding_dim=dim,
        output_dim=output_dim,
        max_token_bytes=max_bytes,
        byte_position_config=TextLearnedPositionalEmbeddingConfig(
            num_embeddings=max_bytes + 1, embedding_dim=dim
        ),
        encoder_config=TransformerConfig(
            encoder_stack_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=dim,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_postprocessing_flag=True,
                layer_config=TransformerEncoderBlockLayerConfig(
                    activation=ActivationOptions.DISABLED,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    layer_model_config=layer_config,
                ),
            )
        ),
        projection_config=linear_stack(dim, output_dim),
    )

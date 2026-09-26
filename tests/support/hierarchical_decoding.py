"""Small native decoder compositions for behavioral tests."""

from emperor.decoding.hierarchical import HierarchicalByteDecoderConfig
from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.transformer import (
    TransformerConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
)
from tests.support.hierarchical_embedding import encoder_layer_config, linear_stack


def decoder_config(*, dim=8, conditioning_dim=6, max_bytes=8, max_batch=16):
    source = encoder_layer_config(
        dim, max_batch=max_batch, max_length=max_bytes + 1, causal=True
    )
    layer = TransformerDecoderLayerConfig(
        embedding_dim=dim,
        layer_norm_position=source.layer_norm_position,
        dropout_probability=0.0,
        residual_config=source.residual_config,
        self_attention_config=source.attention_config,
        feed_forward_config=source.feed_forward_config,
    )
    return HierarchicalByteDecoderConfig(
        conditioning_dim=conditioning_dim,
        byte_embedding_dim=dim,
        max_token_bytes=max_bytes,
        byte_position_config=TextLearnedPositionalEmbeddingConfig(
            num_embeddings=max_bytes + 1, embedding_dim=dim
        ),
        conditioning_projection_config=linear_stack(conditioning_dim, dim),
        output_projection_config=linear_stack(dim, 258),
        decoder_config=TransformerConfig(
            decoder_stack_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=dim,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_postprocessing_flag=True,
                layer_config=TransformerDecoderBlockLayerConfig(
                    activation=ActivationOptions.DISABLED,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    layer_model_config=layer,
                ),
            )
        ),
    )

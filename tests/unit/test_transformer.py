import unittest

import torch

from emperor.attention import (
    AttentionLayerState,
    IndependentAttentionConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    RecurrentLayer,
    RecurrentLayerConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.transformer import (
    FeedForwardConfig,
    Transformer,
    TransformerConfig,
    TransformerDecoderBlockLayer,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayer,
    TransformerDecoderLayerConfig,
    TransformerEncoderBlockLayer,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayer,
    TransformerEncoderLayerConfig,
)


def linear_stack(
    input_dim: int,
    output_dim: int,
    *,
    hidden_dim: int | None = None,
    num_layers: int = 1,
    activation: ActivationOptions = ActivationOptions.DISABLED,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=input_dim if hidden_dim is None else hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            activation=activation,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def self_attention(
    embedding_dim: int,
    *,
    causal: bool,
    maximum_length: int = 8,
) -> SelfAttentionConfig:
    return SelfAttentionConfig(
        batch_size=4,
        num_heads=2,
        embedding_dim=embedding_dim,
        query_key_projection_dim=embedding_dim,
        value_projection_dim=embedding_dim,
        target_sequence_length=maximum_length,
        source_sequence_length=maximum_length,
        target_dtype=torch.float32,
        dropout_probability=0.0,
        zero_attention_flag=False,
        causal_attention_mask_flag=causal,
        add_key_value_bias_flag=False,
        average_attention_weights_flag=True,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=linear_stack(embedding_dim, embedding_dim),
        relative_positional_embedding_config=None,
        projection_strategy=SelfAttentionProjectionStrategy.FUSED,
    )


def cross_attention(
    embedding_dim: int,
    *,
    maximum_target_length: int = 8,
    maximum_source_length: int = 8,
) -> IndependentAttentionConfig:
    return IndependentAttentionConfig(
        batch_size=4,
        num_heads=2,
        embedding_dim=embedding_dim,
        query_key_projection_dim=embedding_dim,
        value_projection_dim=embedding_dim,
        target_sequence_length=maximum_target_length,
        source_sequence_length=maximum_source_length,
        target_dtype=torch.float32,
        dropout_probability=0.0,
        zero_attention_flag=False,
        causal_attention_mask_flag=False,
        add_key_value_bias_flag=False,
        average_attention_weights_flag=True,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=linear_stack(embedding_dim, embedding_dim),
        relative_positional_embedding_config=None,
    )


def feed_forward(embedding_dim: int) -> FeedForwardConfig:
    return FeedForwardConfig(
        input_dim=embedding_dim,
        output_dim=embedding_dim,
        stack_config=linear_stack(
            embedding_dim,
            embedding_dim,
            hidden_dim=embedding_dim * 2,
            num_layers=1,
            activation=ActivationOptions.RELU,
        ),
    )


def encoder_stack(
    embedding_dim: int = 8,
    *,
    num_layers: int = 2,
) -> LayerStackConfig:
    transformer_layer = TransformerEncoderLayerConfig(
        embedding_dim=embedding_dim,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        dropout_probability=0.0,
        residual_config=ResidualConfig(
            option=ResidualConnectionOptions.RESIDUAL,
            model_config=None,
        ),
        attention_config=self_attention(embedding_dim, causal=False),
        feed_forward_config=feed_forward(embedding_dim),
    )
    return LayerStackConfig(
        input_dim=embedding_dim,
        hidden_dim=embedding_dim,
        output_dim=embedding_dim,
        num_layers=num_layers,
        apply_output_pipeline_flag=True,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=TransformerEncoderBlockLayerConfig(
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=transformer_layer,
        ),
    )


def decoder_stack(
    embedding_dim: int = 8,
    *,
    num_layers: int = 2,
    include_cross_attention: bool = True,
) -> LayerStackConfig:
    transformer_layer = TransformerDecoderLayerConfig(
        embedding_dim=embedding_dim,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        dropout_probability=0.0,
        residual_config=ResidualConfig(
            option=ResidualConnectionOptions.RESIDUAL,
            model_config=None,
        ),
        self_attention_config=self_attention(embedding_dim, causal=True),
        cross_attention_config=(
            cross_attention(embedding_dim) if include_cross_attention else None
        ),
        feed_forward_config=feed_forward(embedding_dim),
    )
    return LayerStackConfig(
        input_dim=embedding_dim,
        hidden_dim=embedding_dim,
        output_dim=embedding_dim,
        num_layers=num_layers,
        apply_output_pipeline_flag=True,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=TransformerDecoderBlockLayerConfig(
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=transformer_layer,
        ),
    )


def recurrent(stack_config: LayerStackConfig) -> RecurrentLayerConfig:
    return RecurrentLayerConfig(
        input_dim=stack_config.input_dim,
        output_dim=stack_config.output_dim,
        max_steps=2,
        recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
        block_config=stack_config,
        gate_config=None,
        residual_config=None,
        halting_config=None,
        memory_config=None,
    )


class TestTransformerGenericStacks(unittest.TestCase):
    def test_stacks_wrap_layer_instances(self):
        model = Transformer(
            TransformerConfig(
                encoder_stack_config=encoder_stack(),
                decoder_stack_config=decoder_stack(),
            )
        )

        self.assertIsInstance(model.encoder_model, LayerStack)
        self.assertIsInstance(model.decoder_model, LayerStack)
        self.assertTrue(all(isinstance(layer, Layer) for layer in model.encoder_model))
        self.assertTrue(all(isinstance(layer, Layer) for layer in model.decoder_model))
        self.assertTrue(
            all(
                isinstance(layer, TransformerEncoderBlockLayer)
                for layer in model.encoder_model
            )
        )
        self.assertTrue(
            all(
                isinstance(layer, TransformerDecoderBlockLayer)
                for layer in model.decoder_model
            )
        )
        self.assertTrue(
            all(
                isinstance(layer.model, TransformerEncoderLayer)
                for layer in model.encoder_model
            )
        )
        self.assertTrue(
            all(
                isinstance(layer.model, TransformerDecoderLayer)
                for layer in model.decoder_model
            )
        )
        encoder_layer = model.encoder_model[0].model
        decoder_layer = model.decoder_model[0].model
        self.assertIsInstance(encoder_layer.self_attention_layer, Layer)
        self.assertIsInstance(encoder_layer.feed_forward_layer, Layer)
        self.assertIsInstance(decoder_layer.self_attention_layer, Layer)
        self.assertIsInstance(decoder_layer.cross_attention_layer, Layer)
        self.assertIsInstance(decoder_layer.feed_forward_layer, Layer)

    def test_full_transformer_preserves_runtime_lengths_masks_and_gradients(self):
        torch.manual_seed(7)
        model = Transformer(
            TransformerConfig(
                encoder_stack_config=encoder_stack(),
                decoder_stack_config=decoder_stack(),
            )
        )
        source = torch.randn(2, 5, 8, requires_grad=True)
        target = torch.randn(2, 3, 8, requires_grad=True)
        source_padding = torch.tensor(
            [[False, False, False, True, True], [False, False, False, False, True]]
        )
        target_padding = torch.tensor([[False, False, True], [False, False, False]])
        source_mask = torch.zeros(5, 5, dtype=torch.bool)
        target_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        cross_mask = torch.zeros(3, 5, dtype=torch.bool)

        output, loss = model(
            source_token_embeddings=source,
            target_token_embeddings=target,
            source_attention_mask=source_mask,
            target_attention_mask=target_mask,
            encoder_attention_mask=cross_mask,
            source_key_padding_mask=source_padding,
            target_key_padding_mask=target_padding,
            encoder_key_padding_mask=source_padding,
        )

        self.assertEqual(output.shape, (2, 3, 8))
        self.assertEqual(loss.shape, ())
        self.assertEqual(output.dtype, target.dtype)
        self.assertTrue(torch.isfinite(output).all())
        (output.square().mean() + loss).backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(target.grad)
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )

    def test_encoder_only_and_decoder_only(self):
        encoder_model = Transformer(
            TransformerConfig(
                encoder_stack_config=encoder_stack(num_layers=1),
                decoder_stack_config=None,
            )
        )
        decoder_model = Transformer(
            TransformerConfig(
                encoder_stack_config=None,
                decoder_stack_config=decoder_stack(
                    num_layers=1,
                    include_cross_attention=False,
                ),
            )
        )

        encoder_output, encoder_loss = encoder_model(
            source_token_embeddings=torch.randn(2, 4, 8)
        )
        decoder_output, decoder_loss = decoder_model(
            target_token_embeddings=torch.randn(2, 4, 8)
        )

        self.assertEqual(encoder_output.shape, (2, 4, 8))
        self.assertEqual(decoder_output.shape, (2, 4, 8))
        self.assertEqual(encoder_loss.shape, ())
        self.assertEqual(decoder_loss.shape, ())

    def test_decoder_only_rejects_cross_attention(self):
        with self.assertRaisesRegex(ValueError, "cross_attention_config=None"):
            Transformer(
                TransformerConfig(
                    encoder_stack_config=None,
                    decoder_stack_config=decoder_stack(include_cross_attention=True),
                )
            )

    def test_recurrent_generic_stacks_preserve_transformer_state(self):
        model = Transformer(
            TransformerConfig(
                encoder_stack_config=recurrent(encoder_stack(num_layers=1)),
                decoder_stack_config=recurrent(decoder_stack(num_layers=1)),
            )
        )
        self.assertIsInstance(model.encoder_model, RecurrentLayer)
        self.assertIsInstance(model.decoder_model, RecurrentLayer)

        output, loss = model(
            source_token_embeddings=torch.randn(2, 5, 8),
            target_token_embeddings=torch.randn(2, 3, 8),
            source_key_padding_mask=torch.zeros(2, 5, dtype=torch.bool),
            encoder_key_padding_mask=torch.zeros(2, 5, dtype=torch.bool),
        )

        self.assertEqual(output.shape, (2, 3, 8))
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(output).all())

    def test_final_normalization_matches_generic_stack_output(self):
        stack_config = encoder_stack(num_layers=1)
        model = Transformer(
            TransformerConfig(
                encoder_stack_config=stack_config,
                decoder_stack_config=None,
            )
        ).eval()
        stack = stack_config.build().eval()
        stack.load_state_dict(model.encoder_model.state_dict())
        source = torch.randn(2, 4, 8)

        with torch.no_grad():
            state = stack(AttentionLayerState(hidden=source))
            expected = model.encoder_layer_norm(state.hidden)
            actual, _ = model(source_token_embeddings=source)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from emperor.attention import AttentionLayerState
from emperor.layers import Layer, LayerStack, RecurrentLayer
from emperor.transformer import (
    TransformerDecoderLayer,
    TransformerDecoderLayerState,
    TransformerEncoderLayer,
)
from unit.test_transformer import decoder_stack, encoder_stack, recurrent


class TestTransformerLayerComposition(unittest.TestCase):
    def test_encoder_forward_is_composed_from_layer_wrappers(self):
        config = encoder_stack(num_layers=1).layer_config.layer_model_config
        model = TransformerEncoderLayer(config).eval()
        source = torch.randn(2, 4, config.embedding_dim)
        padding_mask = torch.tensor(
            [[False, False, True, True], [False, False, False, True]]
        )
        attention_mask = torch.zeros(4, 4, dtype=torch.bool)

        manual_state = AttentionLayerState(
            hidden=source.clone(),
            key_padding_mask=padding_mask,
            attention_mask=attention_mask,
        )
        with torch.no_grad():
            manual_state = model.self_attention_layer(manual_state)
            manual_state = model.feed_forward_layer(manual_state)
            manual_loss = (
                manual_state.loss
                if manual_state.loss is not None
                else source.new_zeros(())
            )
            actual, actual_loss = model(
                source,
                source_key_padding_mask=padding_mask,
                attention_mask=attention_mask,
            )

        self.assertIsInstance(model.self_attention_layer, Layer)
        self.assertIsInstance(model.feed_forward_layer, Layer)
        torch.testing.assert_close(actual, manual_state.hidden)
        torch.testing.assert_close(actual_loss, manual_loss)

    def test_decoder_forward_is_composed_from_layer_wrappers(self):
        config = decoder_stack(num_layers=1).layer_config.layer_model_config
        model = TransformerDecoderLayer(config).eval()
        target = torch.randn(2, 3, config.embedding_dim)
        encoder_output = torch.randn(2, 5, config.embedding_dim)
        target_padding_mask = torch.tensor(
            [[False, False, True], [False, False, False]]
        )
        encoder_padding_mask = torch.tensor(
            [[False, False, False, True, True], [False, False, False, False, True]]
        )
        target_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        cross_mask = torch.zeros(3, 5, dtype=torch.bool)

        manual_state = TransformerDecoderLayerState(
            hidden=target.clone(),
            target_key_padding_mask=target_padding_mask,
            target_attention_mask=target_mask,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            cross_attention_mask=cross_mask,
        )
        with torch.no_grad():
            manual_state = model.self_attention_layer(manual_state)
            manual_state = model.cross_attention_layer(manual_state)
            manual_state = model.feed_forward_layer(manual_state)
            manual_loss = (
                manual_state.loss
                if manual_state.loss is not None
                else target.new_zeros(())
            )
            actual, actual_loss = model(
                target,
                encoder_output=encoder_output,
                key_padding_mask=target_padding_mask,
                encoder_padding_mask=encoder_padding_mask,
                attention_mask=target_mask,
                encoder_attention_mask=cross_mask,
            )

        self.assertIsInstance(model.self_attention_layer, Layer)
        self.assertIsInstance(model.cross_attention_layer, Layer)
        self.assertIsInstance(model.feed_forward_layer, Layer)
        torch.testing.assert_close(actual, manual_state.hidden)
        torch.testing.assert_close(actual_loss, manual_loss)

    def test_generic_stack_and_recurrent_components_own_transformer_blocks(self):
        stack = encoder_stack().build()
        recurrent_model = recurrent(encoder_stack(num_layers=1)).build()

        self.assertIsInstance(stack, LayerStack)
        self.assertIsInstance(recurrent_model, RecurrentLayer)
        self.assertTrue(all(isinstance(layer, Layer) for layer in stack))
        self.assertIsInstance(recurrent_model.block_model, LayerStack)
        self.assertTrue(
            all(isinstance(layer, Layer) for layer in recurrent_model.block_model)
        )


if __name__ == "__main__":
    unittest.main()

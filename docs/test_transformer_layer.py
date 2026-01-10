import torch
import itertools
import unittest

from Emperor.attention.utils.layer import MultiHeadAttention
from Emperor.transformer.utils.feed_forward import FeedForward
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.transformer.utils.layers import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


class TestTransformerEncoderLayer(unittest.TestCase):
    def test_init(self):
        c = TransformerPresets.transformer_preset()
        m = TransformerEncoderLayer(c)

        self.assertIsInstance(m.self_attention_model, MultiHeadAttention)
        self.assertIsInstance(m.feed_forward_model, FeedForward)

    def test_forward_with_different_inputs(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        c = TransformerPresets.transformer_preset(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            attention_batch_size=batch_size,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = TransformerEncoderLayer(c)

        soruce_token_embeddings = torch.randn(
            source_sequence_length,
            batch_size,
            embedding_dim,
        )
        key_padding_mask_options = (
            None,
            torch.randn(batch_size, source_sequence_length),
        )
        attention_mask_options = (
            None,
            torch.randn(
                batch_size * num_heads,
                source_sequence_length,
                target_sequence_length,
            ),
        )

        for (
            key_padding_mask,
            attention_mask,
        ) in itertools.product(
            key_padding_mask_options,
            attention_mask_options,
        ):
            parts = (
                f"key_padding_mask: {key_padding_mask.shape if key_padding_mask is not None else None}",
                f"attention_mask: {attention_mask.shape if attention_mask is not None else None}",
            )
            message = f"Test failed for the inputs: ".join(parts)
            with self.subTest(i=message):
                output = m(
                    source_token_embeddings=soruce_token_embeddings,
                    attention_mask=attention_mask,
                    source_key_padding_mask=key_padding_mask,
                )

                expected_output_shape = (
                    source_sequence_length,
                    batch_size,
                    embedding_dim,
                )

                if isinstance(output, tuple):
                    output, _ = output

                self.assertEqual(output.shape, expected_output_shape)


class TestTransformerDecoderLayer(unittest.TestCase):
    def test_init(self):
        c = TransformerPresets.transformer_preset()
        self.model = TransformerDecoderLayer(c)

        self.assertIsInstance(self.model.self_attention_model, MultiHeadAttention)
        self.assertIsInstance(self.model.cross_attention_model, MultiHeadAttention)
        self.assertIsInstance(self.model.feed_forward_model, FeedForward)

    def test_forward_width_different_inputs(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        c = TransformerPresets.transformer_preset(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            attention_batch_size=batch_size,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = TransformerDecoderLayer(c)
        target_token_embeddings = torch.randn(
            target_sequence_length,
            batch_size,
            embedding_dim,
        )
        encoder_output = torch.randn(
            source_sequence_length,
            batch_size,
            embedding_dim,
        )

        key_padding_mask_options = (
            None,
            torch.randn(batch_size, target_sequence_length),
        )
        encoder_padding_mask_options = (
            None,
            torch.randn(batch_size, source_sequence_length),
        )
        attention_mask_options = (
            None,
            torch.randn(
                batch_size * num_heads,
                target_sequence_length,
                target_sequence_length,
            ),
        )
        encoder_attention_mask_options = (
            None,
            torch.randn(
                batch_size * num_heads,
                source_sequence_length,
                target_sequence_length,
            ),
        )

        for (
            key_padding_mask,
            encoder_padding_mask,
            attention_mask,
            encoder_attention_mask,
        ) in itertools.product(
            key_padding_mask_options,
            encoder_padding_mask_options,
            attention_mask_options,
            encoder_attention_mask_options,
        ):
            parts = (
                f"key_padding_mask: {key_padding_mask}",
                f"encoder_padding_mask: {encoder_padding_mask}",
                f"attention_mask: {attention_mask}",
                f"encoder_attention_mask: {encoder_attention_mask}",
            )
            message = f"Test failed for the inputs: ".join(parts)
            with self.subTest(i=message):
                output = m(
                    target_token_embeddings=target_token_embeddings,
                    encoder_output=encoder_output,
                    key_padding_mask=key_padding_mask,
                    encoder_padding_mask=encoder_padding_mask,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                expected_output = (
                    target_sequence_length,
                    batch_size,
                    embedding_dim,
                )

                if isinstance(output, tuple):
                    output, _ = output

                self.assertEqual(output.shape, expected_output)

import torch
import unittest
import itertools

from torch.nn import LayerNorm, ModuleList
from Emperor.transformer.presets import TransformerPresets
from Emperor.transformer.stack import (
    Transformer,
    TransformerDecoder,
    TransformerEncoder,
)


class TestTransformerEncoder(unittest.TestCase):
    def test_init(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        num_layers = 2
        layer_norm_dim = 10

        c = TransformerPresets.transformer_preset(
            num_layers=num_layers,
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            layer_norm_dim=layer_norm_dim,
            attention_batch_size=batch_size,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = TransformerEncoder(c)

        self.assertEqual(m.num_layers, num_layers)
        self.assertEqual(m.source_sequence_length, source_sequence_length)
        self.assertEqual(m.target_sequence_length, target_sequence_length)
        self.assertEqual(m.layer_norm_dim, layer_norm_dim)
        self.assertIsInstance(m.layer_norm_module, LayerNorm)
        self.assertIsInstance(m.layers, ModuleList)
        self.assertEqual(len(m.layers), num_layers)

    def test_forward(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        num_layers = 2
        layer_norm_dim = 10

        c = TransformerPresets.transformer_preset(
            num_layers=num_layers,
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            layer_norm_dim=layer_norm_dim,
            attention_batch_size=batch_size,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = TransformerEncoder(c)

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

                expected_output = (source_sequence_length, batch_size, embedding_dim)
                output, loss = output

                self.assertEqual(output.shape, expected_output)
                self.assertIsInstance(loss, torch.Tensor)


class TestTransformerDecoder(unittest.TestCase):
    def test_init(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        num_layers = 2
        layer_norm_dim = 10

        c = TransformerPresets.transformer_preset(
            num_layers=num_layers,
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            layer_norm_dim=layer_norm_dim,
            attention_batch_size=batch_size,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = TransformerDecoder(c)

        self.assertEqual(m.num_layers, num_layers)
        self.assertEqual(m.source_sequence_length, source_sequence_length)
        self.assertEqual(m.target_sequence_length, target_sequence_length)
        self.assertEqual(m.layer_norm_dim, layer_norm_dim)
        self.assertIsInstance(m.layer_norm_module, LayerNorm)
        self.assertIsInstance(m.layers, ModuleList)
        self.assertEqual(len(m.layers), num_layers)

    def test_forward(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        num_layers = 2
        layer_norm_dim = 10

        c = TransformerPresets.transformer_preset(
            num_layers=num_layers,
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            layer_norm_dim=layer_norm_dim,
            attention_batch_size=batch_size,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = TransformerDecoder(c)
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
                    target_key_padding_mask=key_padding_mask,
                    encoder_key_padding_mask=encoder_padding_mask,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )

                expected_output = (
                    source_sequence_length,
                    batch_size,
                    embedding_dim,
                )

                output, loss = output

                self.assertEqual(output.shape, expected_output)
                self.assertIsInstance(loss, torch.Tensor)


class TestTransformer(unittest.TestCase):
    def test_init(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        num_layers = 2
        layer_norm_dim = 10

        c = TransformerPresets.transformer_preset(
            num_layers=num_layers,
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            layer_norm_dim=layer_norm_dim,
            attention_batch_size=batch_size,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = Transformer(c)

        self.assertIsInstance(m.encoder_model, TransformerEncoder)
        self.assertIsInstance(m.decoder_model, TransformerDecoder)

    def test_all_possible_inputs(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        num_layers = 2
        layer_norm_dim = 10

        c = TransformerPresets.transformer_preset(
            num_layers=num_layers,
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            layer_norm_dim=layer_norm_dim,
            attention_batch_size=batch_size,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = Transformer(c)

        source_token_embeddings = torch.randn(
            source_sequence_length,
            batch_size,
            embedding_dim,
        )
        target_token_embeddings = torch.randn(
            target_sequence_length,
            batch_size,
            embedding_dim,
        )

        source_key_padding_mask_options = (
            None,
            torch.randn(batch_size, source_sequence_length),
        )
        source_attention_mask_options = (
            None,
            torch.randn(
                batch_size * num_heads,
                source_sequence_length,
                target_sequence_length,
            ),
        )
        target_key_padding_mask_options = (
            None,
            torch.randn(batch_size, target_sequence_length),
        )
        memory_padding_mask_options = (
            None,
            torch.randn(batch_size, source_sequence_length),
        )
        target_attention_mask_options = (
            None,
            torch.randn(
                batch_size * num_heads,
                target_sequence_length,
                target_sequence_length,
            ),
        )
        memory_attention_mask_options = (
            None,
            torch.randn(
                batch_size * num_heads,
                source_sequence_length,
                target_sequence_length,
            ),
        )

        for (
            source_key_padding_mask,
            source_attention_mask,
            target_key_padding_mask,
            memory_key_padding_mask,
            target_attention_mask,
            memory_attention_mask,
        ) in itertools.product(
            source_key_padding_mask_options,
            source_attention_mask_options,
            target_key_padding_mask_options,
            memory_padding_mask_options,
            target_attention_mask_options,
            memory_attention_mask_options,
        ):
            parts = (
                f"source_key_padding_mask: {source_key_padding_mask.shape if source_key_padding_mask is not None else None}",
                f"source_attention_mask: {source_attention_mask.shape if source_attention_mask is not None else None}",
                f"target_key_padding_mask: {target_key_padding_mask.shape if target_key_padding_mask is not None else None}",
                f"memory_key_padding_mask: {memory_key_padding_mask.shape if memory_key_padding_mask is not None else None}",
                f"target_attention_mask: {target_attention_mask.shape if target_attention_mask is not None else None}",
                f"memory_attention_mask: {memory_attention_mask.shape if memory_attention_mask is not None else None}",
            )
            message = f"Test failed for the inputs: ".join(parts)
            with self.subTest(i=message):
                output, loss = m(
                    source_token_embeddings=source_token_embeddings,
                    target_token_embeddings=target_token_embeddings,
                    source_attention_mask=source_attention_mask,
                    target_attention_mask=target_attention_mask,
                    memory_attention_mask=memory_attention_mask,
                    source_key_padding_mask=source_key_padding_mask,
                    target_key_padding_mask=target_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

                expected_output = (source_sequence_length, batch_size, embedding_dim)

                self.assertEqual(output.shape, expected_output)
                self.assertIsInstance(loss, torch.Tensor)

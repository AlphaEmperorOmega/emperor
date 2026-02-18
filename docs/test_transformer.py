import torch
import unittest
import itertools

from torch.nn import LayerNorm, ModuleList
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.transformer.utils.models import Transformer
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.transformer.utils.stack import (
    TransformerDecoderStack,
    TransformerEncoderStack,
)


def create_key_padding_mask(
    batch_size: int, source_sequence_length: int
) -> torch.Tensor:
    key_padding_mask_shape = (
        batch_size,
        source_sequence_length,
    )
    key_padding_mask = torch.randint(0, 2, key_padding_mask_shape)
    key_padding_mask = torch.where(
        key_padding_mask > 0,
        torch.tensor(float("-inf")),
        torch.tensor(0.0),
    )
    return key_padding_mask


def create_attention_mask(
    target_sequence_length: int,
    source_sequence_length: int,
    attention_mask_repeat: int = 1,
) -> torch.Tensor:
    attention_mask = torch.triu(
        torch.full((target_sequence_length, source_sequence_length), float("-inf")),
        diagonal=1,
    )
    return attention_mask.unsqueeze(0).repeat(attention_mask_repeat, 1, 1)


class TestTransformerEncoderStack(unittest.TestCase):
    def test_init(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        num_layers = 2

        c = TransformerPresets.transformer_preset(
            batch_size=batch_size,
            num_layers=num_layers,
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = TransformerEncoderStack(c)

        self.assertEqual(m.num_layers, num_layers)
        self.assertEqual(m.source_sequence_length, source_sequence_length)
        self.assertEqual(m.target_sequence_length, target_sequence_length)
        self.assertIsInstance(m.layer_norm_module, LayerNorm)
        self.assertIsInstance(m.layers, ModuleList)
        self.assertEqual(len(m.layers), num_layers)

    def test_forward(self):
        batch_size = 4
        num_heads = 2
        embedding_dim = 10
        num_layers = 2
        sequence_lengths = [6, 10]
        qkv_dimensions = [0, 16, 20]
        model_types = list(LinearLayerStackOptions)
        bool_options = [True, False]

        for model_type in model_types:
            for sequence_length in sequence_lengths:
                for query_key_projection_dim in qkv_dimensions:
                    for value_projection_dim in qkv_dimensions:
                        for add_key_value_bias_flag in bool_options:
                            for zero_attention_flag in bool_options:
                                for average_attention_weights_flag in bool_options:
                                    c = TransformerPresets.transformer_preset(
                                        batch_size=batch_size,
                                        num_layers=num_layers,
                                        input_dim=embedding_dim,
                                        hidden_dim=embedding_dim,
                                        output_dim=embedding_dim,
                                        attention_num_heads=num_heads,
                                        embedding_dim=embedding_dim,
                                        attention_model_type=model_type,
                                        attention_target_sequence_length=sequence_length,
                                        attention_source_sequence_length=sequence_length,
                                        attention_query_key_projection_dim=query_key_projection_dim,
                                        attention_value_projection_dim=value_projection_dim,
                                        attention_add_key_value_bias_flag=add_key_value_bias_flag,
                                        attention_zero_attention_flag=zero_attention_flag,
                                        attention_average_attention_weights_flag=average_attention_weights_flag,
                                    )
                                    m = TransformerEncoderStack(c)
                                    source_token_embeddings = torch.randn(
                                        sequence_length,
                                        batch_size,
                                        embedding_dim,
                                    )
                                    key_padding_mask_options = (
                                        None,
                                        create_key_padding_mask(
                                            batch_size, sequence_length
                                        ),
                                    )
                                    attention_mask_options = (
                                        None,
                                        create_attention_mask(
                                            sequence_length,
                                            sequence_length,
                                            batch_size * num_heads,
                                        ),
                                    )

                                    for (
                                        key_padding_mask,
                                        attention_mask,
                                    ) in itertools.product(
                                        key_padding_mask_options,
                                        attention_mask_options,
                                    ):
                                        message = f"Test failed for the inputs: model_type={model_type}, sequence_length={sequence_length}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, add_key_value_bias_flag={add_key_value_bias_flag}, zero_attention_flag={zero_attention_flag}, average_attention_weights_flag={average_attention_weights_flag}, key_padding_mask={key_padding_mask.shape if key_padding_mask is not None else None}, attention_mask={attention_mask.shape if attention_mask is not None else None}"
                                        with self.subTest(i=message):
                                            output = m(
                                                source_token_embeddings=source_token_embeddings,
                                                attention_mask=attention_mask,
                                                source_key_padding_mask=key_padding_mask,
                                            )

                                            expected_output = (
                                                sequence_length,
                                                batch_size,
                                                embedding_dim,
                                            )
                                            output, loss = output

                                            self.assertEqual(
                                                output.shape, expected_output
                                            )
                                            self.assertIsInstance(loss, torch.Tensor)


class TestTransformerDecoderStack(unittest.TestCase):
    def test_init(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        num_layers = 2

        c = TransformerPresets.transformer_preset(
            batch_size=batch_size,
            num_layers=num_layers,
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = TransformerDecoderStack(c)

        self.assertEqual(m.num_layers, num_layers)
        self.assertEqual(m.source_sequence_length, source_sequence_length)
        self.assertEqual(m.target_sequence_length, target_sequence_length)
        self.assertIsInstance(m.layer_norm_module, LayerNorm)
        self.assertIsInstance(m.layers, ModuleList)
        self.assertEqual(len(m.layers), num_layers)

    def test_forward(self):
        batch_size = 4
        num_heads = 2
        embedding_dim = 10
        num_layers = 2
        sequence_lengths = [6, 10]
        qkv_dimensions = [0, 16, 20]
        model_types = list(LinearLayerStackOptions)
        bool_options = [True, False]

        for model_type in model_types:
            for target_sequence_length in sequence_lengths:
                for source_sequence_length in sequence_lengths:
                    for query_key_projection_dim in qkv_dimensions:
                        for value_projection_dim in qkv_dimensions:
                            for add_key_value_bias_flag in bool_options:
                                for zero_attention_flag in bool_options:
                                    for average_attention_weights_flag in bool_options:
                                        c = TransformerPresets.transformer_preset(
                                            batch_size=batch_size,
                                            num_layers=num_layers,
                                            input_dim=embedding_dim,
                                            hidden_dim=embedding_dim,
                                            output_dim=embedding_dim,
                                            attention_num_heads=num_heads,
                                            embedding_dim=embedding_dim,
                                            attention_model_type=model_type,
                                            attention_target_sequence_length=target_sequence_length,
                                            attention_source_sequence_length=source_sequence_length,
                                            attention_query_key_projection_dim=query_key_projection_dim,
                                            attention_value_projection_dim=value_projection_dim,
                                            attention_add_key_value_bias_flag=add_key_value_bias_flag,
                                            attention_zero_attention_flag=zero_attention_flag,
                                            attention_average_attention_weights_flag=average_attention_weights_flag,
                                        )
                                        m = TransformerDecoderStack(c)
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
                                            create_key_padding_mask(
                                                batch_size, target_sequence_length
                                            ),
                                        )
                                        encoder_padding_mask_options = (
                                            None,
                                            create_key_padding_mask(
                                                batch_size, source_sequence_length
                                            ),
                                        )
                                        attention_mask_options = (
                                            None,
                                            create_attention_mask(
                                                target_sequence_length,
                                                target_sequence_length,
                                                batch_size * num_heads,
                                            ),
                                        )
                                        encoder_attention_mask_options = (
                                            None,
                                            create_attention_mask(
                                                target_sequence_length,
                                                source_sequence_length,
                                                batch_size * num_heads,
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
                                            message = f"Test failed for the inputs: model_type={model_type}, target_sequence_length={target_sequence_length}, source_sequence_length={source_sequence_length}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, add_key_value_bias_flag={add_key_value_bias_flag}, zero_attention_flag={zero_attention_flag}, average_attention_weights_flag={average_attention_weights_flag}, key_padding_mask={key_padding_mask.shape if key_padding_mask is not None else None}, encoder_padding_mask={encoder_padding_mask.shape if encoder_padding_mask is not None else None}, attention_mask={attention_mask.shape if attention_mask is not None else None}, encoder_attention_mask={encoder_attention_mask.shape if encoder_attention_mask is not None else None}"
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
                                                    target_sequence_length,
                                                    batch_size,
                                                    embedding_dim,
                                                )

                                                output, loss = output

                                                self.assertEqual(
                                                    output.shape, expected_output
                                                )
                                                self.assertIsInstance(
                                                    loss, torch.Tensor
                                                )


class TestTransformer(unittest.TestCase):
    def test_init(self):
        batch_size = 4
        num_heads = 2
        source_sequence_length = 6
        target_sequence_length = 6
        embedding_dim = 10
        num_layers = 2

        c = TransformerPresets.transformer_preset(
            batch_size=batch_size,
            num_layers=num_layers,
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            attention_num_heads=num_heads,
            embedding_dim=embedding_dim,
            attention_target_sequence_length=target_sequence_length,
            attention_source_sequence_length=source_sequence_length,
        )
        m = Transformer(c)

        self.assertIsInstance(m.encoder_model, TransformerEncoderStack)
        self.assertIsInstance(m.decoder_model, TransformerDecoderStack)

    def test_all_possible_inputs(self):
        batch_size = 4
        num_heads = 2
        embedding_dim = 10
        num_layers = 2
        sequence_lengths = [6, 10]
        qkv_dimensions = [0, 16, 20]
        model_types = list(LinearLayerStackOptions)
        bool_options = [True, False]

        for model_type in model_types:
            for target_sequence_length in sequence_lengths:
                for source_sequence_length in sequence_lengths:
                    for query_key_projection_dim in qkv_dimensions:
                        for value_projection_dim in qkv_dimensions:
                            for add_key_value_bias_flag in bool_options:
                                for zero_attention_flag in bool_options:
                                    for average_attention_weights_flag in bool_options:
                                        c = TransformerPresets.transformer_preset(
                                            batch_size=batch_size,
                                            num_layers=num_layers,
                                            input_dim=embedding_dim,
                                            hidden_dim=embedding_dim,
                                            output_dim=embedding_dim,
                                            attention_num_heads=num_heads,
                                            embedding_dim=embedding_dim,
                                            attention_model_type=model_type,
                                            attention_target_sequence_length=target_sequence_length,
                                            attention_source_sequence_length=source_sequence_length,
                                            attention_query_key_projection_dim=query_key_projection_dim,
                                            attention_value_projection_dim=value_projection_dim,
                                            attention_add_key_value_bias_flag=add_key_value_bias_flag,
                                            attention_zero_attention_flag=zero_attention_flag,
                                            attention_average_attention_weights_flag=average_attention_weights_flag,
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
                                            create_key_padding_mask(
                                                batch_size, source_sequence_length
                                            ),
                                        )
                                        source_attention_mask_options = (
                                            None,
                                            create_attention_mask(
                                                source_sequence_length,
                                                source_sequence_length,
                                                batch_size * num_heads,
                                            ),
                                        )
                                        target_key_padding_mask_options = (
                                            None,
                                            create_key_padding_mask(
                                                batch_size, target_sequence_length
                                            ),
                                        )
                                        memory_padding_mask_options = (
                                            None,
                                            create_key_padding_mask(
                                                batch_size, source_sequence_length
                                            ),
                                        )
                                        target_attention_mask_options = (
                                            None,
                                            create_attention_mask(
                                                target_sequence_length,
                                                target_sequence_length,
                                                batch_size * num_heads,
                                            ),
                                        )
                                        memory_attention_mask_options = (
                                            None,
                                            create_attention_mask(
                                                target_sequence_length,
                                                source_sequence_length,
                                                batch_size * num_heads,
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
                                            message = f"Test failed for the inputs: model_type={model_type}, target_sequence_length={target_sequence_length}, source_sequence_length={source_sequence_length}, query_key_projection_dim={query_key_projection_dim}, value_projection_dim={value_projection_dim}, add_key_value_bias_flag={add_key_value_bias_flag}, zero_attention_flag={zero_attention_flag}, average_attention_weights_flag={average_attention_weights_flag}, source_key_padding_mask={source_key_padding_mask.shape if source_key_padding_mask is not None else None}, source_attention_mask={source_attention_mask.shape if source_attention_mask is not None else None}, target_key_padding_mask={target_key_padding_mask.shape if target_key_padding_mask is not None else None}, memory_key_padding_mask={memory_key_padding_mask.shape if memory_key_padding_mask is not None else None}, target_attention_mask={target_attention_mask.shape if target_attention_mask is not None else None}, memory_attention_mask={memory_attention_mask.shape if memory_attention_mask is not None else None}"
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

                                                expected_output = (
                                                    target_sequence_length,
                                                    batch_size,
                                                    embedding_dim,
                                                )

                                                self.assertEqual(
                                                    output.shape, expected_output
                                                )
                                                self.assertIsInstance(
                                                    loss, torch.Tensor
                                                )

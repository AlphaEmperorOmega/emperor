import torch
import itertools
import unittest

from dataclasses import asdict
from torch.nn import LayerNorm, ModuleList
from Emperor.transformer.layer import (
    TransformerConfig,
    TransformerDecoder,
    TransformerEncoder,
)
from docs.config import default_unittest_config


class TestTransformerEncoder(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.num_heads = None
        self.input_dim = None
        self.output_dim = None
        self.embedding_dim = None
        self.layer_norm_position = None
        self.layer_norm_dim = None
        self.causalattention = None

    def rebuild_presets(self, config: TransformerConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.transformer_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = TransformerDecoder(self.cfg)

        self.batch_size: int = self.cfg.batch_size
        self.num_heads: int = self.cfg.multi_head_attention_model_config.num_heads
        self.input_dim: int = self.cfg.input_dim
        self.embedding_dim: int = (
            self.cfg.multi_head_attention_model_config.embedding_dim
        )

        # Model attributes
        self.num_layers: int = self.config.num_layers
        self.source_sequence_length: int = self.config.source_sequence_length
        self.target_sequence_length: int = self.config.target_sequence_length
        self.layer_norm_dim: int = self.config.layer_norm_dim


class Test___init(TestTransformerEncoder):
    def test___init(self):
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.source_sequence_length, self.source_sequence_length)
        self.assertEqual(self.model.target_sequence_length, self.target_sequence_length)
        self.assertEqual(self.model.layer_norm_dim, self.layer_norm_dim)
        self.assertIsInstance(self.model.layer_norm_module, LayerNorm)
        self.assertIsInstance(self.model.layers, ModuleList)
        self.assertEqual(len(self.model.layers), self.num_layers)


class Test_forward(TestTransformerEncoder):
    def test_all_possible_inputs(self):
        target_token_embeddings = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        encoder_output = torch.randn(
            self.source_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )

        key_padding_mask_options = (
            None,
            torch.randn(self.batch_size, self.target_sequence_length),
        )
        encoder_padding_mask_options = (
            None,
            torch.randn(self.batch_size, self.source_sequence_length),
        )
        attention_mask_options = (
            None,
            torch.randn(
                self.batch_size * self.num_heads,
                self.target_sequence_length,
                self.target_sequence_length,
            ),
        )
        encoder_attention_mask_options = (
            None,
            torch.randn(
                self.batch_size * self.num_heads,
                self.source_sequence_length,
                self.target_sequence_length,
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
                output = self.model(
                    target_token_embeddings=target_token_embeddings,
                    encoder_output=encoder_output,
                    target_key_padding_mask=key_padding_mask,
                    encoder_key_padding_mask=encoder_padding_mask,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )

                expected_output = (
                    self.source_sequence_length,
                    self.batch_size,
                    self.embedding_dim,
                )

                output, loss = output

                self.assertEqual(output.shape, expected_output)
                self.assertIsInstance(loss, torch.Tensor)

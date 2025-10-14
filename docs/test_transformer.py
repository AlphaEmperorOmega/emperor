import torch
import itertools
import unittest

from dataclasses import asdict
from torch.nn import LayerNorm, ModuleList
from Emperor.transformer.layer import (
    Transformer,
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

        self.model = Transformer(self.cfg)

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
        self.assertIsInstance(self.model.encoder_model, TransformerEncoder)
        self.assertIsInstance(self.model.decoder_model, TransformerDecoder)


class Test_forward(TestTransformerEncoder):
    def test_all_possible_inputs(self):
        source_token_embeddings = torch.randn(
            self.source_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        target_token_embeddings = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )

        source_key_padding_mask_options = (
            None,
            torch.randn(self.batch_size, self.source_sequence_length),
        )
        source_attention_mask_options = (
            None,
            torch.randn(
                self.batch_size * self.num_heads,
                self.source_sequence_length,
                self.target_sequence_length,
            ),
        )
        target_key_padding_mask_options = (
            None,
            torch.randn(self.batch_size, self.target_sequence_length),
        )
        memory_padding_mask_options = (
            None,
            torch.randn(self.batch_size, self.source_sequence_length),
        )
        target_attention_mask_options = (
            None,
            torch.randn(
                self.batch_size * self.num_heads,
                self.target_sequence_length,
                self.target_sequence_length,
            ),
        )
        memory_attention_mask_options = (
            None,
            torch.randn(
                self.batch_size * self.num_heads,
                self.source_sequence_length,
                self.target_sequence_length,
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
                output, loss = self.model(
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
                    self.source_sequence_length,
                    self.batch_size,
                    self.embedding_dim,
                )

                self.assertEqual(output.shape, expected_output)
                self.assertIsInstance(loss, torch.Tensor)

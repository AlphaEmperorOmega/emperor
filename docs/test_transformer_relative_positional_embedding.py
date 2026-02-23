import torch
import unittest
import torch.nn as nn

from torch import Tensor
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.embedding.relative.options.learned_embedding import LearnedPositionalBias
from Emperor.embedding.options import RelativePositionalEmbeddingOptions
from Emperor.embedding.relative.factory import RelativePositionalEmbeddingFactory


class TestLearnedPositionalBias(unittest.TestCase):
    def test_init(self):
        num_embeddings = 64
        embedding_dim = 8
        num_heads = 4
        max_positions = 32
        padding_idx = 0
        init_size = 64
        auto_expand_flag = False

        c = TransformerPresets.transformer_relative_positional_embedding_preset(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_positions=max_positions,
            padding_idx=padding_idx,
            init_size=init_size,
            auto_expand_flag=auto_expand_flag,
        )
        m = LearnedPositionalBias(c)

        expected_head_dim = embedding_dim // num_heads
        expected_num_embeddings = num_embeddings + padding_idx + 1

        self.assertEqual(m.embedding_dim, embedding_dim)
        self.assertEqual(m.padding_idx, padding_idx)
        self.assertEqual(m.num_heads, num_heads)
        self.assertEqual(m.head_dim, expected_head_dim)
        self.assertEqual(m.max_positions, max_positions)
        self.assertEqual(m.init_size, init_size)
        self.assertEqual(m.auto_expand_flag, auto_expand_flag)
        self.assertEqual(m.num_embeddings, expected_num_embeddings)
        self.assertIsInstance(m.relative_positional_emnbeddings, nn.Parameter)
        self.assertEqual(
            m.relative_positional_emnbeddings.shape,
            (num_heads, expected_head_dim, max_positions * 2 + 1),
        )

    def test_forward(self):
        num_embeddings = 64
        embedding_dim = 8
        num_heads = 4
        max_positions = 32
        padding_idx = 0
        init_size = 64
        auto_expand_flag = False

        c = TransformerPresets.transformer_relative_positional_embedding_preset(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_positions=max_positions,
            padding_idx=padding_idx,
            init_size=init_size,
            auto_expand_flag=auto_expand_flag,
        )
        m = LearnedPositionalBias(c)

        batch_size = 4
        seq_len = 6
        head_dim = embedding_dim // num_heads
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        output = m(query, sequence_length=seq_len)

        expected_shape = (batch_size, num_heads, seq_len, seq_len)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_last(self):
        num_embeddings = 64
        embedding_dim = 8
        num_heads = 4
        max_positions = 32
        padding_idx = 0
        init_size = 64
        auto_expand_flag = False

        c = TransformerPresets.transformer_relative_positional_embedding_preset(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_positions=max_positions,
            padding_idx=padding_idx,
            init_size=init_size,
            auto_expand_flag=auto_expand_flag,
        )
        m = LearnedPositionalBias(c)

        batch_size = 4
        seq_len = 1
        head_dim = embedding_dim // num_heads
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        output = m(query, sequence_length=seq_len, last=True)

        expected_shape = (batch_size, num_heads, seq_len, seq_len)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, expected_shape)


class TestRelativePositionalEmbeddingFactory(unittest.TestCase):
    def test_init(self):
        num_embeddings = 64
        embedding_dim = 8
        num_heads = 4
        max_positions = 32
        padding_idx = 0
        init_size = 64
        auto_expand_flag = False

        for positional_embedding_option in RelativePositionalEmbeddingOptions:
            message = f"Testing option: {positional_embedding_option.value}"
            with self.subTest(msg=message):
                c = TransformerPresets.transformer_relative_positional_embedding_preset(
                    positional_embedding_option=positional_embedding_option,
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    max_positions=max_positions,
                    padding_idx=padding_idx,
                    init_size=init_size,
                    auto_expand_flag=auto_expand_flag,
                )
                if (
                    positional_embedding_option
                    == RelativePositionalEmbeddingOptions.DISABLED
                ):
                    with self.assertRaises(ValueError):
                        RelativePositionalEmbeddingFactory(c).build()
                    continue

                m = RelativePositionalEmbeddingFactory(c).build()
                if (
                    positional_embedding_option
                    == RelativePositionalEmbeddingOptions.LEARNED
                ):
                    self.assertIsInstance(m, LearnedPositionalBias)

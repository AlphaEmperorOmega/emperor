import torch
import unittest
import torch.nn as nn

from torch import Tensor
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.transformer.utils.embedding.selector import (
    PositionalEmbedding,
    PositionalEmbeddingOptions,
)
from Emperor.transformer.utils.embedding.options.learned_embedding import (
    LearnedPositionalEmbedding,
)
from Emperor.transformer.utils.embedding.options.sinusoidal_embedding import (
    SinusoidalPositionalEmbedding,
)


class TestSinusoidalPositionalEmbedding(unittest.TestCase):
    def test_init(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings
        auto_expand_flag = True

        c = TransformerPresets.transformer_positional_embedding_preset(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
            auto_expand_flag=auto_expand_flag,
        )
        m = SinusoidalPositionalEmbedding(c)

        self.assertEqual(m.embedding_dim, embedding_dim)
        self.assertEqual(m.padding_idx, padding_idx)
        self.assertEqual(m.init_size, init_size + padding_idx + 1)

    def test_forward(self):
        num_embeddings = 64
        embedding_dim = 5
        padding_idx = 0
        init_size = num_embeddings
        auto_expand_flag = True

        c = TransformerPresets.transformer_positional_embedding_preset(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
            auto_expand_flag=auto_expand_flag,
        )

        m = SinusoidalPositionalEmbedding(c)

        batch_size = 4
        sequence_length = 6
        input_tokens = torch.randint(0, embedding_dim, (batch_size, sequence_length))
        output_positional_tokens = m(input_tokens)

        expected_shape = (batch_size, sequence_length, embedding_dim)
        self.assertIsInstance(output_positional_tokens, Tensor)
        self.assertEqual(output_positional_tokens.shape, expected_shape)


class TestLearnedPositionalEmbedding(unittest.TestCase):
    def test_init(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings
        auto_expand_flag = True

        c = TransformerPresets.transformer_positional_embedding_preset(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
            auto_expand_flag=auto_expand_flag,
        )
        m = LearnedPositionalEmbedding(c)

        self.assertEqual(m.embedding_dim, embedding_dim)
        self.assertEqual(m.padding_idx, padding_idx)
        self.assertEqual(m.num_embeddings, num_embeddings + padding_idx + 1)
        self.assertEqual(m.init_size, init_size)
        self.assertEqual(m.auto_expand_flag, auto_expand_flag)
        self.assertIsInstance(m.embedding_model, nn.Embedding)

    def test_forward(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings
        auto_expand_flag = True

        c = TransformerPresets.transformer_positional_embedding_preset(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
            auto_expand_flag=auto_expand_flag,
        )
        m = LearnedPositionalEmbedding(c)

        batch_size = 4
        sequence_length = 16
        input_tokens = torch.randint(0, embedding_dim, (batch_size, sequence_length))
        output_positional_tokens = m(input_tokens)

        expected_shape = (batch_size, sequence_length, embedding_dim)
        self.assertIsInstance(output_positional_tokens, Tensor)
        self.assertEqual(output_positional_tokens.shape, expected_shape)


class TestPositionalEmbedding(unittest.TestCase):
    def test_init(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings
        auto_expand_flag = True

        for positional_embedding_option in PositionalEmbeddingOptions:
            message = f"Testing model type: {positional_embedding_option.value}"
            with self.subTest(msg=message):
                c = TransformerPresets.transformer_positional_embedding_preset(
                    positional_embedding_option=positional_embedding_option,
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                    init_size=init_size,
                    auto_expand_flag=auto_expand_flag,
                )
                if positional_embedding_option == PositionalEmbeddingOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        m = PositionalEmbedding(c).build_model()
                    continue

                m = PositionalEmbedding(c).build_model()
                if positional_embedding_option == PositionalEmbeddingOptions.SINUSOIDAL:
                    self.assertIsInstance(m, SinusoidalPositionalEmbedding)
                if positional_embedding_option == PositionalEmbeddingOptions.LEARNED:
                    self.assertIsInstance(m, LearnedPositionalEmbedding)

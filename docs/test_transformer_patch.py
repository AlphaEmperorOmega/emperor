import torch
import unittest
import torch.nn as nn

from emperor.transformer.utils.presets import TransformerPresets
from emperor.transformer.utils.patch.selector import PatchOptions
from emperor.transformer.utils.patch.options.patch_embedding_conv import (
    PatchEmbeddingConv,
)
from emperor.transformer.utils.patch.options.patch_embedding_linear import (
    PatchEmbeddingLinear,
)


class TestPatchTokenizerEmbedding(unittest.TestCase):
    def test_init(self):
        patch_option = PatchOptions.LINEAR
        embedding_dim = 32
        patch_size = 3
        stride = 3
        padding = 2
        dropout = 0.0

        c = TransformerPresets.transformer_patch_preset(
            patch_option=patch_option,
            embedding_dim=embedding_dim,
            patch_size=patch_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
        )
        m = PatchEmbeddingLinear(c)

        self.assertEqual(m.embedding_dim, embedding_dim)
        self.assertEqual(m.patch_size, patch_size)
        self.assertEqual(m.stride, stride)
        self.assertEqual(m.padding, padding)
        self.assertEqual(m.dropout_probability, dropout)
        self.assertEqual(m.class_token.shape, (1, 1, embedding_dim))
        self.assertIsInstance(m.patch_model, nn.Unfold)

    def test_forward(self):
        patch_option = PatchOptions.LINEAR
        embedding_dim = 32
        patch_size = 5
        stride = 5
        padding = 0
        dropout = 0.0

        c = TransformerPresets.transformer_patch_preset(
            patch_option=patch_option,
            embedding_dim=embedding_dim,
            patch_size=patch_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
        )
        m = PatchEmbeddingLinear(c)

        batch_size = 4
        num_channels = 1
        image_width = 28
        image_height = 28

        input_shape = (batch_size, num_channels, image_height, image_width)
        input_tokens = torch.randn(input_shape)
        output_positional_tokens = m(input_tokens)

        # expected_shape = (batch_size, sequence_length, embedding_dim)
        # self.assertIsInstance(output_positional_tokens, Tensor)
        # self.assertEqual(output_positional_tokens.shape, expected_shape)


# class TestLearnedPositionalEmbedding(unittest.TestCase):
#     def test_init(self):
#         num_embeddings = 64
#         embedding_dim = 10
#         padding_idx = 0
#         init_size = num_embeddings
#         auto_expand_flag = True
#
#         c = TransformerPresets.transformer_positional_embedding_preset(
#             num_embeddings=num_embeddings,
#             embedding_dim=embedding_dim,
#             padding_idx=padding_idx,
#             init_size=init_size,
#             auto_expand_flag=auto_expand_flag,
#         )
#         m = LearnedPositionalEmbedding(c)
#
#         self.assertEqual(m.embedding_dim, embedding_dim)
#         self.assertEqual(m.padding_idx, padding_idx)
#         self.assertEqual(m.num_embeddings, num_embeddings + padding_idx + 1)
#         self.assertEqual(m.init_size, init_size)
#         self.assertEqual(m.auto_expand_flag, auto_expand_flag)
#         self.assertIsInstance(m.embedding_model, nn.Embedding)
#
#     def test_forward(self):
#         num_embeddings = 64
#         embedding_dim = 10
#         padding_idx = 0
#         init_size = num_embeddings
#         auto_expand_flag = True
#
#         c = TransformerPresets.transformer_positional_embedding_preset(
#             num_embeddings=num_embeddings,
#             embedding_dim=embedding_dim,
#             padding_idx=padding_idx,
#             init_size=init_size,
#             auto_expand_flag=auto_expand_flag,
#         )
#         m = LearnedPositionalEmbedding(c)
#
#         batch_size = 4
#         sequence_length = 16
#         input_tokens = torch.randint(0, embedding_dim, (batch_size, sequence_length))
#         output_positional_tokens = m(input_tokens)
#
#         expected_shape = (batch_size, sequence_length, embedding_dim)
#         self.assertIsInstance(output_positional_tokens, Tensor)
#         self.assertEqual(output_positional_tokens.shape, expected_shape)
#
#
# class TestPositionalEmbedding(unittest.TestCase):
#     def test_init(self):
#         num_embeddings = 64
#         embedding_dim = 10
#         padding_idx = 0
#         init_size = num_embeddings
#         auto_expand_flag = True
#
#         for positional_embedding_option in PositionalEmbeddingOptions:
#             message = f"Testing model type: {positional_embedding_option.value}"
#             with self.subTest(msg=message):
#                 c = TransformerPresets.transformer_positional_embedding_preset(
#                     positional_embedding_option=positional_embedding_option,
#                     num_embeddings=num_embeddings,
#                     embedding_dim=embedding_dim,
#                     padding_idx=padding_idx,
#                     init_size=init_size,
#                     auto_expand_flag=auto_expand_flag,
#                 )
#                 if positional_embedding_option == PositionalEmbeddingOptions.DISABLED:
#                     with self.assertRaises(ValueError):
#                         m = PositionalEmbeddingSelector(c).build()
#                     continue
#
#                 m = PositionalEmbeddingSelector(c).build()
#                 if positional_embedding_option == PositionalEmbeddingOptions.SINUSOIDAL:
#                     self.assertIsInstance(m, SinusoidalPositionalEmbedding)
#                 if positional_embedding_option == PositionalEmbeddingOptions.LEARNED:
#                     self.assertIsInstance(m, LearnedPositionalEmbedding)

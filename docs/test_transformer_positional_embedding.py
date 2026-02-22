import torch
import unittest
import torch.nn as nn

from torch import Tensor
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.embedding.absolute.options.learned_embedding import (
    LearnedPositionalEmbedding,
    TextLearnedPositionalEmbedding,
    ImageLearnedPositionalEmbedding,
)
from Emperor.embedding.absolute.options.sinusoidal_embedding import (
    SinusoidalPositionalEmbedding,
    TextSinusoidalPositionalEmbedding,
    ImageSinusoidalPositionalEmbedding,
)
from Emperor.embedding.options import AbsolutePositionalEmbeddingOptions
from Emperor.embedding.absolute.factory import AbsolutePositionalEmbeddingFactory


class TestTextLearnedPositionalEmbedding(unittest.TestCase):
    def test_init(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings
        auto_expand_flag = True

        c = TransformerPresets.transformer_positional_embedding_preset(
            text_processing_flag=True,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
            auto_expand_flag=auto_expand_flag,
        )
        m = TextLearnedPositionalEmbedding(c)

        self.assertEqual(m.embedding_dim, embedding_dim)
        self.assertEqual(m.padding_idx, padding_idx)
        self.assertEqual(m.num_embeddings, num_embeddings + 1)
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
            text_processing_flag=True,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
            auto_expand_flag=auto_expand_flag,
        )
        m = TextLearnedPositionalEmbedding(c)

        batch_size = 4
        sequence_length = num_embeddings
        input_tokens = torch.randint(0, num_embeddings, (batch_size, sequence_length))
        output = m(input_tokens)

        expected_shape = (batch_size, sequence_length, embedding_dim)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, expected_shape)


class TestImageLearnedPositionalEmbedding(unittest.TestCase):
    def test_init(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings
        auto_expand_flag = True

        for class_token_flag in [True, False]:
            message = f"Test failed for class_token_flag: {class_token_flag}"
            with self.subTest(i=message):
                c = TransformerPresets.transformer_positional_embedding_preset(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                    init_size=init_size,
                    auto_expand_flag=auto_expand_flag,
                    class_token_flag=class_token_flag,
                )
                m = ImageLearnedPositionalEmbedding(c)

                expected_num_embeddings = (
                    num_embeddings + padding_idx + 1
                    if class_token_flag
                    else num_embeddings
                )
                self.assertEqual(m.embedding_dim, embedding_dim)
                self.assertEqual(m.padding_idx, padding_idx)
                self.assertEqual(m.num_embeddings, expected_num_embeddings)
                self.assertEqual(m.init_size, init_size)
                self.assertEqual(m.auto_expand_flag, auto_expand_flag)
                self.assertEqual(m.class_token_flag, class_token_flag)
                self.assertIsInstance(m.embedding_model, nn.Embedding)

    def test_forward(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings
        auto_expand_flag = True

        for class_token_flag in [True, False]:
            message = f"Test failed for class_token_flag: {class_token_flag}"
            with self.subTest(i=message):
                c = TransformerPresets.transformer_positional_embedding_preset(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                    init_size=init_size,
                    auto_expand_flag=auto_expand_flag,
                    class_token_flag=class_token_flag,
                )
                m = ImageLearnedPositionalEmbedding(c)

                batch_size = 4
                patch_features = torch.randn(batch_size, num_embeddings, embedding_dim)

                if class_token_flag:
                    cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
                    cls_tokens = cls_token.expand(batch_size, -1, -1)
                    input_features = torch.cat([cls_tokens, patch_features], dim=1)
                    expected_shape = (batch_size, num_embeddings + 1, embedding_dim)
                else:
                    input_features = patch_features
                    expected_shape = (batch_size, num_embeddings, embedding_dim)

                output = m(input_features)

                self.assertIsInstance(output, Tensor)
                self.assertEqual(output.shape, expected_shape)


# class TestTextSinusoidalPositionalEmbedding(unittest.TestCase):
#     def test_init(self):
#         num_embeddings = 64
#         embedding_dim = 10
#         padding_idx = 0
#         init_size = num_embeddings
#         auto_expand_flag = True
#
#         c = TransformerPresets.transformer_positional_embedding_preset(
#             text_processing_flag=True,
#             num_embeddings=num_embeddings,
#             embedding_dim=embedding_dim,
#             padding_idx=padding_idx,
#             init_size=init_size,
#             auto_expand_flag=auto_expand_flag,
#         )
#         m = TextSinusoidalPositionalEmbedding(c)
#
#         self.assertEqual(m.embedding_dim, embedding_dim)
#         self.assertEqual(m.padding_idx, padding_idx)
#         self.assertEqual(m.init_size, init_size + padding_idx + 1)
#         self.assertEqual(m.auto_expand_flag, auto_expand_flag)
#
#     def test_forward(self):
#         num_embeddings = 64
#         embedding_dim = 10
#         padding_idx = 0
#         init_size = num_embeddings
#         auto_expand_flag = True
#
#         c = TransformerPresets.transformer_positional_embedding_preset(
#             text_processing_flag=True,
#             num_embeddings=num_embeddings,
#             embedding_dim=embedding_dim,
#             padding_idx=padding_idx,
#             init_size=init_size,
#             auto_expand_flag=auto_expand_flag,
#         )
#         m = TextSinusoidalPositionalEmbedding(c)
#
#         batch_size = 4
#         sequence_length = 6
#         input_tokens = torch.randint(0, num_embeddings, (batch_size, sequence_length))
#         output = m(input_tokens)
#
#         expected_shape = (batch_size, sequence_length, embedding_dim)
#         self.assertIsInstance(output, Tensor)
#         self.assertEqual(output.shape, expected_shape)
#
#
# class TestImageSinusoidalPositionalEmbedding(unittest.TestCase):
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
#         m = ImageSinusoidalPositionalEmbedding(c)
#
#         self.assertEqual(m.embedding_dim, embedding_dim)
#         self.assertEqual(m.padding_idx, padding_idx)
#         self.assertEqual(m.init_size, init_size + padding_idx + 1)
#         self.assertEqual(m.auto_expand_flag, auto_expand_flag)
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
#         m = ImageSinusoidalPositionalEmbedding(c)
#
#         batch_size = 4
#         sequence_length = init_size + padding_idx + 1
#         input_features = torch.randn(batch_size, sequence_length, embedding_dim)
#         output = m(input_features)
#
#         expected_shape = (batch_size, sequence_length, embedding_dim)
#         self.assertIsInstance(output, Tensor)
#         self.assertEqual(output.shape, expected_shape)
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
#         for positional_embedding_option in AbsolutePositionalEmbeddingOptions:
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
#                 if (
#                     positional_embedding_option
#                     == AbsolutePositionalEmbeddingOptions.DISABLED
#                 ):
#                     with self.assertRaises(ValueError):
#                         m = AbsolutePositionalEmbeddingFactory(c).build()
#                     continue
#
#                 m = AbsolutePositionalEmbeddingFactory(c).build()
#                 if (
#                     positional_embedding_option
#                     == AbsolutePositionalEmbeddingOptions.SINUSOIDAL
#                 ):
#                     self.assertIsInstance(m, SinusoidalPositionalEmbedding)
#                 if (
#                     positional_embedding_option
#                     == AbsolutePositionalEmbeddingOptions.LEARNED
#                 ):
#                     self.assertIsInstance(m, LearnedPositionalEmbedding)

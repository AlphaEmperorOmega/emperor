import torch
import unittest
import torch.nn as nn

from torch import Tensor
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.embedding.absolute.utils.options.learned_embedding import (
    TextLearnedPositionalEmbedding,
    ImageLearnedPositionalEmbedding,
)
from Emperor.embedding.absolute.utils.options.sinusoidal_embedding import (
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

    def test_forward_incremental(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings

        c = TransformerPresets.transformer_positional_embedding_preset(
            text_processing_flag=True,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
        )
        m = TextLearnedPositionalEmbedding(c)

        batch_size = 4
        sequence_length = 1
        input_tokens = torch.randint(1, num_embeddings, (batch_size, sequence_length))
        output = m(input_tokens, incremental_state={})

        expected_shape = (1, 1, embedding_dim)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_explicit_positions(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings

        c = TransformerPresets.transformer_positional_embedding_preset(
            text_processing_flag=True,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
        )
        m = TextLearnedPositionalEmbedding(c)

        batch_size = 4
        sequence_length = num_embeddings
        input_tokens = torch.randint(0, num_embeddings, (batch_size, sequence_length))
        positions = (
            torch.arange(1, sequence_length + 1).unsqueeze(0).expand(batch_size, -1)
        )
        output = m(input_tokens, positions=positions)

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


class TestTextSinusoidalPositionalEmbedding(unittest.TestCase):
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
        m = TextSinusoidalPositionalEmbedding(c)

        self.assertEqual(m.embedding_dim, embedding_dim)
        self.assertEqual(m.padding_idx, padding_idx)
        self.assertEqual(m.num_embeddings, num_embeddings)
        self.assertEqual(m.init_size, init_size + padding_idx + 1)
        self.assertEqual(m.auto_expand_flag, auto_expand_flag)
        self.assertIsInstance(m.weights, Tensor)

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
        m = TextSinusoidalPositionalEmbedding(c)

        batch_size = 4
        sequence_length = num_embeddings
        input_tokens = torch.randint(0, num_embeddings, (batch_size, sequence_length))
        output = m(input_tokens)

        expected_shape = (batch_size, sequence_length, embedding_dim)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, expected_shape)

    def test_auto_expand(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings
        batch_size = 4
        sequence_within_bounds = init_size
        sequence_beyond_bounds = init_size + 10

        for auto_expand_flag in [True, False]:
            message = f"Test failed for auto_expand_flag: {auto_expand_flag}"
            with self.subTest(i=message):
                c = TransformerPresets.transformer_positional_embedding_preset(
                    text_processing_flag=True,
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                    init_size=init_size,
                    auto_expand_flag=auto_expand_flag,
                )
                m = TextSinusoidalPositionalEmbedding(c)
                initial_weights_size = m.weights.size(0)

                input_within = torch.randint(
                    1, num_embeddings, (batch_size, sequence_within_bounds)
                )
                m(input_within)
                self.assertEqual(m.weights.size(0), initial_weights_size)

                input_beyond = torch.randint(
                    1, num_embeddings, (batch_size, sequence_beyond_bounds)
                )
                expected_expanded_size = padding_idx + 1 + sequence_beyond_bounds

                if auto_expand_flag:
                    output = m(input_beyond)
                    self.assertIsInstance(output, Tensor)
                    self.assertEqual(
                        output.shape,
                        (batch_size, sequence_beyond_bounds, embedding_dim),
                    )
                    self.assertEqual(m.weights.size(0), expected_expanded_size)
                else:
                    self.assertEqual(m.weights.size(0), initial_weights_size)

    def test_forward_incremental(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings
        batch_size = 4
        sequence_length = 6

        for use_timestep in [True, False]:
            message = f"Test failed for use_timestep: {use_timestep}"
            with self.subTest(i=message):
                c = TransformerPresets.transformer_positional_embedding_preset(
                    text_processing_flag=True,
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                    init_size=init_size,
                    auto_expand_flag=True,
                )
                m = TextSinusoidalPositionalEmbedding(c)
                input_tokens = torch.randint(
                    1, num_embeddings, (batch_size, sequence_length)
                )

                if use_timestep:
                    timestep = torch.tensor([sequence_length - 1])
                    output = m(input_tokens, incremental_state={}, timestep=timestep)
                else:
                    output = m(input_tokens, incremental_state={})

                expected_shape = (batch_size, 1, embedding_dim)
                self.assertIsInstance(output, Tensor)
                self.assertEqual(output.shape, expected_shape)

    def test_padding_masking(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings

        c = TransformerPresets.transformer_positional_embedding_preset(
            text_processing_flag=True,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
            auto_expand_flag=True,
        )
        m = TextSinusoidalPositionalEmbedding(c)

        batch_size = 2
        sequence_length = 8
        input_tokens = torch.randint(1, num_embeddings, (batch_size, sequence_length))
        input_tokens[:, 0] = padding_idx

        output = m(input_tokens)

        self.assertTrue(output[:, 0, :].eq(0).all())
        self.assertFalse(output[:, 1:, :].eq(0).all())


class TestImageSinusoidalPositionalEmbedding(unittest.TestCase):
    def test_init(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings

        c = TransformerPresets.transformer_positional_embedding_preset(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
        )
        m = ImageSinusoidalPositionalEmbedding(c)

        self.assertEqual(m.embedding_dim, embedding_dim)
        self.assertEqual(m.padding_idx, padding_idx)
        self.assertEqual(m.num_embeddings, num_embeddings)
        self.assertEqual(m.init_size, init_size + padding_idx + 1)
        self.assertIsInstance(m.weights, Tensor)

    def test_forward(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings

        c = TransformerPresets.transformer_positional_embedding_preset(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            init_size=init_size,
        )
        m = ImageSinusoidalPositionalEmbedding(c)

        batch_size = 4
        sequence_length = init_size + padding_idx + 1
        input_features = torch.randn(batch_size, sequence_length, embedding_dim)
        output = m(input_features)

        expected_shape = (batch_size, sequence_length, embedding_dim)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, expected_shape)


class TestAbsolutePositionalEmbeddingFactory(unittest.TestCase):
    def test_init(self):
        num_embeddings = 64
        embedding_dim = 10
        padding_idx = 0
        init_size = num_embeddings
        auto_expand_flag = True

        for text_processing_flag in [True, False]:
            for positional_embedding_option in AbsolutePositionalEmbeddingOptions:
                message = f"Testing option: {positional_embedding_option.value}, text_processing_flag: {text_processing_flag}"
                with self.subTest(msg=message):
                    c = TransformerPresets.transformer_positional_embedding_preset(
                        text_processing_flag=text_processing_flag,
                        positional_embedding_option=positional_embedding_option,
                        num_embeddings=num_embeddings,
                        embedding_dim=embedding_dim,
                        padding_idx=padding_idx,
                        init_size=init_size,
                        auto_expand_flag=auto_expand_flag,
                    )
                    if (
                        positional_embedding_option
                        == AbsolutePositionalEmbeddingOptions.DISABLED
                    ):
                        with self.assertRaises(ValueError):
                            AbsolutePositionalEmbeddingFactory(c).build()
                        continue

                    m = AbsolutePositionalEmbeddingFactory(c).build()
                    if (
                        positional_embedding_option
                        == AbsolutePositionalEmbeddingOptions.SINUSOIDAL
                    ):
                        if text_processing_flag:
                            self.assertIsInstance(m, TextSinusoidalPositionalEmbedding)
                        else:
                            self.assertIsInstance(m, ImageSinusoidalPositionalEmbedding)
                    if (
                        positional_embedding_option
                        == AbsolutePositionalEmbeddingOptions.LEARNED
                    ):
                        if text_processing_flag:
                            self.assertIsInstance(m, TextLearnedPositionalEmbedding)
                        else:
                            self.assertIsInstance(m, ImageLearnedPositionalEmbedding)

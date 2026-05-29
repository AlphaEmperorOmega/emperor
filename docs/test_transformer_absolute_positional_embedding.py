import torch
import unittest
import torch.nn as nn

from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    ImageLearnedPositionalEmbedding,
    ImageLearnedPositionalEmbeddingConfig,
    ImageSinusoidalPositionalEmbedding,
    ImageSinusoidalPositionalEmbeddingConfig,
    TextLearnedPositionalEmbedding,
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbedding,
    TextSinusoidalPositionalEmbeddingConfig,
)


class TestAbsolutePositionalEmbeddingConfig(unittest.TestCase):
    def test_base_config_cannot_build(self):
        cfg = AbsolutePositionalEmbeddingConfig(
            num_embeddings=8,
            embedding_dim=6,
            init_size=8,
            padding_idx=0,
            auto_expand_flag=False,
        )

        with self.assertRaises(NotImplementedError):
            cfg.build()


class TestTextLearnedPositionalEmbedding(unittest.TestCase):
    def preset(
        self,
        num_embeddings: int = 8,
        embedding_dim: int = 6,
        init_size: int = 8,
        padding_idx: int | None = 0,
        auto_expand_flag: bool = False,
    ) -> TextLearnedPositionalEmbeddingConfig:
        return TextLearnedPositionalEmbeddingConfig(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            init_size=init_size,
            padding_idx=padding_idx,
            auto_expand_flag=auto_expand_flag,
        )

    def test_init(self):
        cfg = self.preset(num_embeddings=10, embedding_dim=4)
        model = TextLearnedPositionalEmbedding(cfg)

        self.assertEqual(model.embedding_dim, cfg.embedding_dim)
        self.assertEqual(model.padding_idx, cfg.padding_idx)
        self.assertEqual(model.num_embeddings, cfg.num_embeddings + 1)
        self.assertEqual(model.init_size, cfg.init_size)
        self.assertEqual(model.auto_expand_flag, cfg.auto_expand_flag)
        self.assertIsInstance(model.embedding_model, nn.Embedding)

    def test_forward(self):
        cfg = self.preset(num_embeddings=10, embedding_dim=4)
        model = TextLearnedPositionalEmbedding(cfg)
        input_tokens = torch.randint(1, cfg.num_embeddings, (3, 5))

        output = model(input_tokens)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (3, 5, cfg.embedding_dim))

    def test_forward_incremental(self):
        cfg = self.preset(num_embeddings=10, embedding_dim=4)
        model = TextLearnedPositionalEmbedding(cfg)
        input_tokens = torch.randint(1, cfg.num_embeddings, (3, 1))

        output = model(input_tokens, incremental_state={})

        self.assertEqual(output.shape, (1, 1, cfg.embedding_dim))

    def test_forward_explicit_positions(self):
        cfg = self.preset(num_embeddings=10, embedding_dim=4)
        model = TextLearnedPositionalEmbedding(cfg)
        input_tokens = torch.randint(1, cfg.num_embeddings, (3, 5))
        positions = torch.arange(1, 6).unsqueeze(0).expand(3, -1)

        output = model(input_tokens, positions=positions)

        self.assertEqual(output.shape, (3, 5, cfg.embedding_dim))

    def test_config_build_returns_text_learned_embedding(self):
        cfg = self.preset()
        model = cfg.build()

        self.assertIsInstance(model, TextLearnedPositionalEmbedding)
        self.assertIsInstance(model, cfg._registry_owner())

    def test_config_build_applies_overrides(self):
        cfg = self.preset(num_embeddings=8, embedding_dim=4)
        overrides = TextLearnedPositionalEmbeddingConfig(
            num_embeddings=12,
            embedding_dim=6,
            init_size=12,
            padding_idx=0,
            auto_expand_flag=True,
        )
        model = cfg.build(overrides)

        self.assertEqual(model.embedding_dim, overrides.embedding_dim)
        self.assertEqual(model.num_embeddings, overrides.num_embeddings + 1)
        self.assertEqual(model.auto_expand_flag, overrides.auto_expand_flag)

    def test_init_raises_on_missing_or_invalid_fields(self):
        invalid_cases = [
            ("missing_num_embeddings", TextLearnedPositionalEmbeddingConfig(
                embedding_dim=4,
                init_size=8,
                padding_idx=0,
                auto_expand_flag=False,
            )),
            ("zero_embedding_dim", self.preset(embedding_dim=0)),
            ("negative_padding_idx", self.preset(padding_idx=-1)),
        ]

        for case, cfg in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    TextLearnedPositionalEmbedding(cfg)


class TestImageLearnedPositionalEmbedding(unittest.TestCase):
    def preset(
        self,
        num_embeddings: int = 4,
        embedding_dim: int = 6,
        init_size: int = 4,
        padding_idx: int | None = 0,
        auto_expand_flag: bool = False,
        class_token_flag: bool = True,
    ) -> ImageLearnedPositionalEmbeddingConfig:
        return ImageLearnedPositionalEmbeddingConfig(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            init_size=init_size,
            padding_idx=padding_idx,
            auto_expand_flag=auto_expand_flag,
            class_token_flag=class_token_flag,
        )

    def test_init_and_forward_with_class_token_options(self):
        for class_token_flag in [True, False]:
            with self.subTest(class_token_flag=class_token_flag):
                cfg = self.preset(class_token_flag=class_token_flag)
                model = ImageLearnedPositionalEmbedding(cfg)
                sequence_length = cfg.num_embeddings + int(class_token_flag)
                patch_embeddings = torch.randn(2, sequence_length, cfg.embedding_dim)

                output = model(patch_embeddings)

                self.assertEqual(model.class_token_flag, class_token_flag)
                self.assertEqual(model.embedding_model.num_embeddings, sequence_length)
                self.assertEqual(output.shape, patch_embeddings.shape)

    def test_config_build_returns_image_learned_embedding(self):
        cfg = self.preset()
        model = cfg.build()

        self.assertIsInstance(model, ImageLearnedPositionalEmbedding)
        self.assertIsInstance(model, cfg._registry_owner())

    def test_init_raises_without_class_token_flag(self):
        cfg = ImageLearnedPositionalEmbeddingConfig(
            num_embeddings=4,
            embedding_dim=6,
            init_size=4,
            padding_idx=0,
            auto_expand_flag=False,
        )

        with self.assertRaises(ValueError):
            ImageLearnedPositionalEmbedding(cfg)


class TestTextSinusoidalPositionalEmbedding(unittest.TestCase):
    def preset(
        self,
        num_embeddings: int = 8,
        embedding_dim: int = 6,
        init_size: int = 8,
        padding_idx: int | None = 0,
        auto_expand_flag: bool = False,
    ) -> TextSinusoidalPositionalEmbeddingConfig:
        return TextSinusoidalPositionalEmbeddingConfig(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            init_size=init_size,
            padding_idx=padding_idx,
            auto_expand_flag=auto_expand_flag,
        )

    def test_init(self):
        cfg = self.preset(num_embeddings=10, embedding_dim=6)
        model = TextSinusoidalPositionalEmbedding(cfg)

        self.assertEqual(model.embedding_dim, cfg.embedding_dim)
        self.assertEqual(model.padding_idx, cfg.padding_idx)
        self.assertEqual(model.num_embeddings, cfg.num_embeddings)
        self.assertEqual(model.init_size, cfg.num_embeddings + cfg.padding_idx + 1)
        self.assertIsInstance(model.weights, torch.Tensor)
        torch.testing.assert_close(
            model.weights[cfg.padding_idx], torch.zeros(cfg.embedding_dim)
        )

    def test_forward(self):
        cfg = self.preset(num_embeddings=10, embedding_dim=6)
        model = TextSinusoidalPositionalEmbedding(cfg)
        input_tokens = torch.randint(1, cfg.num_embeddings, (3, 5))

        output = model(input_tokens)

        self.assertEqual(output.shape, (3, 5, cfg.embedding_dim))

    def test_forward_incremental(self):
        cfg = self.preset(num_embeddings=10, embedding_dim=6)
        model = TextSinusoidalPositionalEmbedding(cfg)
        input_tokens = torch.randint(1, cfg.num_embeddings, (3, 1))

        output = model(input_tokens, incremental_state={})

        self.assertEqual(output.shape, (3, 1, cfg.embedding_dim))

    def test_auto_expand(self):
        cfg = self.preset(num_embeddings=4, embedding_dim=6, auto_expand_flag=True)
        model = TextSinusoidalPositionalEmbedding(cfg)
        input_tokens = torch.ones(2, 10)

        output = model(input_tokens)

        self.assertEqual(output.shape, (2, 10, cfg.embedding_dim))
        self.assertGreaterEqual(model.weights.size(0), 11)

    def test_config_build_returns_text_sinusoidal_embedding(self):
        cfg = self.preset()
        model = cfg.build()

        self.assertIsInstance(model, TextSinusoidalPositionalEmbedding)
        self.assertIsInstance(model, cfg._registry_owner())


class TestImageSinusoidalPositionalEmbedding(unittest.TestCase):
    def preset(
        self,
        num_embeddings: int = 4,
        embedding_dim: int = 6,
        init_size: int = 4,
        padding_idx: int | None = 0,
        auto_expand_flag: bool = False,
        class_token_flag: bool = True,
    ) -> ImageSinusoidalPositionalEmbeddingConfig:
        return ImageSinusoidalPositionalEmbeddingConfig(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            init_size=init_size,
            padding_idx=padding_idx,
            auto_expand_flag=auto_expand_flag,
            class_token_flag=class_token_flag,
        )

    def test_forward(self):
        cfg = self.preset()
        model = ImageSinusoidalPositionalEmbedding(cfg)
        patch_embeddings = torch.randn(2, cfg.num_embeddings + 1, cfg.embedding_dim)

        output = model(patch_embeddings)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, patch_embeddings.shape)

    def test_config_build_returns_image_sinusoidal_embedding(self):
        cfg = self.preset()
        model = cfg.build()

        self.assertIsInstance(model, ImageSinusoidalPositionalEmbedding)
        self.assertIsInstance(model, cfg._registry_owner())

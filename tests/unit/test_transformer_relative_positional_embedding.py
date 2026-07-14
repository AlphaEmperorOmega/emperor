import unittest

import torch
from emperor.embedding.relative import (
    DynamicPositionalBias,
    DynamicPositionalBiasConfig,
    RelativePositionalEmbeddingConfig,
)


class TestRelativePositionalEmbeddingConfig(unittest.TestCase):
    def test_base_config_cannot_build(self):
        cfg = RelativePositionalEmbeddingConfig(
            text_processing_flag=False,
            num_heads=2,
            num_embeddings=16,
            embedding_dim=8,
            init_size=16,
            padding_idx=0,
            auto_expand_flag=False,
            max_positions=8,
        )

        with self.assertRaises(NotImplementedError):
            cfg.build()


class TestDynamicPositionalBias(unittest.TestCase):
    def preset(
        self,
        text_processing_flag: bool = False,
        num_heads: int = 2,
        num_embeddings: int = 16,
        embedding_dim: int = 8,
        init_size: int = 16,
        padding_idx: int | None = 0,
        auto_expand_flag: bool = False,
        max_positions: int = 8,
    ) -> DynamicPositionalBiasConfig:
        return DynamicPositionalBiasConfig(
            text_processing_flag=text_processing_flag,
            num_heads=num_heads,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            init_size=init_size,
            padding_idx=padding_idx,
            auto_expand_flag=auto_expand_flag,
            max_positions=max_positions,
        )

    def test_init(self):
        cfg = self.preset(num_heads=4, embedding_dim=12, max_positions=6)
        model = DynamicPositionalBias(cfg)

        self.assertIsInstance(model, DynamicPositionalBias)
        self.assertEqual(model.embedding_dim, cfg.embedding_dim)
        self.assertEqual(model.num_heads, cfg.num_heads)
        self.assertEqual(model.head_dim, cfg.embedding_dim // cfg.num_heads)
        self.assertEqual(model.max_positions, cfg.max_positions)
        self.assertEqual(
            model.relative_positional_embeddings.shape,
            (cfg.num_heads, model.head_dim, cfg.max_positions * 2 + 1),
        )

    def test_forward(self):
        cfg = self.preset(num_heads=2, embedding_dim=8, max_positions=4)
        model = DynamicPositionalBias(cfg)
        batch_size = 3
        sequence_length = 5
        query = torch.randn(
            batch_size,
            cfg.num_heads,
            sequence_length,
            cfg.embedding_dim // cfg.num_heads,
        )

        output = model(query, sequence_length)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
            output.shape,
            (batch_size, cfg.num_heads, sequence_length, sequence_length),
        )

    def test_forward_last_position(self):
        cfg = self.preset(num_heads=2, embedding_dim=8, max_positions=4)
        model = DynamicPositionalBias(cfg)
        query = torch.randn(3, cfg.num_heads, 1, cfg.embedding_dim // cfg.num_heads)

        output = model(query, sequence_length=5, last=True)

        self.assertEqual(output.shape, (3, cfg.num_heads, 1, 5))

    def test_forward_supports_rectangular_target_and_source_lengths(self):
        cfg = self.preset(num_heads=1, embedding_dim=1, max_positions=4)
        model = DynamicPositionalBias(cfg)
        with torch.no_grad():
            model.relative_positional_embeddings.copy_(
                torch.arange(9, dtype=torch.float32).view(1, 1, 9)
            )
        query = torch.ones(1, 1, 2, 1)

        output = model(query, sequence_length=3)

        expected = torch.tensor([[[[4.0, 5.0, 6.0], [3.0, 4.0, 5.0]]]])
        torch.testing.assert_close(output, expected)

    def test_output_matches_constant_embedding_projection(self):
        cfg = self.preset(num_heads=2, embedding_dim=8, max_positions=4)
        model = DynamicPositionalBias(cfg)
        with torch.no_grad():
            model.relative_positional_embeddings.fill_(1.0)
        query = torch.ones(1, cfg.num_heads, 3, cfg.embedding_dim // cfg.num_heads)

        output = model(query, sequence_length=3)

        expected = torch.full(
            (1, cfg.num_heads, 3, 3),
            float(model.head_dim),
            dtype=query.dtype,
        )
        torch.testing.assert_close(output, expected)

    def test_gradients_flow_through_relative_embeddings(self):
        cfg = self.preset()
        model = DynamicPositionalBias(cfg)
        query = torch.randn(
            2,
            cfg.num_heads,
            4,
            cfg.embedding_dim // cfg.num_heads,
            requires_grad=True,
        )

        output = model(query, sequence_length=4)
        output.sum().backward()

        self.assertIsNotNone(model.relative_positional_embeddings.grad)
        self.assertIsNotNone(query.grad)

    def test_config_build_returns_dynamic_positional_bias(self):
        cfg = self.preset()
        model = cfg.build()

        self.assertIsInstance(model, DynamicPositionalBias)
        self.assertIsInstance(model, DynamicPositionalBias)
        self.assertIsInstance(model, cfg._registry_owner())

    def test_config_build_applies_overrides(self):
        cfg = self.preset(num_heads=2, embedding_dim=8)
        overrides = DynamicPositionalBiasConfig(
            text_processing_flag=True,
            num_heads=4,
            num_embeddings=32,
            embedding_dim=12,
            init_size=32,
            padding_idx=0,
            auto_expand_flag=True,
            max_positions=6,
        )
        model = cfg.build(overrides)

        self.assertEqual(model.text_processing_flag, overrides.text_processing_flag)
        self.assertEqual(model.num_heads, overrides.num_heads)
        self.assertEqual(model.embedding_dim, overrides.embedding_dim)
        self.assertEqual(model.max_positions, overrides.max_positions)

    def test_init_raises_on_missing_or_invalid_fields(self):
        invalid_cases = [
            (
                "missing_num_heads",
                DynamicPositionalBiasConfig(
                    text_processing_flag=False,
                    num_embeddings=16,
                    embedding_dim=8,
                    init_size=16,
                    padding_idx=0,
                    auto_expand_flag=False,
                    max_positions=8,
                ),
            ),
            ("zero_max_positions", self.preset(max_positions=0)),
            ("non_divisible_heads", self.preset(num_heads=3, embedding_dim=8)),
            ("negative_padding_idx", self.preset(padding_idx=-1)),
        ]

        for case, cfg in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    DynamicPositionalBias(cfg)

    def test_forward_raises_on_invalid_input(self):
        model = DynamicPositionalBias(self.preset())

        invalid_cases = [
            ("non_tensor", [1, 2, 3], 4, TypeError),
            ("rank_3", torch.randn(2, 2, 4), 4, ValueError),
            ("zero_sequence_length", torch.randn(2, 2, 4, 4), 0, ValueError),
        ]

        for case, query, sequence_length, error_type in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(error_type):
                    model(query, sequence_length)

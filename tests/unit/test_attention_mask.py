import unittest

import torch
from support.attention import build_attention_config
from emperor.attention.core.config import MultiHeadAttentionConfig
from emperor.attention.core.handlers.mask import Mask


class TestMask(unittest.TestCase):
    def preset(
        self,
        batch_size: int = 2,
        num_heads: int = 2,
        embedding_dim: int = 8,
        target_sequence_length: int = 4,
        source_sequence_length: int = 3,
        target_dtype: torch.dtype = torch.float32,
        causal_attention_mask_flag: bool = False,
        return_attention_weights_flag: bool = False,
    ) -> MultiHeadAttentionConfig:
        cfg = build_attention_config(
            batch_size=batch_size,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            target_sequence_length=target_sequence_length,
            source_sequence_length=source_sequence_length,
            causal_attention_mask_flag=causal_attention_mask_flag,
            return_attention_weights_flag=return_attention_weights_flag,
        )
        cfg.target_dtype = target_dtype
        return cfg

    def key_tensor(self, cfg: MultiHeadAttentionConfig) -> torch.Tensor:
        head_dim = cfg.embedding_dim // cfg.num_heads
        return torch.randn(
            cfg.batch_size * cfg.num_heads,
            cfg.source_sequence_length,
            head_dim,
        )

    def query_tensor(self, cfg: MultiHeadAttentionConfig) -> torch.Tensor:
        return torch.randn(
            cfg.target_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
        )

    def bool_attention_mask(self, cfg: MultiHeadAttentionConfig) -> torch.Tensor:
        mask = torch.zeros(
            cfg.batch_size * cfg.num_heads,
            cfg.target_sequence_length,
            cfg.source_sequence_length,
            dtype=torch.bool,
        )
        mask[:, 0, -1] = True
        mask[:, -1, 0] = True
        return mask

    def float_attention_mask(self, cfg: MultiHeadAttentionConfig) -> torch.Tensor:
        values = torch.arange(
            cfg.batch_size
            * cfg.num_heads
            * cfg.target_sequence_length
            * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        )
        return values.view(
            cfg.batch_size * cfg.num_heads,
            cfg.target_sequence_length,
            cfg.source_sequence_length,
        )

    def bool_key_padding_mask(self, cfg: MultiHeadAttentionConfig) -> torch.Tensor:
        mask = torch.zeros(
            cfg.batch_size,
            cfg.source_sequence_length,
            dtype=torch.bool,
        )
        mask[:, -1] = True
        mask[0, 0] = True
        return mask

    def canonical_bool_mask(
        self,
        cfg: MultiHeadAttentionConfig,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = torch.zeros_like(mask, dtype=cfg.target_dtype)
        return expected.masked_fill(mask, float("-inf"))

    def expanded_key_padding_mask(
        self,
        cfg: MultiHeadAttentionConfig,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return (
            key_padding_mask.view(cfg.batch_size, 1, 1, cfg.source_sequence_length)
            .expand(-1, cfg.num_heads, -1, -1)
            .reshape(
                cfg.batch_size * cfg.num_heads,
                1,
                cfg.source_sequence_length,
            )
        )

    def expected_causal_mask(self, cfg: MultiHeadAttentionConfig) -> torch.Tensor:
        mask = torch.full(
            (cfg.target_sequence_length, cfg.source_sequence_length),
            float("-inf"),
            dtype=cfg.target_dtype,
        )
        return torch.triu(mask, diagonal=1)


class TestMaskInit(TestMask):
    def test_init_stores_config_attributes(self):
        cfg = self.preset()
        model = Mask(cfg)

        self.assertEqual(model.cfg, cfg)
        self.assertEqual(model.batch_size, cfg.batch_size)
        self.assertEqual(model.num_heads, cfg.num_heads)
        self.assertEqual(model.target_dtype, cfg.target_dtype)
        self.assertEqual(
            model.causal_attention_mask_flag,
            cfg.causal_attention_mask_flag,
        )
        self.assertEqual(
            model.return_attention_weights_flag,
            cfg.return_attention_weights_flag,
        )
        self.assertEqual(model.source_sequence_length, cfg.source_sequence_length)
        self.assertEqual(model.target_sequence_length, cfg.target_sequence_length)


class TestResolveCausalAttentionMask(TestMask):
    def test_returns_existing_attention_mask(self):
        cfg = self.preset(causal_attention_mask_flag=True)
        model = Mask(cfg)
        query = self.query_tensor(cfg)
        attention_mask = self.float_attention_mask(cfg)

        output = model.resolve_causal_attention_mask(query, attention_mask)

        self.assertIs(output, attention_mask)

    def test_returns_none_when_causal_mask_is_disabled(self):
        cfg = self.preset(causal_attention_mask_flag=False)
        model = Mask(cfg)
        query = self.query_tensor(cfg)

        output = model.resolve_causal_attention_mask(query, None)

        self.assertIsNone(output)

    def test_generates_rectangular_causal_mask(self):
        cfg = self.preset(
            causal_attention_mask_flag=True,
            target_sequence_length=4,
            source_sequence_length=6,
            target_dtype=torch.float64,
        )
        model = Mask(cfg)
        query = self.query_tensor(cfg)

        output = model.resolve_causal_attention_mask(query, None)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (4, 6))
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.device, query.device)
        torch.testing.assert_close(output, self.expected_causal_mask(cfg))


class TestProcessAttentionMasks(TestMask):
    def test_returns_none_when_masks_are_absent_and_causal_disabled(self):
        cfg = self.preset(
            causal_attention_mask_flag=False,
            return_attention_weights_flag=False,
        )
        model = Mask(cfg)

        key_padding_mask, attention_mask = model.process_attention_masks(None, None)

        self.assertIsNone(key_padding_mask)
        self.assertIsNone(attention_mask)

    def test_canonicalizes_key_padding_mask(self):
        cfg = self.preset()
        model = Mask(cfg)
        key_padding_mask = self.bool_key_padding_mask(cfg)

        output_key_padding_mask, output_attention_mask = (
            model.process_attention_masks(key_padding_mask, None)
        )

        torch.testing.assert_close(
            output_key_padding_mask,
            self.canonical_bool_mask(cfg, key_padding_mask),
        )
        self.assertIsNone(output_attention_mask)

    def test_canonicalizes_attention_mask_to_target_dtype(self):
        cfg = self.preset(target_dtype=torch.float64)
        model = Mask(cfg)
        attention_mask = self.bool_attention_mask(cfg)

        output_key_padding_mask, output_attention_mask = (
            model.process_attention_masks(None, attention_mask)
        )

        self.assertIsNone(output_key_padding_mask)
        self.assertEqual(output_attention_mask.dtype, torch.float64)
        torch.testing.assert_close(
            output_attention_mask,
            self.canonical_bool_mask(cfg, attention_mask),
        )

    def test_canonicalizes_both_masks(self):
        cfg = self.preset()
        model = Mask(cfg)
        key_padding_mask = self.bool_key_padding_mask(cfg)
        attention_mask = self.bool_attention_mask(cfg)

        output_key_padding_mask, output_attention_mask = (
            model.process_attention_masks(key_padding_mask, attention_mask)
        )

        torch.testing.assert_close(
            output_key_padding_mask,
            self.canonical_bool_mask(cfg, key_padding_mask),
        )
        torch.testing.assert_close(
            output_attention_mask,
            self.canonical_bool_mask(cfg, attention_mask),
        )

    def test_canonicalizes_without_mutating_causal_configuration(self):
        cfg = self.preset(
            causal_attention_mask_flag=True,
            return_attention_weights_flag=True,
        )
        model = Mask(cfg)
        key_padding_mask = self.bool_key_padding_mask(cfg)
        attention_mask = self.bool_attention_mask(cfg)

        output_key_padding_mask, output_attention_mask = (
            model.process_attention_masks(key_padding_mask, attention_mask)
        )

        self.assertTrue(model.causal_attention_mask_flag)
        torch.testing.assert_close(
            output_key_padding_mask,
            self.canonical_bool_mask(cfg, key_padding_mask),
        )
        torch.testing.assert_close(
            output_attention_mask,
            self.canonical_bool_mask(cfg, attention_mask),
        )

    def test_rejects_integer_masks(self):
        cfg = self.preset()
        model = Mask(cfg)
        cases = (
            (
                "key_padding_mask",
                torch.ones(
                    cfg.batch_size,
                    cfg.source_sequence_length,
                    dtype=torch.int64,
                ),
                None,
            ),
            (
                "attention_mask",
                None,
                torch.ones(
                    cfg.target_sequence_length,
                    cfg.source_sequence_length,
                    dtype=torch.int64,
                ),
            ),
        )

        for case_name, key_padding_mask, attention_mask in cases:
            with self.subTest(case_name=case_name):
                with self.assertRaisesRegex(RuntimeError, "Only bool and floating"):
                    model.process_attention_masks(
                        key_padding_mask,
                        attention_mask,
                    )

    def test_preserves_float_masks(self):
        cfg = self.preset(
            causal_attention_mask_flag=True,
            return_attention_weights_flag=True,
        )
        model = Mask(cfg)
        key_padding_mask = torch.randn(
            cfg.batch_size,
            cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        )
        attention_mask = self.float_attention_mask(cfg)

        output_key_padding_mask, output_attention_mask = (
            model.process_attention_masks(key_padding_mask, attention_mask)
        )

        self.assertIs(output_key_padding_mask, key_padding_mask)
        self.assertIs(output_attention_mask, attention_mask)

    def test_raises_when_causal_mask_is_required_but_absent(self):
        cases = [
            ("no_fast_path", False, False),
            ("return_weights", True, False),
            ("padding_mask", False, True),
        ]

        for case_name, return_attention_weights_flag, has_key_padding in cases:
            with self.subTest(case_name=case_name):
                cfg = self.preset(
                    causal_attention_mask_flag=True,
                    return_attention_weights_flag=return_attention_weights_flag,
                )
                model = Mask(cfg)
                key_padding_mask = None
                if has_key_padding:
                    key_padding_mask = self.bool_key_padding_mask(cfg)

                with self.assertRaises(RuntimeError):
                    model.process_attention_masks(key_padding_mask, None)

    def test_processes_resolved_causal_attention_mask(self):
        cfg = self.preset(causal_attention_mask_flag=True)
        model = Mask(cfg)
        query = self.query_tensor(cfg)
        attention_mask = model.resolve_causal_attention_mask(query, None)

        key_padding_mask, attention_mask = model.process_attention_masks(
            None,
            attention_mask,
        )

        self.assertIsNone(key_padding_mask)
        torch.testing.assert_close(attention_mask, self.expected_causal_mask(cfg))


class TestMergePaddingAndAttentionMask(TestMask):
    def test_returns_none_when_masks_are_absent(self):
        cfg = self.preset()
        model = Mask(cfg)

        output = model.merge_padding_and_attention_mask(
            self.key_tensor(cfg),
            None,
            None,
        )

        self.assertIsNone(output)

    def test_returns_attention_mask_when_key_padding_mask_is_absent(self):
        cfg = self.preset()
        model = Mask(cfg)
        attention_mask = self.float_attention_mask(cfg)

        output = model.merge_padding_and_attention_mask(
            self.key_tensor(cfg),
            None,
            attention_mask,
        )

        self.assertIs(output, attention_mask)
        torch.testing.assert_close(output, attention_mask)

    def test_expands_key_padding_mask_across_heads(self):
        cfg = self.preset()
        model = Mask(cfg)
        key_padding_mask = self.canonical_bool_mask(
            cfg,
            self.bool_key_padding_mask(cfg),
        )

        output = model.merge_padding_and_attention_mask(
            self.key_tensor(cfg),
            key_padding_mask,
            None,
        )

        expected = self.expanded_key_padding_mask(cfg, key_padding_mask)
        self.assertEqual(
            output.shape,
            (
                cfg.batch_size * cfg.num_heads,
                1,
                cfg.source_sequence_length,
            ),
        )
        torch.testing.assert_close(output, expected)
    def test_adds_key_padding_mask_to_attention_mask(self):
        cfg = self.preset()
        model = Mask(cfg)
        key_padding_mask = self.canonical_bool_mask(
            cfg,
            self.bool_key_padding_mask(cfg),
        )
        attention_mask = self.float_attention_mask(cfg)

        output = model.merge_padding_and_attention_mask(
            self.key_tensor(cfg),
            key_padding_mask,
            attention_mask,
        )

        expected = attention_mask + self.expanded_key_padding_mask(
            cfg,
            key_padding_mask,
        )
        self.assertEqual(
            output.shape,
            (
                cfg.batch_size * cfg.num_heads,
                cfg.target_sequence_length,
                cfg.source_sequence_length,
            ),
        )
        torch.testing.assert_close(output, expected)

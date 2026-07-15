import unittest

import torch
from emperor.attention.core.config import MultiHeadAttentionConfig
from emperor.attention.core.handlers.mask import Mask
from emperor.attention.core.runtime import AttentionMasks, AttentionRuntimeShape

from support.attention import build_attention_config


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

    def runtime_shape(self, cfg: MultiHeadAttentionConfig) -> AttentionRuntimeShape:
        return AttentionRuntimeShape(
            batch_size=cfg.batch_size,
            target_sequence_length=cfg.target_sequence_length,
            source_sequence_length=cfg.source_sequence_length,
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


class TestMaskShapeValidation(TestMask):
    def test_rejects_each_invalid_mask_shape_contract(self):
        cfg = self.preset()
        model = Mask(cfg)
        cases = (
            (
                "key_padding_mask",
                torch.zeros(cfg.batch_size, cfg.source_sequence_length + 1),
                None,
                "key_padding_mask must have shape",
            ),
            (
                "sequence_dimensions",
                None,
                torch.zeros(
                    cfg.target_sequence_length,
                    cfg.source_sequence_length + 1,
                ),
                "target/source dimensions",
            ),
            (
                "leading_dimension",
                None,
                torch.zeros(
                    2,
                    cfg.target_sequence_length,
                    cfg.source_sequence_length,
                ),
                "leading dimension must be 1 or",
            ),
        )

        for name, padding_mask, attention_mask, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, message):
                    model.prepare_attention_masks(
                        self.query_tensor(cfg),
                        AttentionMasks(
                            key_padding_mask=padding_mask,
                            attention_mask=attention_mask,
                        ),
                        self.runtime_shape(cfg),
                    )

    def test_prepare_rejects_each_invalid_mask_shape_with_exact_message(self):
        cfg = self.preset()
        model = Mask(cfg)
        cases = (
            (
                "key_padding_mask",
                AttentionMasks(
                    key_padding_mask=torch.zeros(
                        cfg.batch_size,
                        cfg.source_sequence_length + 1,
                    )
                ),
                "key_padding_mask must have shape (2, 3), got (2, 4).",
            ),
            (
                "sequence_dimensions",
                AttentionMasks(
                    attention_mask=torch.zeros(
                        1,
                        cfg.target_sequence_length,
                        cfg.source_sequence_length + 1,
                    )
                ),
                "attention_mask must have target/source dimensions (4, 3), got (4, 4).",
            ),
            (
                "leading_dimension",
                AttentionMasks(
                    attention_mask=torch.zeros(
                        2,
                        cfg.target_sequence_length,
                        cfg.source_sequence_length,
                    )
                ),
                "3-D attention_mask leading dimension must be 1 or batch_size * "
                "num_heads (4), got 2.",
            ),
            (
                "rank",
                AttentionMasks(
                    attention_mask=torch.zeros(
                        1,
                        1,
                        cfg.target_sequence_length,
                        cfg.source_sequence_length,
                    )
                ),
                "attention_mask must be 2-D or 3-D, got 4-D.",
            ),
        )

        for name, masks, message in cases:
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError) as caught:
                    model.prepare_attention_masks(
                        self.query_tensor(cfg),
                        masks,
                        self.runtime_shape(cfg),
                    )
                self.assertEqual(str(caught.exception), message)

    def test_prepare_accepts_only_standard_attention_mask_branches(self):
        cfg = self.preset()
        model = Mask(cfg)
        runtime_shape = self.runtime_shape(cfg)
        sequence_shape = (
            cfg.target_sequence_length,
            cfg.source_sequence_length,
        )
        standard_branch_count = cfg.batch_size * cfg.num_heads
        accepted_masks = (
            torch.zeros(*sequence_shape),
            torch.zeros(1, *sequence_shape),
            torch.zeros(standard_branch_count, *sequence_shape),
        )

        for attention_mask in accepted_masks:
            with self.subTest(shape=tuple(attention_mask.shape)):
                prepared = model.prepare_attention_masks(
                    self.query_tensor(cfg),
                    AttentionMasks(attention_mask=attention_mask),
                    runtime_shape,
                )
                self.assertIs(prepared.attention_mask, attention_mask)

        clean_multiple = torch.zeros(
            standard_branch_count * 2,
            *sequence_shape,
        )
        with self.assertRaises(RuntimeError) as caught:
            model.prepare_attention_masks(
                self.query_tensor(cfg),
                AttentionMasks(attention_mask=clean_multiple),
                runtime_shape,
            )
        self.assertEqual(
            str(caught.exception),
            "3-D attention_mask leading dimension must be 1 or batch_size * "
            "num_heads (4), got 8.",
        )


class TestPrepareAttentionMasks(TestMask):
    def test_preserves_existing_attention_mask(self):
        cfg = self.preset(causal_attention_mask_flag=True)
        model = Mask(cfg)
        query = self.query_tensor(cfg)
        attention_mask = self.float_attention_mask(cfg)

        input_masks = AttentionMasks(attention_mask=attention_mask)
        output = model.prepare_attention_masks(
            query,
            input_masks,
            self.runtime_shape(cfg),
        )

        self.assertIs(output, input_masks)
        self.assertIs(output.attention_mask, attention_mask)

    def test_returns_original_masks_when_absent_and_causal_disabled(self):
        cfg = self.preset(causal_attention_mask_flag=False)
        model = Mask(cfg)
        query = self.query_tensor(cfg)
        input_masks = AttentionMasks()

        output = model.prepare_attention_masks(
            query,
            input_masks,
            self.runtime_shape(cfg),
        )

        self.assertIs(output, input_masks)
        self.assertIsNone(output.key_padding_mask)
        self.assertIsNone(output.attention_mask)

    def test_generates_rectangular_causal_mask(self):
        cfg = self.preset(
            causal_attention_mask_flag=True,
            target_sequence_length=4,
            source_sequence_length=6,
            target_dtype=torch.float64,
        )
        model = Mask(cfg)
        query = self.query_tensor(cfg)

        output = model.prepare_attention_masks(
            query,
            AttentionMasks(),
            self.runtime_shape(cfg),
        )

        self.assertIsInstance(output, AttentionMasks)
        self.assertEqual(output.attention_mask.shape, (4, 6))
        self.assertEqual(output.attention_mask.dtype, torch.float64)
        self.assertEqual(output.attention_mask.device, query.device)
        torch.testing.assert_close(
            output.attention_mask,
            self.expected_causal_mask(cfg),
        )

    def test_generates_causal_mask_on_query_device(self):
        cfg = self.preset(causal_attention_mask_flag=True)
        model = Mask(cfg)
        query = torch.empty(
            cfg.target_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
            device="meta",
        )

        output = model.prepare_attention_masks(
            query,
            AttentionMasks(),
            self.runtime_shape(cfg),
        )

        self.assertEqual(output.attention_mask.device, query.device)
        self.assertEqual(
            output.attention_mask.shape,
            (cfg.target_sequence_length, cfg.source_sequence_length),
        )


class TestPrepareAttentionMaskCanonicalization(TestMask):
    def test_returns_none_when_masks_are_absent_and_causal_disabled(self):
        cfg = self.preset(
            causal_attention_mask_flag=False,
            return_attention_weights_flag=False,
        )
        model = Mask(cfg)

        output = model.prepare_attention_masks(
            self.query_tensor(cfg),
            AttentionMasks(),
            self.runtime_shape(cfg),
        )

        self.assertIsNone(output.key_padding_mask)
        self.assertIsNone(output.attention_mask)

    def test_canonicalizes_key_padding_mask(self):
        cfg = self.preset()
        model = Mask(cfg)
        key_padding_mask = self.bool_key_padding_mask(cfg)

        output = model.prepare_attention_masks(
            self.query_tensor(cfg),
            AttentionMasks(key_padding_mask=key_padding_mask),
            self.runtime_shape(cfg),
        )

        torch.testing.assert_close(
            output.key_padding_mask,
            self.canonical_bool_mask(cfg, key_padding_mask),
        )
        self.assertIsNone(output.attention_mask)

    def test_canonicalizes_attention_mask_to_target_dtype(self):
        cfg = self.preset(target_dtype=torch.float64)
        model = Mask(cfg)
        attention_mask = self.bool_attention_mask(cfg)

        output = model.prepare_attention_masks(
            self.query_tensor(cfg),
            AttentionMasks(attention_mask=attention_mask),
            self.runtime_shape(cfg),
        )

        self.assertIsNone(output.key_padding_mask)
        self.assertEqual(output.attention_mask.dtype, torch.float64)
        torch.testing.assert_close(
            output.attention_mask,
            self.canonical_bool_mask(cfg, attention_mask),
        )

    def test_canonicalizes_both_masks(self):
        cfg = self.preset()
        model = Mask(cfg)
        key_padding_mask = self.bool_key_padding_mask(cfg)
        attention_mask = self.bool_attention_mask(cfg)

        output = model.prepare_attention_masks(
            self.query_tensor(cfg),
            AttentionMasks(
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            ),
            self.runtime_shape(cfg),
        )

        torch.testing.assert_close(
            output.key_padding_mask,
            self.canonical_bool_mask(cfg, key_padding_mask),
        )
        torch.testing.assert_close(
            output.attention_mask,
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

        output = model.prepare_attention_masks(
            self.query_tensor(cfg),
            AttentionMasks(
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            ),
            self.runtime_shape(cfg),
        )

        self.assertTrue(model.causal_attention_mask_flag)
        torch.testing.assert_close(
            output.key_padding_mask,
            self.canonical_bool_mask(cfg, key_padding_mask),
        )
        torch.testing.assert_close(
            output.attention_mask,
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
                with self.assertRaises(RuntimeError) as caught:
                    model.prepare_attention_masks(
                        self.query_tensor(cfg),
                        AttentionMasks(
                            key_padding_mask=key_padding_mask,
                            attention_mask=attention_mask,
                        ),
                        self.runtime_shape(cfg),
                    )
                self.assertEqual(
                    str(caught.exception),
                    f"Only bool and floating types of {case_name} are supported.",
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

        input_masks = AttentionMasks(
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )
        output = model.prepare_attention_masks(
            self.query_tensor(cfg),
            input_masks,
            self.runtime_shape(cfg),
        )

        self.assertIs(output, input_masks)
        self.assertIs(output.key_padding_mask, key_padding_mask)
        self.assertIs(output.attention_mask, attention_mask)

    def test_generates_causal_mask_for_all_runtime_paths(self):
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

                output = model.prepare_attention_masks(
                    self.query_tensor(cfg),
                    AttentionMasks(key_padding_mask=key_padding_mask),
                    self.runtime_shape(cfg),
                )

                torch.testing.assert_close(
                    output.attention_mask,
                    self.expected_causal_mask(cfg),
                )

    def test_processes_generated_causal_attention_mask(self):
        cfg = self.preset(causal_attention_mask_flag=True)
        model = Mask(cfg)
        query = self.query_tensor(cfg)

        output = model.prepare_attention_masks(
            query,
            AttentionMasks(),
            self.runtime_shape(cfg),
        )

        self.assertIsNone(output.key_padding_mask)
        torch.testing.assert_close(
            output.attention_mask,
            self.expected_causal_mask(cfg),
        )

class TestMergePaddingAndAttentionMask(TestMask):
    def test_returns_none_when_masks_are_absent(self):
        cfg = self.preset()
        model = Mask(cfg)

        output = model.merge_padding_and_attention_mask(
            self.key_tensor(cfg),
            AttentionMasks(),
            self.runtime_shape(cfg),
        )

        self.assertIsNone(output)

    def test_returns_attention_mask_when_key_padding_mask_is_absent(self):
        cfg = self.preset()
        model = Mask(cfg)
        attention_mask = self.float_attention_mask(cfg)

        output = model.merge_padding_and_attention_mask(
            self.key_tensor(cfg),
            AttentionMasks(attention_mask=attention_mask),
            self.runtime_shape(cfg),
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
            AttentionMasks(key_padding_mask=key_padding_mask),
            self.runtime_shape(cfg),
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
            AttentionMasks(
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            ),
            self.runtime_shape(cfg),
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

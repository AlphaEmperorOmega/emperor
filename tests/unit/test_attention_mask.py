import unittest

import torch

from emperor.attention import MultiHeadAttentionConfig
from emperor.attention._ops.masking import Mask
from emperor.attention._runtime import (
    AttentionRuntimeLayout,
    MultiHeadAttentionInputs,
)
from emperor.attention._validation import AttentionValidatorBase
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
            dtype=cfg.target_dtype,
        )

    def runtime_layout(self, cfg: MultiHeadAttentionConfig) -> AttentionRuntimeLayout:
        return AttentionRuntimeLayout(
            batch_size=cfg.batch_size,
            target_sequence_length=cfg.target_sequence_length,
            source_sequence_length=cfg.source_sequence_length,
        )

    def attention_inputs(
        self,
        cfg: MultiHeadAttentionConfig,
        *,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        runtime_layout: AttentionRuntimeLayout | None = None,
    ) -> MultiHeadAttentionInputs:
        query = self.query_tensor(cfg) if query is None else query
        key = self.key_tensor(cfg) if key is None else key
        return MultiHeadAttentionInputs(
            query=query,
            key=key,
            value=key,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
            runtime_layout=runtime_layout or self.runtime_layout(cfg),
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

    def unresolved_attention_inputs(
        self,
        cfg: MultiHeadAttentionConfig,
    ) -> MultiHeadAttentionInputs:
        query = self.query_tensor(cfg)
        key = self.key_tensor(cfg)
        return MultiHeadAttentionInputs(query=query, key=key, value=key)

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
        self.assertIsNone(model.query_dtype)
        self.assertIsNone(model.query_device)
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
                        self.attention_inputs(
                            cfg,
                            key_padding_mask=padding_mask,
                            attention_mask=attention_mask,
                        )
                    )

    def test_prepare_rejects_each_invalid_mask_shape_with_exact_message(self):
        cfg = self.preset()
        model = Mask(cfg)
        cases = (
            (
                "key_padding_mask",
                torch.zeros(
                    cfg.batch_size,
                    cfg.source_sequence_length + 1,
                ),
                None,
                "key_padding_mask must have shape (2, 3), got (2, 4).",
            ),
            (
                "sequence_dimensions",
                None,
                torch.zeros(
                    1,
                    cfg.target_sequence_length,
                    cfg.source_sequence_length + 1,
                ),
                "attention_mask must have target/source dimensions (4, 3), got (4, 4).",
            ),
            (
                "leading_dimension",
                None,
                torch.zeros(
                    2,
                    cfg.target_sequence_length,
                    cfg.source_sequence_length,
                ),
                "3-D attention_mask leading dimension must be 1 or batch_size * "
                "num_heads (4), got 2.",
            ),
            (
                "rank",
                None,
                torch.zeros(
                    1,
                    1,
                    cfg.target_sequence_length,
                    cfg.source_sequence_length,
                ),
                "attention_mask must be 2-D or 3-D, got 4-D.",
            ),
        )

        for name, key_padding_mask, attention_mask, message in cases:
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError) as caught:
                    model.prepare_attention_masks(
                        self.attention_inputs(
                            cfg,
                            key_padding_mask=key_padding_mask,
                            attention_mask=attention_mask,
                        )
                    )
                self.assertEqual(str(caught.exception), message)

    def test_prepare_accepts_only_standard_attention_mask_branches(self):
        cfg = self.preset()
        model = Mask(cfg)
        runtime_layout = self.runtime_layout(cfg)
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
                    self.attention_inputs(
                        cfg,
                        attention_mask=attention_mask,
                        runtime_layout=runtime_layout,
                    )
                )
                self.assertIs(prepared.attention_mask, attention_mask)

        clean_multiple = torch.zeros(
            standard_branch_count * 2,
            *sequence_shape,
        )
        with self.assertRaises(RuntimeError) as caught:
            model.prepare_attention_masks(
                self.attention_inputs(
                    cfg,
                    attention_mask=clean_multiple,
                    runtime_layout=runtime_layout,
                )
            )
        self.assertEqual(
            str(caught.exception),
            "3-D attention_mask leading dimension must be 1 or batch_size * "
            "num_heads (4), got 8.",
        )


class TestPrepareAttentionMasks(TestMask):
    def test_requires_resolved_runtime_layout(self):
        cfg = self.preset()
        model = Mask(cfg)

        with self.assertRaises(RuntimeError) as caught:
            model.prepare_attention_masks(self.unresolved_attention_inputs(cfg))

        self.assertEqual(
            str(caught.exception),
            "Attention mask preparation requires resolved runtime layout.",
        )

    def test_runtime_layout_validation_dispatches_through_subclass(self):
        class RejectingValidator(AttentionValidatorBase):
            @staticmethod
            def validate_attention_mask_preparation_runtime_layout(*args, **kwargs):
                raise RuntimeError("substituted mask-preparation validator was called")

        class RejectingMask(Mask):
            VALIDATOR = RejectingValidator

        cfg = self.preset()
        model = RejectingMask(cfg)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted mask-preparation validator was called",
        ):
            model.prepare_attention_masks(self.unresolved_attention_inputs(cfg))

    def test_causal_mask_composes_with_existing_attention_mask(self):
        cfg = self.preset(causal_attention_mask_flag=True)
        model = Mask(cfg)
        query = self.query_tensor(cfg)
        attention_mask = self.float_attention_mask(cfg)
        original_attention_mask = attention_mask.clone()

        input_values = self.attention_inputs(
            cfg,
            query=query,
            attention_mask=attention_mask,
        )
        output = model.prepare_attention_masks(input_values)

        expected = attention_mask.masked_fill(
            self.expected_causal_mask(cfg).isneginf(),
            -torch.inf,
        )
        torch.testing.assert_close(output.attention_mask, expected)
        torch.testing.assert_close(attention_mask, original_attention_mask)

    def test_causal_overlay_dominates_positive_infinity_without_nan(self):
        cfg = self.preset(causal_attention_mask_flag=True)
        model = Mask(cfg)
        attention_mask = torch.arange(
            cfg.target_sequence_length * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).reshape(cfg.target_sequence_length, cfg.source_sequence_length)
        future_positions = self.expected_causal_mask(cfg).isneginf()
        attention_mask[future_positions] = torch.inf
        original_attention_mask = attention_mask.clone()

        output = model.prepare_attention_masks(
            self.attention_inputs(cfg, attention_mask=attention_mask)
        )

        expected = attention_mask.masked_fill(future_positions, -torch.inf)
        torch.testing.assert_close(output.attention_mask, expected)
        self.assertFalse(torch.isnan(output.attention_mask).any())
        torch.testing.assert_close(attention_mask, original_attention_mask)

    def test_returns_original_masks_when_absent_and_causal_disabled(self):
        cfg = self.preset(causal_attention_mask_flag=False)
        model = Mask(cfg)
        query = self.query_tensor(cfg)

        input_values = self.attention_inputs(
            cfg,
            query=query,
        )
        output = model.prepare_attention_masks(input_values)

        self.assertIs(output, input_values)
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

        output = model.prepare_attention_masks(self.attention_inputs(cfg, query=query))

        self.assertIsInstance(output, MultiHeadAttentionInputs)
        self.assertEqual(output.attention_mask.shape, (4, 6))
        self.assertEqual(output.attention_mask.dtype, torch.float64)
        self.assertEqual(output.attention_mask.device, query.device)
        torch.testing.assert_close(
            output.attention_mask,
            self.expected_causal_mask(cfg),
        )

    def test_composes_causal_mask_from_smaller_rectangular_runtime_lengths(self):
        cfg = self.preset(
            causal_attention_mask_flag=True,
            target_sequence_length=6,
            source_sequence_length=7,
            target_dtype=torch.float64,
        )
        model = Mask(cfg)
        runtime_layout = AttentionRuntimeLayout(
            batch_size=cfg.batch_size,
            target_sequence_length=3,
            source_sequence_length=5,
        )
        query = torch.randn(3, cfg.batch_size, cfg.embedding_dim, dtype=torch.float64)
        key = torch.randn(
            cfg.batch_size * cfg.num_heads,
            5,
            cfg.embedding_dim // cfg.num_heads,
            dtype=torch.float64,
        )
        attention_mask = torch.arange(15, dtype=torch.float64).reshape(3, 5)

        output = model.prepare_attention_masks(
            self.attention_inputs(
                cfg,
                query=query,
                key=key,
                attention_mask=attention_mask,
                runtime_layout=runtime_layout,
            )
        )

        causal_positions = torch.triu(
            torch.ones(3, 5, dtype=torch.bool),
            diagonal=1,
        )
        expected = attention_mask.masked_fill(causal_positions, -torch.inf)
        torch.testing.assert_close(output.attention_mask, expected)

    def test_generates_causal_mask_on_query_device(self):
        cfg = self.preset(causal_attention_mask_flag=True)
        model = Mask(cfg)
        query = torch.empty(
            cfg.target_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
            device="meta",
        )

        output = model.prepare_attention_masks(self.attention_inputs(cfg, query=query))

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

        output = model.prepare_attention_masks(self.attention_inputs(cfg))

        self.assertIsNone(output.key_padding_mask)
        self.assertIsNone(output.attention_mask)

    def test_canonicalizes_key_padding_mask(self):
        cfg = self.preset()
        model = Mask(cfg)
        key_padding_mask = self.bool_key_padding_mask(cfg)

        output = model.prepare_attention_masks(
            self.attention_inputs(
                cfg,
                key_padding_mask=key_padding_mask,
            )
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
            self.attention_inputs(
                cfg,
                attention_mask=attention_mask,
            )
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
            self.attention_inputs(
                cfg,
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            )
        )

        torch.testing.assert_close(
            output.key_padding_mask,
            self.canonical_bool_mask(cfg, key_padding_mask),
        )
        torch.testing.assert_close(
            output.attention_mask,
            self.canonical_bool_mask(cfg, attention_mask),
        )

    def test_moves_supplied_masks_to_the_runtime_query_device(self):
        cfg = self.preset(target_dtype=torch.float64)
        model = Mask(cfg)
        query = torch.empty(
            cfg.target_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
            dtype=torch.float64,
            device="meta",
        )
        key_padding_mask = self.bool_key_padding_mask(cfg)
        attention_mask = self.float_attention_mask(cfg).to(dtype=torch.float32)

        output = model.prepare_attention_masks(
            self.attention_inputs(
                cfg,
                query=query,
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            )
        )

        self.assertEqual(output.key_padding_mask.device, query.device)
        self.assertEqual(output.attention_mask.device, query.device)
        self.assertEqual(output.key_padding_mask.dtype, query.dtype)
        self.assertEqual(output.attention_mask.dtype, query.dtype)
        self.assertEqual(
            tuple(output.key_padding_mask.shape),
            tuple(key_padding_mask.shape),
        )
        self.assertEqual(
            tuple(output.attention_mask.shape),
            tuple(attention_mask.shape),
        )
        self.assertIsNone(model.query_dtype)
        self.assertIsNone(model.query_device)

    def test_canonicalizes_boolean_mask_before_causal_composition(self):
        cfg = self.preset(
            causal_attention_mask_flag=True,
            return_attention_weights_flag=True,
        )
        model = Mask(cfg)
        key_padding_mask = self.bool_key_padding_mask(cfg)
        attention_mask = self.bool_attention_mask(cfg)

        output = model.prepare_attention_masks(
            self.attention_inputs(
                cfg,
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            )
        )

        self.assertTrue(model.causal_attention_mask_flag)
        torch.testing.assert_close(
            output.key_padding_mask,
            self.canonical_bool_mask(cfg, key_padding_mask),
        )
        expected_attention_mask = self.canonical_bool_mask(
            cfg,
            attention_mask,
        ).masked_fill(
            self.expected_causal_mask(cfg).isneginf(),
            -torch.inf,
        )
        torch.testing.assert_close(
            output.attention_mask,
            expected_attention_mask,
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
                        self.attention_inputs(
                            cfg,
                            key_padding_mask=key_padding_mask,
                            attention_mask=attention_mask,
                        )
                    )
                self.assertEqual(
                    str(caught.exception),
                    f"Only bool and floating types of {case_name} are supported.",
                )

    def test_preserves_float_mask_identity_when_causality_is_disabled(self):
        cfg = self.preset(
            causal_attention_mask_flag=False,
            return_attention_weights_flag=True,
        )
        model = Mask(cfg)
        key_padding_mask = torch.randn(
            cfg.batch_size,
            cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        )
        attention_mask = self.float_attention_mask(cfg)

        input_values = self.attention_inputs(
            cfg,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )
        output = model.prepare_attention_masks(input_values)

        self.assertIs(output, input_values)
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
                    self.attention_inputs(
                        cfg,
                        key_padding_mask=key_padding_mask,
                    )
                )

                torch.testing.assert_close(
                    output.attention_mask,
                    self.expected_causal_mask(cfg),
                )

    def test_processes_generated_causal_attention_mask(self):
        cfg = self.preset(causal_attention_mask_flag=True)
        model = Mask(cfg)
        query = self.query_tensor(cfg)

        output = model.prepare_attention_masks(self.attention_inputs(cfg, query=query))

        self.assertIsNone(output.key_padding_mask)
        torch.testing.assert_close(
            output.attention_mask,
            self.expected_causal_mask(cfg),
        )


class TestMergePaddingAndAttentionMask(TestMask):
    def test_requires_resolved_runtime_layout(self):
        cfg = self.preset()
        model = Mask(cfg)

        with self.assertRaisesRegex(
            RuntimeError,
            "Attention mask merging requires resolved runtime layout.",
        ):
            model.merge_padding_and_attention_mask(
                self.unresolved_attention_inputs(cfg)
            )

    def test_runtime_layout_validation_dispatches_through_subclass(self):
        class RejectingValidator(AttentionValidatorBase):
            @staticmethod
            def validate_attention_mask_merging_runtime_layout(*args, **kwargs):
                raise RuntimeError("substituted mask-merging validator was called")

        class RejectingMask(Mask):
            VALIDATOR = RejectingValidator

        cfg = self.preset()
        model = RejectingMask(cfg)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted mask-merging validator was called",
        ):
            model.merge_padding_and_attention_mask(
                self.unresolved_attention_inputs(cfg)
            )

    def test_returns_none_when_masks_are_absent(self):
        cfg = self.preset()
        model = Mask(cfg)

        output = model.merge_padding_and_attention_mask(
            self.attention_inputs(cfg)
        ).merged_attention_mask

        self.assertIsNone(output)

    def test_returns_attention_mask_when_key_padding_mask_is_absent(self):
        cfg = self.preset()
        model = Mask(cfg)
        attention_mask = self.float_attention_mask(cfg)

        output = model.merge_padding_and_attention_mask(
            self.attention_inputs(
                cfg,
                attention_mask=attention_mask,
            )
        ).merged_attention_mask

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
            self.attention_inputs(
                cfg,
                key_padding_mask=key_padding_mask,
            )
        ).merged_attention_mask

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
            self.attention_inputs(
                cfg,
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            )
        ).merged_attention_mask

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

    def test_causal_explicit_and_key_padding_masks_merge_exactly(self):
        cfg = self.preset(causal_attention_mask_flag=True)
        model = Mask(cfg)
        key_padding_mask = self.bool_key_padding_mask(cfg)
        attention_mask = torch.arange(
            cfg.target_sequence_length * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).reshape(cfg.target_sequence_length, cfg.source_sequence_length)
        attention_inputs = self.attention_inputs(
            cfg,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )

        prepared = model.prepare_attention_masks(attention_inputs)
        output = model.merge_padding_and_attention_mask(
            prepared
        ).merged_attention_mask

        causal_attention_mask = attention_mask.masked_fill(
            self.expected_causal_mask(cfg).isneginf(),
            -torch.inf,
        )
        canonical_key_padding_mask = self.canonical_bool_mask(
            cfg,
            key_padding_mask,
        )
        expected = causal_attention_mask + self.expanded_key_padding_mask(
            cfg,
            canonical_key_padding_mask,
        )
        torch.testing.assert_close(output, expected)

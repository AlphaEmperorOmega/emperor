import unittest
from dataclasses import asdict

import torch

from emperor.attention import MultiHeadAttentionConfig
from emperor.attention._ops.batching import BatchDimensionManager
from emperor.attention._runtime import MultiHeadAttentionInputs
from emperor.attention._validation import AttentionValidatorBase
from support.attention import build_attention_config


class TestBatchDimensionManager(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.target_sequence_length = None

    def rebuild_presets(self, config: MultiHeadAttentionConfig | None = None):
        self.config = build_attention_config(
            embedding_dim=12,
            query_key_projection_dim=12,
            value_projection_dim=12,
            source_sequence_length=8,
            target_sequence_length=8,
        )
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = BatchDimensionManager(self.config)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length

    @staticmethod
    def attention_inputs(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **runtime_values,
    ) -> MultiHeadAttentionInputs:
        return MultiHeadAttentionInputs(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
            **runtime_values,
        )


class TestConvertInputsToInternalLayout(TestBatchDimensionManager):
    def test_sequence_first_batched_inputs_preserve_value_objects(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=False))
        query = torch.randn(5, 3, 12)
        key = torch.randn(7, 3, 12)
        value = torch.randn(7, 3, 12)

        output_inputs = self.model.convert_inputs_to_internal_layout(
            self.attention_inputs(query, key, value)
        )

        self.assertIs(output_inputs.query, query)
        self.assertIs(output_inputs.key, key)
        self.assertIs(output_inputs.value, value)
        self.assertIsNone(output_inputs.key_padding_mask)
        self.assertIsNone(output_inputs.attention_mask)
        runtime_layout = output_inputs.runtime_layout
        self.assertEqual(runtime_layout.batch_size, 3)
        self.assertEqual(runtime_layout.target_sequence_length, 5)
        self.assertEqual(runtime_layout.source_sequence_length, 7)

    def test_static_keys_define_runtime_source_sequence_length(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=False))
        query = torch.randn(5, 3, 12)
        key = torch.randn(7, 3, 12)
        value = torch.randn(7, 3, 12)
        static_keys = torch.randn(6, 4, 2)

        output_inputs = self.model.convert_inputs_to_internal_layout(
            self.attention_inputs(
                query,
                key,
                value,
                static_key=static_keys,
            )
        )

        self.assertIs(output_inputs.key, key)
        self.assertEqual(output_inputs.runtime_layout.source_sequence_length, 4)

    def test_distinct_unbatched_qkv_without_masks_are_all_expanded(self):
        query = torch.randn(self.target_sequence_length, self.embedding_dim)
        key = torch.randn(self.source_sequence_length, self.embedding_dim)
        value = torch.randn(self.source_sequence_length, self.embedding_dim)

        output_inputs = self.model.convert_inputs_to_internal_layout(
            self.attention_inputs(query, key, value)
        )

        self.assertEqual(output_inputs.query.shape, (8, 1, 12))
        self.assertEqual(output_inputs.key.shape, (8, 1, 12))
        self.assertEqual(output_inputs.value.shape, (8, 1, 12))
        self.assertIsNone(output_inputs.key_padding_mask)
        self.assertFalse(output_inputs.runtime_layout.input_was_batched)

    def test_shared_qkv_input_tensors_preserve_identity(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=True))
        query = torch.randn(
            self.batch_size,
            self.target_sequence_length,
            self.embedding_dim,
        )
        output_inputs = self.model.convert_inputs_to_internal_layout(
            self.attention_inputs(query, query, query)
        )

        expected_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        self.assertIsNone(output_inputs.key_padding_mask)
        self.assertIsNone(output_inputs.attention_mask)
        self.assertEqual(output_inputs.query.shape, expected_output_shape)
        self.assertEqual(output_inputs.key.shape, expected_output_shape)
        self.assertEqual(output_inputs.value.shape, expected_output_shape)
        self.assertIs(output_inputs.query, output_inputs.key)
        self.assertIs(output_inputs.key, output_inputs.value)
        torch.testing.assert_close(output_inputs.query, query.transpose(0, 1))
        self.assertTrue(output_inputs.runtime_layout.input_was_batch_first)

    def test_shared_key_value_input_tensors_preserve_identity(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=True))
        query = torch.randn(
            self.batch_size,
            self.target_sequence_length,
            self.embedding_dim,
        )
        key = torch.randn(
            self.batch_size,
            self.source_sequence_length,
            self.embedding_dim,
        )

        output_inputs = self.model.convert_inputs_to_internal_layout(
            self.attention_inputs(query, key, key)
        )

        expected_q_shape = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        expected_kv_shape = (
            self.source_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )

        self.assertIsNone(output_inputs.key_padding_mask)
        self.assertIsNone(output_inputs.attention_mask)
        self.assertEqual(output_inputs.query.shape, expected_q_shape)
        self.assertEqual(output_inputs.key.shape, expected_kv_shape)
        self.assertEqual(output_inputs.value.shape, expected_kv_shape)
        self.assertIs(output_inputs.key, output_inputs.value)
        torch.testing.assert_close(output_inputs.query, query.transpose(0, 1))
        torch.testing.assert_close(output_inputs.key, key.transpose(0, 1))
        self.assertTrue(output_inputs.runtime_layout.input_was_batch_first)

    def test_distinct_batch_first_qkv_are_all_transposed(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=True))
        query = torch.randn(self.batch_size, 5, self.embedding_dim)
        key = torch.randn(self.batch_size, 7, self.embedding_dim)
        value = torch.randn(self.batch_size, 7, self.embedding_dim)

        output_inputs = self.model.convert_inputs_to_internal_layout(
            self.attention_inputs(query, key, value)
        )

        self.assertEqual(output_inputs.query.shape, (5, self.batch_size, 12))
        self.assertEqual(output_inputs.key.shape, (7, self.batch_size, 12))
        self.assertEqual(output_inputs.value.shape, (7, self.batch_size, 12))
        torch.testing.assert_close(output_inputs.query, query.transpose(0, 1))
        torch.testing.assert_close(output_inputs.key, key.transpose(0, 1))
        torch.testing.assert_close(output_inputs.value, value.transpose(0, 1))
        self.assertEqual(output_inputs.runtime_layout.target_sequence_length, 5)
        self.assertEqual(output_inputs.runtime_layout.source_sequence_length, 7)
        self.assertTrue(output_inputs.runtime_layout.input_was_batch_first)

    def test_unbatched_padding_mask_gains_batch_dimension(self):
        query = torch.randn(self.target_sequence_length, self.embedding_dim)
        key = value = torch.randn(
            self.source_sequence_length,
            self.embedding_dim,
        )
        key_padding_mask = torch.zeros(
            self.source_sequence_length,
            dtype=torch.bool,
        )
        attention_mask = torch.zeros(
            self.target_sequence_length,
            self.source_sequence_length,
            dtype=torch.bool,
        )

        output_inputs = self.model.convert_inputs_to_internal_layout(
            self.attention_inputs(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            )
        )

        self.assertEqual(
            output_inputs.query.shape, (self.target_sequence_length, 1, 12)
        )
        self.assertIs(output_inputs.key, output_inputs.value)
        self.assertEqual(output_inputs.key_padding_mask.shape, (1, 8))
        self.assertIs(output_inputs.attention_mask, attention_mask)
        self.assertFalse(output_inputs.runtime_layout.input_was_batched)

    def test_every_shared_tensor_pair_preserves_identity_when_unsqueezed(self):
        first = torch.randn(8, 12)
        second = torch.randn(8, 12)
        cases = (
            ("query_key", (first, first, second), (0, 1)),
            ("query_value", (first, second, first), (0, 2)),
            ("key_value", (second, first, first), (1, 2)),
        )

        for name, tensors, shared_indices in cases:
            with self.subTest(name=name):
                output_inputs = self.model.convert_inputs_to_internal_layout(
                    self.attention_inputs(*tensors)
                )
                tensors = (
                    output_inputs.query,
                    output_inputs.key,
                    output_inputs.value,
                )
                self.assertIs(
                    tensors[shared_indices[0]],
                    tensors[shared_indices[1]],
                )
                self.assertEqual(tensors[0].shape, (8, 1, 12))

    def test_every_shared_tensor_pair_preserves_identity_when_transposed(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=True))
        first = torch.randn(3, 5, 12)
        second = torch.randn(3, 5, 12)
        cases = (
            ("query_key", (first, first, second), (0, 1)),
            ("query_value", (first, second, first), (0, 2)),
            ("key_value", (second, first, first), (1, 2)),
        )

        for name, tensors, shared_indices in cases:
            with self.subTest(name=name):
                output_inputs = self.model.convert_inputs_to_internal_layout(
                    self.attention_inputs(*tensors)
                )
                tensors = (
                    output_inputs.query,
                    output_inputs.key,
                    output_inputs.value,
                )
                self.assertIs(
                    tensors[shared_indices[0]],
                    tensors[shared_indices[1]],
                )
                self.assertEqual(tensors[0].shape, (5, 3, 12))


class TestRestoreOutputLayout(TestBatchDimensionManager):
    def test_uses_the_shared_validator_adapter(self):
        self.assertIs(BatchDimensionManager.VALIDATOR, AttentionValidatorBase)

    def test_requires_resolved_runtime_layout(self):
        output = torch.zeros(2, 1, self.embedding_dim)
        unresolved_inputs = self.attention_inputs(output, output, output)

        with self.assertRaisesRegex(
            RuntimeError,
            "Output layout restoration requires resolved attention runtime layout.",
        ):
            self.model.restore_output_layout(output, unresolved_inputs)

    def test_runtime_layout_validation_dispatches_through_subclass(self):
        class RejectingValidator(AttentionValidatorBase):
            @staticmethod
            def validate_output_layout_restoration_runtime_layout(*args, **kwargs):
                raise RuntimeError("substituted runtime-layout validator was called")

        class RejectingBatchDimensionManager(BatchDimensionManager):
            VALIDATOR = RejectingValidator

        output = torch.zeros(2, 1, self.embedding_dim)
        attention_inputs = self.attention_inputs(output, output, output)
        model = RejectingBatchDimensionManager(self.config)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime-layout validator was called",
        ):
            model.restore_output_layout(output, attention_inputs)

    def test_sequence_first_output_is_returned_unchanged(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=False))
        query = torch.arange(5 * 3 * 12, dtype=torch.float32).view(5, 3, 12)
        attention_inputs = self.model.convert_inputs_to_internal_layout(
            self.attention_inputs(query, query, query)
        )

        restored = self.model.restore_output_layout(
            attention_inputs.query,
            attention_inputs,
        )

        self.assertIs(restored, attention_inputs.query)

    def test_batch_first_output_is_transposed_back(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=True))
        query = torch.arange(
            self.batch_size * 5 * self.embedding_dim,
            dtype=torch.float32,
        ).view(self.batch_size, 5, self.embedding_dim)
        attention_inputs = self.model.convert_inputs_to_internal_layout(
            self.attention_inputs(query, query, query)
        )

        restored = self.model.restore_output_layout(
            attention_inputs.query,
            attention_inputs,
        )

        self.assertEqual(restored.shape, query.shape)
        torch.testing.assert_close(restored, query)

    def test_unbatched_output_loses_synthetic_batch_dimension(self):
        query = torch.arange(5 * self.embedding_dim, dtype=torch.float32).view(
            5, self.embedding_dim
        )
        attention_inputs = self.model.convert_inputs_to_internal_layout(
            self.attention_inputs(query, query, query)
        )

        restored = self.model.restore_output_layout(
            attention_inputs.query,
            attention_inputs,
        )

        self.assertEqual(restored.shape, query.shape)
        torch.testing.assert_close(restored, query)

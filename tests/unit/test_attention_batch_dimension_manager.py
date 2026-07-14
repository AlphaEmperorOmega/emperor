import unittest
from dataclasses import asdict

import torch
from emperor.attention.core.config import MultiHeadAttentionConfig
from emperor.attention.core.handlers.batch import BatchDimensionManager
from emperor.attention.core.runtime import QKV, AttentionMasks

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

    def test_init_sets_exact_legacy_state(self):
        self.assertIs(self.model.should_transpose_first_two_dims, False)
        self.assertIs(self.model._legacy_input_was_batched, True)


class Test_enforce_batch_as_second_dim(TestBatchDimensionManager):
    def test_qkv_input_tensors_with_batch_as_second_dimension(self):
        query = key = value = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )

        output_q, output_k, output_v = self.model.enforce_batch_as_second_dim(
            query, key, value
        )

        self.assertIs(self.model._legacy_input_was_batched, True)
        self.assertTrue(torch.allclose(output_q, query, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(output_k, key, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(output_v, value, atol=1e-6, rtol=1e-5))

    def test_same_qkv_input_tensors_with_batch_as_first_dimension(self):
        query = key = value = torch.randn(
            self.batch_size, self.target_sequence_length, self.embedding_dim
        )
        output_q, output_k, output_v = self.model.enforce_batch_as_second_dim(
            query, key, value
        )

        expected_qkv_shape = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        self.assertEqual(output_q.shape, expected_qkv_shape)
        self.assertEqual(output_k.shape, expected_qkv_shape)
        self.assertEqual(output_v.shape, expected_qkv_shape)
        self.assertTrue(torch.allclose(output_q, output_k, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(output_k, output_v, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(output_q, output_v, atol=1e-6, rtol=1e-5))

    def test_same_kv_input_tensors_with_batch_as_first_dim(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
        )
        self.rebuild_presets(config)
        query = torch.randn(
            self.batch_size, self.target_sequence_length, self.embedding_dim
        )
        key = value = torch.randn(
            self.batch_size, self.source_sequence_length, self.embedding_dim
        )
        output_q, output_k, output_v = self.model.enforce_batch_as_second_dim(
            query, key, value
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
        self.assertEqual(output_q.shape, expected_q_shape)
        self.assertEqual(output_k.shape, expected_kv_shape)
        self.assertEqual(output_v.shape, expected_kv_shape)
        self.assertTrue(torch.allclose(output_k, output_v, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.all(output_q != output_k))
        self.assertTrue(torch.all(output_q != output_v))

    def test_indepentend_qkv_input_tensors_with_batch_as_first_dimension(self):
        config = MultiHeadAttentionConfig(
            target_sequence_length=16,
            source_sequence_length=16,
        )
        self.rebuild_presets(config)

        query = torch.randn(
            self.batch_size, self.target_sequence_length, self.embedding_dim
        )
        key = torch.randn(
            self.batch_size, self.source_sequence_length, self.embedding_dim
        )
        value = torch.randn(
            self.batch_size, self.source_sequence_length, self.embedding_dim
        )
        output_q, output_k, output_v = self.model.enforce_batch_as_second_dim(
            query, key, value
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
        self.assertEqual(output_q.shape, expected_q_shape)
        self.assertEqual(output_k.shape, expected_kv_shape)
        self.assertEqual(output_v.shape, expected_kv_shape)
        self.assertTrue(torch.all(output_k != output_v))
        self.assertTrue(torch.all(output_q != output_k))
        self.assertTrue(torch.all(output_q != output_v))


class TestConvertInputsToInternalLayout(TestBatchDimensionManager):
    def test_sequence_first_batched_inputs_preserve_value_objects(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=False))
        query = torch.randn(5, 3, 12)
        key = torch.randn(7, 3, 12)
        value = torch.randn(7, 3, 12)
        qkv = QKV(query=query, key=key, value=value)
        masks = AttentionMasks()

        output_qkv, output_masks, runtime_shape = (
            self.model.convert_inputs_to_internal_layout(qkv, masks)
        )

        self.assertIs(output_qkv, qkv)
        self.assertIs(output_masks, masks)
        self.assertEqual(runtime_shape.batch_size, 3)
        self.assertEqual(runtime_shape.target_sequence_length, 5)
        self.assertEqual(runtime_shape.source_sequence_length, 7)

    def test_distinct_unbatched_qkv_without_masks_are_all_expanded(self):
        query = torch.randn(self.target_sequence_length, self.embedding_dim)
        key = torch.randn(self.source_sequence_length, self.embedding_dim)
        value = torch.randn(self.source_sequence_length, self.embedding_dim)

        output_qkv, output_masks, runtime_shape = (
            self.model.convert_inputs_to_internal_layout(
                QKV(query=query, key=key, value=value),
                AttentionMasks(),
            )
        )

        self.assertEqual(output_qkv.query.shape, (8, 1, 12))
        self.assertEqual(output_qkv.key.shape, (8, 1, 12))
        self.assertEqual(output_qkv.value.shape, (8, 1, 12))
        self.assertIsNone(output_masks.key_padding_mask)
        self.assertFalse(runtime_shape.input_was_batched)

    def test_shared_qkv_input_tensors_preserve_identity(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=True))
        query = torch.randn(
            self.batch_size,
            self.target_sequence_length,
            self.embedding_dim,
        )
        input_qkv = QKV(query=query, key=query, value=query)
        input_masks = AttentionMasks()
        output_qkv, output_masks, runtime_shape = (
            self.model.convert_inputs_to_internal_layout(
                input_qkv,
                input_masks,
            )
        )

        expected_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        self.assertIsInstance(output_qkv, QKV)
        self.assertIs(output_masks, input_masks)
        self.assertEqual(output_qkv.query.shape, expected_output_shape)
        self.assertEqual(output_qkv.key.shape, expected_output_shape)
        self.assertEqual(output_qkv.value.shape, expected_output_shape)
        self.assertIs(output_qkv.query, output_qkv.key)
        self.assertIs(output_qkv.key, output_qkv.value)
        torch.testing.assert_close(output_qkv.query, query.transpose(0, 1))
        self.assertTrue(runtime_shape.input_was_batch_first)

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

        input_qkv = QKV(query=query, key=key, value=key)
        input_masks = AttentionMasks()
        output_qkv, output_masks, runtime_shape = (
            self.model.convert_inputs_to_internal_layout(
                input_qkv,
                input_masks,
            )
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

        self.assertIsInstance(output_qkv, QKV)
        self.assertIs(output_masks, input_masks)
        self.assertEqual(output_qkv.query.shape, expected_q_shape)
        self.assertEqual(output_qkv.key.shape, expected_kv_shape)
        self.assertEqual(output_qkv.value.shape, expected_kv_shape)
        self.assertIs(output_qkv.key, output_qkv.value)
        torch.testing.assert_close(output_qkv.query, query.transpose(0, 1))
        torch.testing.assert_close(output_qkv.key, key.transpose(0, 1))
        self.assertTrue(runtime_shape.input_was_batch_first)

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

        output_qkv, output_masks, runtime_shape = (
            self.model.convert_inputs_to_internal_layout(
                QKV(query=query, key=key, value=value),
                AttentionMasks(
                    key_padding_mask=key_padding_mask,
                    attention_mask=attention_mask,
                ),
            )
        )

        self.assertEqual(output_qkv.query.shape, (self.target_sequence_length, 1, 12))
        self.assertIs(output_qkv.key, output_qkv.value)
        self.assertEqual(output_masks.key_padding_mask.shape, (1, 8))
        self.assertIs(output_masks.attention_mask, attention_mask)
        self.assertFalse(runtime_shape.input_was_batched)

    def test_every_shared_tensor_pair_preserves_identity_when_unsqueezed(self):
        first = torch.randn(8, 12)
        second = torch.randn(8, 12)
        cases = (
            ("query_key", QKV(query=first, key=first, value=second), (0, 1)),
            ("query_value", QKV(query=first, key=second, value=first), (0, 2)),
            ("key_value", QKV(query=second, key=first, value=first), (1, 2)),
        )

        for name, qkv, shared_indices in cases:
            with self.subTest(name=name):
                output, _, _ = self.model.convert_inputs_to_internal_layout(
                    qkv,
                    AttentionMasks(),
                )
                tensors = (output.query, output.key, output.value)
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
            ("query_key", QKV(query=first, key=first, value=second), (0, 1)),
            ("query_value", QKV(query=first, key=second, value=first), (0, 2)),
            ("key_value", QKV(query=second, key=first, value=first), (1, 2)),
        )

        for name, qkv, shared_indices in cases:
            with self.subTest(name=name):
                output, _, _ = self.model.convert_inputs_to_internal_layout(
                    qkv,
                    AttentionMasks(),
                )
                tensors = (output.query, output.key, output.value)
                self.assertIs(
                    tensors[shared_indices[0]],
                    tensors[shared_indices[1]],
                )
                self.assertEqual(tensors[0].shape, (5, 3, 12))


class Test_reverse_enforced_batch_as_second_dim(TestBatchDimensionManager):
    def test_legacy_enforce_records_unbatched_input_for_reversal(self):
        query = torch.randn(5, 12)

        enforced = self.model.enforce_batch_as_second_dim(query, query, query)
        batched = self.model.add_batch_dimension_if_missing(*enforced)[0]
        restored = self.model.reverse_enforced_batch_as_second_dim(batched)

        self.assertIs(self.model._legacy_input_was_batched, False)
        torch.testing.assert_close(restored, query)

    def test_legacy_unbatched_output_loses_synthetic_batch_dimension(self):
        self.model._legacy_input_was_batched = False
        attention_output = torch.randn(self.target_sequence_length, 1, 12)

        restored = self.model.reverse_enforced_batch_as_second_dim(attention_output)

        self.assertEqual(restored.shape, (self.target_sequence_length, 12))

    def test_where_qkv_batch_dim_was_correct_second_dim(self):
        input_tensor = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        attention_output = self.model.reverse_enforced_batch_as_second_dim(
            attention_output=input_tensor
        )

        expected_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        self.assertEqual(attention_output.shape, expected_output_shape)
        self.assertTrue(
            torch.allclose(input_tensor, attention_output, atol=1e-6, rtol=1e-5)
        )

    def test_where_qkv_batch_dim_was_first_dim(self):
        self.rebuild_presets(
            MultiHeadAttentionConfig(
                batch_size=3,
                target_sequence_length=5,
                batch_first_flag=True,
            )
        )
        self.model.should_transpose_first_two_dims = True

        input_tensor = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        attention_output = self.model.reverse_enforced_batch_as_second_dim(
            attention_output=input_tensor
        )

        expected_output_shape = (
            self.batch_size,
            self.target_sequence_length,
            self.embedding_dim,
        )
        self.assertEqual(attention_output.shape, expected_output_shape)
        torch.testing.assert_close(attention_output, input_tensor.transpose(1, 0))


class TestAddBatchDimensionIfMissing(TestBatchDimensionManager):
    def test_batched_inputs_are_returned_unchanged(self):
        query = torch.randn(8, 4, 12)
        result = self.model.add_batch_dimension_if_missing(query, query, query)

        self.assertEqual(result, (query, query, query, None, None))

    def test_unbatched_inputs_and_padding_mask_gain_batch_dimension(self):
        query = torch.randn(8, 12)
        key = torch.randn(8, 12)
        value = torch.randn(8, 12)
        padding_mask = torch.zeros(8, dtype=torch.bool)
        attention_mask = torch.zeros(8, 8)

        output = self.model.add_batch_dimension_if_missing(
            query,
            key,
            value,
            padding_mask,
            attention_mask,
        )

        self.assertEqual(output[0].shape, (8, 1, 12))
        self.assertEqual(output[1].shape, (8, 1, 12))
        self.assertEqual(output[2].shape, (8, 1, 12))
        self.assertEqual(output[3].shape, (1, 8))
        self.assertIs(output[4], attention_mask)

    def test_unbatched_inputs_without_padding_mask_preserve_missing_mask(self):
        query = torch.randn(8, 12)

        output = self.model.add_batch_dimension_if_missing(query, query, query)

        self.assertEqual(output[0].shape, (8, 1, 12))
        self.assertIsNone(output[3])

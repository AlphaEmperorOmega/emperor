import unittest
from dataclasses import asdict

import torch

from emperor.attention import MultiHeadAttentionConfig
from emperor.attention._ops.batching import BatchDimensionManager
from emperor.attention._runtime import QKV, AttentionMasks
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


class TestConvertInputsToInternalLayout(TestBatchDimensionManager):
    def test_sequence_first_batched_inputs_preserve_value_objects(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=False))
        query = torch.randn(5, 3, 12)
        key = torch.randn(7, 3, 12)
        value = torch.randn(7, 3, 12)
        qkv = QKV(query=query, key=key, value=value)
        masks = AttentionMasks()

        output_qkv, output_masks, runtime_layout = (
            self.model.convert_inputs_to_internal_layout(qkv, masks)
        )

        self.assertIs(output_qkv, qkv)
        self.assertIs(output_masks, masks)
        self.assertEqual(runtime_layout.batch_size, 3)
        self.assertEqual(runtime_layout.target_sequence_length, 5)
        self.assertEqual(runtime_layout.source_sequence_length, 7)

    def test_static_keys_define_runtime_source_sequence_length(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=False))
        query = torch.randn(5, 3, 12)
        key = torch.randn(7, 3, 12)
        value = torch.randn(7, 3, 12)
        static_keys = torch.randn(6, 4, 2)

        output_qkv, _, runtime_layout = self.model.convert_inputs_to_internal_layout(
            QKV(query=query, key=key, value=value),
            AttentionMasks(),
            static_keys=static_keys,
        )

        self.assertIs(output_qkv.key, key)
        self.assertEqual(runtime_layout.source_sequence_length, 4)

    def test_distinct_unbatched_qkv_without_masks_are_all_expanded(self):
        query = torch.randn(self.target_sequence_length, self.embedding_dim)
        key = torch.randn(self.source_sequence_length, self.embedding_dim)
        value = torch.randn(self.source_sequence_length, self.embedding_dim)

        output_qkv, output_masks, runtime_layout = (
            self.model.convert_inputs_to_internal_layout(
                QKV(query=query, key=key, value=value),
                AttentionMasks(),
            )
        )

        self.assertEqual(output_qkv.query.shape, (8, 1, 12))
        self.assertEqual(output_qkv.key.shape, (8, 1, 12))
        self.assertEqual(output_qkv.value.shape, (8, 1, 12))
        self.assertIsNone(output_masks.key_padding_mask)
        self.assertFalse(runtime_layout.input_was_batched)

    def test_shared_qkv_input_tensors_preserve_identity(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=True))
        query = torch.randn(
            self.batch_size,
            self.target_sequence_length,
            self.embedding_dim,
        )
        input_qkv = QKV(query=query, key=query, value=query)
        input_masks = AttentionMasks()
        output_qkv, output_masks, runtime_layout = (
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
        self.assertTrue(runtime_layout.input_was_batch_first)

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
        output_qkv, output_masks, runtime_layout = (
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
        self.assertTrue(runtime_layout.input_was_batch_first)

    def test_distinct_batch_first_qkv_are_all_transposed(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=True))
        query = torch.randn(self.batch_size, 5, self.embedding_dim)
        key = torch.randn(self.batch_size, 7, self.embedding_dim)
        value = torch.randn(self.batch_size, 7, self.embedding_dim)

        output_qkv, _, runtime_layout = self.model.convert_inputs_to_internal_layout(
            QKV(query=query, key=key, value=value),
            AttentionMasks(),
        )

        self.assertEqual(output_qkv.query.shape, (5, self.batch_size, 12))
        self.assertEqual(output_qkv.key.shape, (7, self.batch_size, 12))
        self.assertEqual(output_qkv.value.shape, (7, self.batch_size, 12))
        torch.testing.assert_close(output_qkv.query, query.transpose(0, 1))
        torch.testing.assert_close(output_qkv.key, key.transpose(0, 1))
        torch.testing.assert_close(output_qkv.value, value.transpose(0, 1))
        self.assertEqual(runtime_layout.target_sequence_length, 5)
        self.assertEqual(runtime_layout.source_sequence_length, 7)
        self.assertTrue(runtime_layout.input_was_batch_first)

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

        output_qkv, output_masks, runtime_layout = (
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
        self.assertFalse(runtime_layout.input_was_batched)

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


class TestRestoreOutputLayout(TestBatchDimensionManager):
    def test_sequence_first_output_is_returned_unchanged(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=False))
        query = torch.arange(5 * 3 * 12, dtype=torch.float32).view(5, 3, 12)
        qkv, _, runtime_layout = self.model.convert_inputs_to_internal_layout(
            QKV(query=query, key=query, value=query),
            AttentionMasks(),
        )

        restored = self.model.restore_output_layout(qkv.query, runtime_layout)

        self.assertIs(restored, qkv.query)

    def test_batch_first_output_is_transposed_back(self):
        self.rebuild_presets(MultiHeadAttentionConfig(batch_first_flag=True))
        query = torch.arange(
            self.batch_size * 5 * self.embedding_dim,
            dtype=torch.float32,
        ).view(self.batch_size, 5, self.embedding_dim)
        qkv, _, runtime_layout = self.model.convert_inputs_to_internal_layout(
            QKV(query=query, key=query, value=query),
            AttentionMasks(),
        )

        restored = self.model.restore_output_layout(qkv.query, runtime_layout)

        self.assertEqual(restored.shape, query.shape)
        torch.testing.assert_close(restored, query)

    def test_unbatched_output_loses_synthetic_batch_dimension(self):
        query = torch.arange(5 * self.embedding_dim, dtype=torch.float32).view(
            5, self.embedding_dim
        )
        qkv, _, runtime_layout = self.model.convert_inputs_to_internal_layout(
            QKV(query=query, key=query, value=query),
            AttentionMasks(),
        )

        restored = self.model.restore_output_layout(qkv.query, runtime_layout)

        self.assertEqual(restored.shape, query.shape)
        torch.testing.assert_close(restored, query)

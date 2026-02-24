import torch
import unittest
from dataclasses import asdict
from Emperor.attention.utils.presets import MultiHeadAttentionPresets
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.attention.utils.handlers.batch import BatchDimensionManager


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
        self.config = MultiHeadAttentionPresets.multi_head_attention_preset(
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


class Test_enforce_batch_as_second_dim(TestBatchDimensionManager):
    def test_qkv_input_tensors_with_batch_as_second_dimension(self):
        query = key = value = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )

        output_q, output_k, output_v = self.model.enforce_batch_as_second_dim(
            query, key, value
        )

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


class Test___transpose_shared_tensors(TestBatchDimensionManager):
    def test__same_qk_input_tensors(self):
        query = key = torch.randn(
            self.batch_size,
            self.target_sequence_length,
            self.embedding_dim,
        )
        output_q, output_k, output_v = (
            self.model._BatchDimensionManager__transpose_shared_tensors(query, key)
        )

        expected_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        self.assertEqual(output_q.shape, expected_output_shape)
        self.assertEqual(output_k.shape, expected_output_shape)
        self.assertEqual(output_v.shape, expected_output_shape)
        self.assertTrue(torch.allclose(output_q, output_k, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(output_k, output_v, atol=1e-6, rtol=1e-5))

    def test__independent_qk_input_tensors(self):
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

        output_q, output_k, output_v = (
            self.model._BatchDimensionManager__transpose_shared_tensors(query, key)
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
        self.assertFalse(torch.allclose(output_q, output_k, atol=1e-6, rtol=1e-5))
        self.assertFalse(torch.allclose(output_q, output_v, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(output_k, output_v, atol=1e-6, rtol=1e-5))


class Test_reverse_enforced_batch_as_second_dim(TestBatchDimensionManager):
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

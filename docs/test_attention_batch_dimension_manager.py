import torch
import unittest
from dataclasses import asdict
from docs.utils import default_unittest_config
from Emperor.attention.utils.utils import BatchDimensionManager
from Emperor.attention.attention import MultiHeadAttentionConfig


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
        self.cfg = default_unittest_config()
        self.config = self.cfg.multi_head_attention_model_config
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
    def test_qkv_is_correct_input_shape(self):
        query = key = value = torch.randn(
            self.target_sequence_length, self.batch_size, self.embedding_dim
        )

        output_q, output_k, output_v = self.model.enforce_batch_as_second_dim(
            query, key, value
        )

        self.assertTrue(torch.equal(output_q, query))
        self.assertTrue(torch.equal(output_k, key))
        self.assertTrue(torch.equal(output_v, value))

    def test_same_qkv_inputs_and_input_batch_first(self):
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
        self.assertTrue(torch.equal(output_q, output_k))
        self.assertTrue(torch.equal(output_k, output_v))
        self.assertTrue(torch.equal(output_q, output_v))

    def test_same_kv_inputs_and_input_batch_first(self):
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
        self.assertTrue(torch.equal(output_k, output_v))
        self.assertTrue(torch.all(output_q != output_k))
        self.assertTrue(torch.all(output_q != output_v))

    def test_different_qkv_inputs_and_input_batch_first(self):
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


class Test___transpose_shared_qkv(TestBatchDimensionManager):
    def test__same_tensor_for_kv(self):
        query = key = torch.randn(
            self.batch_size,
            self.target_sequence_length,
            self.embedding_dim,
        )
        output_q, output_k, output_v = (
            self.model._BatchDimensionManager__transpose_shared_qkv(query, key)
        )

        expected_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        self.assertEqual(output_q.shape, expected_output_shape)
        self.assertEqual(output_k.shape, expected_output_shape)
        self.assertEqual(output_v.shape, expected_output_shape)
        self.assertTrue(torch.equal(output_q, output_k))
        self.assertTrue(torch.equal(output_k, output_v))

    def test__different_tensors_for_kv(self):
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
            self.model._BatchDimensionManager__transpose_shared_qkv(query, key)
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
        self.assertFalse(torch.equal(output_q, output_k))
        self.assertFalse(torch.equal(output_q, output_v))
        self.assertTrue(torch.equal(output_k, output_v))


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
        self.assertTrue(torch.equal(input_tensor, attention_output))

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

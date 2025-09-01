import unittest
from dataclasses import asdict
from docs.utils import default_unittest_config
from Emperor.attention.utils.utils import Utils
from Emperor.attention.utils.validation_handler import Validator
from Emperor.attention.attention import MultiHeadAttentionConfig


class TestKeyValueBias(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.target_sequence_length = None
        self.num_heads = None
        self.head_dim = None

    def rebuild_presets(self, config: MultiHeadAttentionConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.multi_head_attention_model_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.validator = Validator(self.config)
        self.model = Utils(self.config, self.validator)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.target_sequence_length = self.config.target_sequence_length
        self.source_sequence_length = self.config.source_sequence_length
        self.qk_head_dim = self.model.qk_head_dim
        self.v_head_dim = self.model.v_head_dim
        self.num_heads = self.config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads


# class Test__add_learnable_bias_vectors(TestKeyValueBias):
#     def test__kv_input_tensor_only__no_kv_biases(self):
#         key_projections = torch.randn(
#             self.source_sequence_length, self.batch_size, self.embedding_dim
#         )
#         value_projections = torch.randn(
#             self.source_sequence_length, self.batch_size, self.embedding_dim
#         )
#
#         (
#             out_key_projections,
#             out_value_projections,
#             out_key_padding_mask,
#             out_attention_mask,
#         ) = self.model.add_learnable_bias_vectors(key_projections, value_projections)
#         self.assertEqual(out_key_projections.shape, key_projections.shape)
#         self.assertEqual(out_value_projections.shape, out_value_projections.shape)
#         self.assertIsNone(out_key_padding_mask)
#         self.assertIsNone(out_attention_mask)
#
#     def test__all_inputs__no_kv_biases(self):
#         key_projections = torch.randn(
#             self.source_sequence_length, self.batch_size, self.embedding_dim
#         )
#         value_projections = torch.randn(
#             self.source_sequence_length, self.batch_size, self.embedding_dim
#         )
#         key_padding_mask = torch.randint(
#             0, 2, (self.batch_size, self.source_sequence_length)
#         )
#         attention_mask = torch.randn(
#             self.batch_size * self.num_heads,
#             self.target_sequence_length,
#             self.source_sequence_length,
#         )
#
#         (
#             out_key_projections,
#             out_value_projections,
#             out_key_padding_mask,
#             out_attention_mask,
#         ) = self.model.add_learnable_bias_vectors(
#             key_projections, value_projections, key_padding_mask, attention_mask
#         )
#         self.assertEqual(out_key_projections.shape, key_projections.shape)
#         self.assertEqual(out_value_projections.shape, out_value_projections.shape)
#         self.assertEqual(out_key_padding_mask.shape, key_padding_mask.shape)
#         self.assertEqual(out_attention_mask.shape, attention_mask.shape)
#         self.assertTrue(torch.equal(out_key_projections, key_projections))
#         self.assertTrue(torch.equal(out_value_projections, out_value_projections))
#         self.assertTrue(torch.equal(out_key_padding_mask, key_padding_mask))
#         self.assertTrue(torch.equal(out_attention_mask, attention_mask))
#
#     def test__all_inputs__add_key_value_bias_flag__True(self):
#         config = MultiHeadAttentionConfig(
#             add_key_value_bias_flag=True,
#         )
#         self.rebuild_presets(config)
#
#         key_projections = torch.randn(
#             self.source_sequence_length, self.batch_size, self.embedding_dim
#         )
#         value_projections = torch.randn(
#             self.source_sequence_length, self.batch_size, self.embedding_dim
#         )
#         key_padding_mask = torch.randint(
#             0, 2, (self.batch_size, self.source_sequence_length)
#         )
#         attention_mask = torch.randn(
#             self.batch_size * self.num_heads,
#             self.target_sequence_length,
#             self.source_sequence_length,
#         )
#
#         (
#             out_key_projections,
#             out_value_projections,
#             out_key_padding_mask,
#             out_attention_mask,
#         ) = self.model.add_learnable_bias_vectors(
#             key_projections, value_projections, key_padding_mask, attention_mask
#         )
#         source_sequence_length_updated = self.source_sequence_length + 1
#         expected_kv_shape = (
#             source_sequence_length_updated,
#             self.batch_size,
#             self.embedding_dim,
#         )
#
#         self.assertEqual(out_key_projections.shape, expected_kv_shape)
#         self.assertEqual(out_value_projections.shape, expected_kv_shape)
#         self.assertEqual(
#             out_key_padding_mask.shape,
#             (self.batch_size, source_sequence_length_updated),
#         )
#         self.assertEqual(
#             out_attention_mask.shape,
#             (
#                 self.batch_size * self.num_heads,
#                 self.target_sequence_length,
#                 source_sequence_length_updated,
#             ),
#         )

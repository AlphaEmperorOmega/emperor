import torch
import unittest
from dataclasses import asdict
from emperor.attention.core.config import MultiHeadAttentionConfig
from emperor.attention.core.handlers.zero_attention import ZeroAttention
from _attention_test_helpers import build_attention_config


class TestZeroAttention(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.num_heads = None
        self.head_dim = None

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

        self.model = ZeroAttention(self.config)

        self.batch_size = self.config.batch_size
        self.embedding_dim = self.config.embedding_dim
        self.num_heads = self.config.num_heads
        self.source_sequence_length = self.config.source_sequence_length
        self.target_sequence_length = self.config.target_sequence_length
        self.head_dim = self.embedding_dim // self.num_heads


class Test_add_zero_attention(TestZeroAttention):
    def test_flag_false_returns_inputs_unchanged(self):
        self.rebuild_presets(MultiHeadAttentionConfig(zero_attention_flag=False))
        key = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
        )
        value = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
        )
        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        out_k, out_v, out_kpm, out_am = self.model.add_zero_attention(
            key, value, key_padding_mask, attention_mask
        )

        self.assertTrue(torch.allclose(out_k, key, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(out_v, value, atol=1e-6, rtol=1e-5))
        self.assertTrue(
            torch.allclose(out_kpm, key_padding_mask, atol=1e-6, rtol=1e-5)
        )
        self.assertTrue(torch.allclose(out_am, attention_mask, atol=1e-6, rtol=1e-5))

    def test_flag_true_kv_only_pads_sequence_and_returns_none_masks(self):
        self.rebuild_presets(MultiHeadAttentionConfig(zero_attention_flag=True))
        key = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
        )
        value = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
        )

        out_k, out_v, out_kpm, out_am = self.model.add_zero_attention(key, value)

        expected_sequence_length = self.source_sequence_length + 1
        expected_shape = (
            self.batch_size * self.num_heads,
            expected_sequence_length,
            self.head_dim,
        )
        self.assertEqual(out_k.shape, expected_shape)
        self.assertEqual(out_v.shape, expected_shape)
        self.assertIsNone(out_kpm)
        self.assertIsNone(out_am)

    def test_flag_true_appended_position_is_zero(self):
        self.rebuild_presets(MultiHeadAttentionConfig(zero_attention_flag=True))
        key = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
        )
        value = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
        )

        out_k, out_v, _, _ = self.model.add_zero_attention(key, value)

        self.assertTrue(
            torch.allclose(out_k[:, -1, :], torch.zeros_like(out_k[:, -1, :]))
        )
        self.assertTrue(
            torch.allclose(out_v[:, -1, :], torch.zeros_like(out_v[:, -1, :]))
        )
        self.assertTrue(torch.allclose(out_k[:, :-1, :], key, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(out_v[:, :-1, :], value, atol=1e-6, rtol=1e-5))

    def test_flag_true_pads_masks_by_one_position(self):
        self.rebuild_presets(MultiHeadAttentionConfig(zero_attention_flag=True))
        key = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
        )
        value = torch.randn(
            self.batch_size * self.num_heads,
            self.source_sequence_length,
            self.head_dim,
        )
        key_padding_mask = torch.randint(
            0, 2, (self.batch_size, self.source_sequence_length)
        )
        attention_mask = torch.randn(
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )

        out_k, out_v, out_kpm, out_am = self.model.add_zero_attention(
            key, value, key_padding_mask, attention_mask
        )

        expected_sequence_length = self.source_sequence_length + 1
        expected_kv_shape = (
            self.batch_size * self.num_heads,
            expected_sequence_length,
            self.head_dim,
        )
        expected_kpm_shape = (self.batch_size, expected_sequence_length)
        expected_am_shape = (
            self.batch_size * self.num_heads,
            self.target_sequence_length,
            expected_sequence_length,
        )
        self.assertEqual(out_k.shape, expected_kv_shape)
        self.assertEqual(out_v.shape, expected_kv_shape)
        self.assertEqual(out_kpm.shape, expected_kpm_shape)
        self.assertEqual(out_am.shape, expected_am_shape)


if __name__ == "__main__":
    unittest.main()

import unittest
from dataclasses import asdict

import torch
from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.attention.core.config import MultiHeadAttentionConfig
from emperor.attention.core.handlers.zero_attention import ZeroAttention
from emperor.attention.core.runtime import QKV, AttentionMasks, AttentionRuntimeShape
from emperor.attention.core.variants.mixture_of_attention_heads.zero_attention import (
    MixtureOfAttentionHeadsZeroAttention,
)

from support.attention import build_attention_config


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
    def test_runtime_batch_controls_key_and_value_zero_shapes(self):
        cfg = build_attention_config(
            batch_size=7,
            num_heads=2,
            embedding_dim=4,
            query_key_projection_dim=4,
            value_projection_dim=6,
            zero_attention_flag=True,
        )
        model = ZeroAttention(cfg)
        runtime_shape = AttentionRuntimeShape(2, 3, 5)
        qkv = QKV(
            query=torch.randn(4, 3, 2),
            key=torch.randn(4, 5, 2),
            value=torch.randn(4, 5, 3),
        )

        output, _ = model.add_zero_attention(
            qkv,
            AttentionMasks(),
            runtime_shape,
        )

        self.assertEqual(output.key.shape, (4, 6, 2))
        self.assertEqual(output.value.shape, (4, 6, 3))
        torch.testing.assert_close(output.key[:, -1], torch.zeros(4, 2))
        torch.testing.assert_close(output.value[:, -1], torch.zeros(4, 3))

    def test_mixture_shared_key_values_forward_runtime_batch_to_base_handler(self):
        cfg = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=7,
            num_heads=2,
            embedding_dim=4,
            query_key_projection_dim=4,
            value_projection_dim=4,
            experts_top_k=3,
            use_kv_expert_models_flag=False,
            zero_attention_flag=True,
        )
        model = MixtureOfAttentionHeadsZeroAttention(cfg)
        runtime_shape = AttentionRuntimeShape(2, 3, 5)
        qkv = QKV(
            query=torch.randn(12, 3, 2),
            key=torch.randn(4, 5, 2),
            value=torch.randn(4, 5, 2),
        )

        output, _ = model.add_zero_attention(
            qkv,
            AttentionMasks(),
            runtime_shape,
        )

        self.assertEqual(output.key.shape, (4, 6, 2))
        self.assertEqual(output.value.shape, (4, 6, 2))

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
        qkv = QKV(
            query=torch.randn(
                self.target_sequence_length,
                self.batch_size,
                self.embedding_dim,
            ),
            key=key,
            value=value,
        )
        masks = AttentionMasks(
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )

        output_qkv, output_masks = self.model.add_zero_attention(qkv, masks)

        self.assertIs(output_qkv, qkv)
        self.assertIs(output_masks, masks)

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
        query = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        qkv = QKV(query=query, key=key, value=value)
        masks = AttentionMasks()

        output_qkv, output_masks = self.model.add_zero_attention(qkv, masks)

        expected_sequence_length = self.source_sequence_length + 1
        expected_shape = (
            self.batch_size * self.num_heads,
            expected_sequence_length,
            self.head_dim,
        )
        self.assertIsNot(output_qkv, qkv)
        self.assertIsNot(output_masks, masks)
        self.assertIs(output_qkv.query, query)
        self.assertEqual(output_qkv.key.shape, expected_shape)
        self.assertEqual(output_qkv.value.shape, expected_shape)
        self.assertIsNone(output_masks.key_padding_mask)
        self.assertIsNone(output_masks.attention_mask)

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
        qkv = QKV(
            query=torch.randn(
                self.target_sequence_length,
                self.batch_size,
                self.embedding_dim,
            ),
            key=key,
            value=value,
        )

        output_qkv, _ = self.model.add_zero_attention(qkv, AttentionMasks())

        self.assertTrue(
            torch.allclose(
                output_qkv.key[:, -1, :],
                torch.zeros_like(output_qkv.key[:, -1, :]),
            )
        )
        self.assertTrue(
            torch.allclose(
                output_qkv.value[:, -1, :],
                torch.zeros_like(output_qkv.value[:, -1, :]),
            )
        )
        torch.testing.assert_close(output_qkv.key[:, :-1, :], key)
        torch.testing.assert_close(output_qkv.value[:, :-1, :], value)

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
        query = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        qkv = QKV(query=query, key=key, value=value)
        masks = AttentionMasks(
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )

        output_qkv, output_masks = self.model.add_zero_attention(qkv, masks)

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
        self.assertIs(output_qkv.query, query)
        self.assertEqual(output_qkv.key.shape, expected_kv_shape)
        self.assertEqual(output_qkv.value.shape, expected_kv_shape)
        self.assertEqual(output_masks.key_padding_mask.shape, expected_kpm_shape)
        self.assertEqual(output_masks.attention_mask.shape, expected_am_shape)
        torch.testing.assert_close(
            output_masks.key_padding_mask[:, :-1],
            key_padding_mask,
        )
        torch.testing.assert_close(
            output_masks.attention_mask[..., :-1],
            attention_mask,
        )
        torch.testing.assert_close(
            output_masks.key_padding_mask[:, -1],
            torch.zeros_like(key_padding_mask[:, -1]),
        )
        torch.testing.assert_close(
            output_masks.attention_mask[..., -1],
            torch.zeros_like(attention_mask[..., -1]),
        )


if __name__ == "__main__":
    unittest.main()

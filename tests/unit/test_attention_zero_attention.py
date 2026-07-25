import unittest
from dataclasses import asdict

import torch

from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
    MultiHeadAttentionConfig,
)
from emperor.attention._ops.zero_attention import ZeroAttention
from emperor.attention._runtime import (
    AttentionRuntimeLayout,
    MultiHeadAttentionInputs,
)
from emperor.attention._variants.mixture.zero_attention import (
    MixtureOfAttentionHeadsZeroAttention,
)
from support.attention import build_attention_config


def attention_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    runtime_layout: AttentionRuntimeLayout | None = None,
) -> MultiHeadAttentionInputs:
    return MultiHeadAttentionInputs(
        query=query,
        key=key,
        value=value,
        key_padding_mask=key_padding_mask,
        attention_mask=attention_mask,
        runtime_layout=runtime_layout,
    )


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
        runtime_layout = AttentionRuntimeLayout(2, 3, 5)
        query = torch.randn(4, 3, 2)
        key = torch.randn(4, 5, 2)
        value = torch.randn(4, 5, 3)

        output_inputs = model.add_zero_attention(
            attention_inputs(
                query,
                key,
                value,
                runtime_layout=runtime_layout,
            )
        )

        self.assertEqual(output_inputs.key.shape, (4, 6, 2))
        self.assertEqual(output_inputs.value.shape, (4, 6, 3))
        torch.testing.assert_close(output_inputs.key[:, -1], torch.zeros(4, 2))
        torch.testing.assert_close(output_inputs.value[:, -1], torch.zeros(4, 3))
        self.assertEqual(output_inputs.runtime_layout.source_sequence_length, 6)
        self.assertEqual(output_inputs.runtime_layout.source_extension_count, 1)
        self.assertEqual(runtime_layout.source_sequence_length, 5)

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
        runtime_layout = AttentionRuntimeLayout(2, 3, 5)
        query = torch.randn(12, 3, 2)
        key = torch.randn(4, 5, 2)
        value = torch.randn(4, 5, 2)

        output_inputs = model.add_zero_attention(
            attention_inputs(
                query,
                key,
                value,
                runtime_layout=runtime_layout,
            )
        )

        self.assertEqual(output_inputs.key.shape, (4, 6, 2))
        self.assertEqual(output_inputs.value.shape, (4, 6, 2))
        self.assertEqual(output_inputs.runtime_layout.source_sequence_length, 6)

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
        query = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )
        runtime_layout = AttentionRuntimeLayout(
            batch_size=self.batch_size,
            target_sequence_length=self.target_sequence_length,
            source_sequence_length=self.source_sequence_length,
        )

        input_values = attention_inputs(
            query,
            key,
            value,
            key_padding_mask,
            attention_mask,
            runtime_layout,
        )
        output_inputs = self.model.add_zero_attention(input_values)

        self.assertIs(output_inputs, input_values)
        self.assertIs(output_inputs.query, query)
        self.assertIs(output_inputs.key, key)
        self.assertIs(output_inputs.value, value)
        self.assertIs(output_inputs.key_padding_mask, key_padding_mask)
        self.assertIs(output_inputs.attention_mask, attention_mask)
        self.assertIs(output_inputs.runtime_layout, runtime_layout)

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
        output_inputs = self.model.add_zero_attention(
            attention_inputs(query, key, value)
        )

        expected_sequence_length = self.source_sequence_length + 1
        expected_shape = (
            self.batch_size * self.num_heads,
            expected_sequence_length,
            self.head_dim,
        )
        self.assertIs(output_inputs.query, query)
        self.assertEqual(output_inputs.key.shape, expected_shape)
        self.assertEqual(output_inputs.value.shape, expected_shape)
        self.assertIsNone(output_inputs.key_padding_mask)
        self.assertIsNone(output_inputs.attention_mask)

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
        query = torch.randn(
            self.target_sequence_length,
            self.batch_size,
            self.embedding_dim,
        )

        output_inputs = self.model.add_zero_attention(
            attention_inputs(query, key, value)
        )

        self.assertTrue(
            torch.allclose(
                output_inputs.key[:, -1, :],
                torch.zeros_like(output_inputs.key[:, -1, :]),
            )
        )
        self.assertTrue(
            torch.allclose(
                output_inputs.value[:, -1, :],
                torch.zeros_like(output_inputs.value[:, -1, :]),
            )
        )
        torch.testing.assert_close(output_inputs.key[:, :-1, :], key)
        torch.testing.assert_close(output_inputs.value[:, :-1, :], value)

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
        output_inputs = self.model.add_zero_attention(
            attention_inputs(
                query,
                key,
                value,
                key_padding_mask,
                attention_mask,
            )
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
        self.assertIs(output_inputs.query, query)
        self.assertEqual(output_inputs.key.shape, expected_kv_shape)
        self.assertEqual(output_inputs.value.shape, expected_kv_shape)
        self.assertEqual(output_inputs.key_padding_mask.shape, expected_kpm_shape)
        self.assertEqual(output_inputs.attention_mask.shape, expected_am_shape)
        torch.testing.assert_close(
            output_inputs.key_padding_mask[:, :-1],
            key_padding_mask,
        )
        torch.testing.assert_close(
            output_inputs.attention_mask[..., :-1],
            attention_mask,
        )
        torch.testing.assert_close(
            output_inputs.key_padding_mask[:, -1],
            torch.zeros_like(key_padding_mask[:, -1]),
        )
        torch.testing.assert_close(
            output_inputs.attention_mask[..., -1],
            torch.zeros_like(attention_mask[..., -1]),
        )


if __name__ == "__main__":
    unittest.main()

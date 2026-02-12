import torch
import unittest

from Emperor.attention.utils.presets import MultiHeadAttentionPresets
from Emperor.attention.utils.handlers.bias import KeyValueBias

class TestKeyValueBias(unittest.TestCase):
    def test_init(self):
        bias_options = [True, False]
        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for add_key_value_bias_flag in bias_options:
            for embedding_dim, qk_dim, v_dim in dimension_options:
                message = f"Test failed for the options: add_key_value_bias_flag={add_key_value_bias_flag}, embedding_dim={embedding_dim}, qk_dim={qk_dim}, v_dim={v_dim}"
                with self.subTest(i=message):
                    cfg = MultiHeadAttentionPresets.multi_head_attention_preset(
                        embedding_dim=embedding_dim,
                        query_key_projection_dim=qk_dim,
                        value_projection_dim=v_dim,
                        add_key_value_bias_flag=add_key_value_bias_flag,
                    )
                    m = KeyValueBias(cfg)

                    self.assertEqual(m.batch_size, cfg.batch_size)
                    self.assertEqual(
                        m.add_key_value_bias_flag, add_key_value_bias_flag
                    )
                    self.assertEqual(m.query_key_projection_dim, qk_dim)
                    self.assertEqual(m.value_projection_dim, v_dim)

                    if add_key_value_bias_flag:
                        self.assertIsInstance(m.key_bias_vector, torch.Tensor)
                        self.assertIsInstance(m.value_bias_vector, torch.Tensor)
                        self.assertEqual(
                            m.key_bias_vector.shape, (1, 1, qk_dim)
                        )
                        self.assertEqual(
                            m.value_bias_vector.shape, (1, 1, v_dim)
                        )
                    else:
                        self.assertIsNone(m.key_bias_vector)
                        self.assertIsNone(m.value_bias_vector)

    def test_forward_no_bias(self):
        batch_size = 2
        source_seq_len = 10

        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for embedding_dim, qk_dim, v_dim in dimension_options:
            message = f"Test failed for the options: embedding_dim={embedding_dim}, qk_dim={qk_dim}, v_dim={v_dim}"
            with self.subTest(i=message):
                cfg = MultiHeadAttentionPresets.multi_head_attention_preset(
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=qk_dim,
                    value_projection_dim=v_dim,
                    batch_size=batch_size,
                    source_sequence_length=source_seq_len,
                    add_key_value_bias_flag=False,
                )
                m = KeyValueBias(cfg)

                key_projections = torch.randn(
                    source_seq_len, batch_size, qk_dim
                )
                value_projections = torch.randn(
                    source_seq_len, batch_size, v_dim
                )

                out_k, out_v, out_k_mask, out_attn_mask = (
                    m.add_kv_learnable_bias_vectors(
                        key_projections, value_projections
                    )
                )

                self.assertEqual(out_k.shape, key_projections.shape)
                self.assertEqual(out_v.shape, value_projections.shape)
                self.assertTrue(torch.equal(out_k, key_projections))
                self.assertTrue(torch.equal(out_v, value_projections))
                self.assertIsNone(out_k_mask)
                self.assertIsNone(out_attn_mask)

    def test_forward_no_bias_with_masks(self):
        batch_size = 4
        num_heads = 4
        source_seq_len = 10
        target_seq_len = 18

        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for embedding_dim, qk_dim, v_dim in dimension_options:
            message = f"Test failed for the options: embedding_dim={embedding_dim}, qk_dim={qk_dim}, v_dim={v_dim}"
            with self.subTest(i=message):
                cfg = MultiHeadAttentionPresets.multi_head_attention_preset(
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=qk_dim,
                    value_projection_dim=v_dim,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    source_sequence_length=source_seq_len,
                    target_sequence_length=target_seq_len,
                    add_key_value_bias_flag=False,
                )
                m = KeyValueBias(cfg)

                key_projections = torch.randn(
                    source_seq_len, batch_size, qk_dim
                )
                value_projections = torch.randn(
                    source_seq_len, batch_size, v_dim
                )
                key_padding_mask = torch.randint(
                    0, 2, (batch_size, source_seq_len)
                )
                attention_mask = torch.randn(
                    batch_size * num_heads, target_seq_len, source_seq_len
                )

                out_k, out_v, out_k_mask, out_attn_mask = (
                    m.add_kv_learnable_bias_vectors(
                        key_projections,
                        value_projections,
                        key_padding_mask,
                        attention_mask,
                    )
                )

                self.assertEqual(out_k.shape, key_projections.shape)
                self.assertEqual(out_v.shape, value_projections.shape)
                self.assertTrue(torch.equal(out_k, key_projections))
                self.assertTrue(torch.equal(out_v, value_projections))
                self.assertTrue(torch.equal(out_k_mask, key_padding_mask))
                self.assertTrue(torch.equal(out_attn_mask, attention_mask))

    def test_forward_with_bias_no_masks(self):
        batch_size = 4
        source_seq_len = 20

        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for embedding_dim, qk_dim, v_dim in dimension_options:
            message = f"Test failed for the options: embedding_dim={embedding_dim}, qk_dim={qk_dim}, v_dim={v_dim}"
            with self.subTest(i=message):
                cfg = MultiHeadAttentionPresets.multi_head_attention_preset(
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=qk_dim,
                    value_projection_dim=v_dim,
                    batch_size=batch_size,
                    source_sequence_length=source_seq_len,
                    add_key_value_bias_flag=True,
                )
                m = KeyValueBias(cfg)

                key_projections = torch.randn(
                    source_seq_len, batch_size, qk_dim
                )
                value_projections = torch.randn(
                    source_seq_len, batch_size, v_dim
                )

                out_k, out_v, out_k_mask, out_attn_mask = (
                    m.add_kv_learnable_bias_vectors(
                        key_projections, value_projections
                    )
                )

                expected_seq_len = source_seq_len + 1
                self.assertEqual(
                    out_k.shape, (expected_seq_len, batch_size, qk_dim)
                )
                self.assertEqual(
                    out_v.shape, (expected_seq_len, batch_size, v_dim)
                )
                self.assertIsNone(out_k_mask)
                self.assertIsNone(out_attn_mask)

    def test_forward_with_bias_and_masks(self):
        batch_size = 4
        num_heads = 4
        source_seq_len = 10
        target_seq_len = 18

        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for embedding_dim, qk_dim, v_dim in dimension_options:
            message = f"Test failed for the options: embedding_dim={embedding_dim}, qk_dim={qk_dim}, v_dim={v_dim}"
            with self.subTest(i=message):
                cfg = MultiHeadAttentionPresets.multi_head_attention_preset(
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=qk_dim,
                    value_projection_dim=v_dim,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    source_sequence_length=source_seq_len,
                    target_sequence_length=target_seq_len,
                    add_key_value_bias_flag=True,
                )
                m = KeyValueBias(cfg)

                key_projections = torch.randn(
                    source_seq_len, batch_size, qk_dim
                )
                value_projections = torch.randn(
                    source_seq_len, batch_size, v_dim
                )
                key_padding_mask = torch.randint(
                    0, 2, (batch_size, source_seq_len)
                )
                attention_mask = torch.randn(
                    batch_size * num_heads,
                    target_seq_len,
                    source_seq_len,
                )

                out_k, out_v, out_k_mask, out_attn_mask = (
                    m.add_kv_learnable_bias_vectors(
                        key_projections,
                        value_projections,
                        key_padding_mask,
                        attention_mask,
                    )
                )

                expected_seq_len = source_seq_len + 1
                self.assertEqual(
                    out_k.shape, (expected_seq_len, batch_size, qk_dim)
                )
                self.assertEqual(
                    out_v.shape, (expected_seq_len, batch_size, v_dim)
                )
                self.assertEqual(
                    out_k_mask.shape,
                    (batch_size, expected_seq_len),
                )
                self.assertEqual(
                    out_attn_mask.shape,
                    (
                        batch_size * num_heads,
                        target_seq_len,
                        expected_seq_len,
                    ),
                )

    def test_gradients_flow_through_bias_vectors(self):
        batch_size = 4
        source_seq_len = 20

        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for embedding_dim, qk_dim, v_dim in dimension_options:
            message = f"Test failed for the options: embedding_dim={embedding_dim}, qk_dim={qk_dim}, v_dim={v_dim}"
            with self.subTest(i=message):
                cfg = MultiHeadAttentionPresets.multi_head_attention_preset(
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=qk_dim,
                    value_projection_dim=v_dim,
                    batch_size=batch_size,
                    source_sequence_length=source_seq_len,
                    add_key_value_bias_flag=True,
                )
                m = KeyValueBias(cfg)

                key_projections = torch.randn(
                    source_seq_len, batch_size, qk_dim
                )
                value_projections = torch.randn(
                    source_seq_len, batch_size, v_dim
                )

                out_k, out_v, _, _ = m.add_kv_learnable_bias_vectors(
                    key_projections, value_projections
                )
                loss = out_k.sum() + out_v.sum()
                loss.backward()

                self.assertIsNotNone(m.key_bias_vector.grad)
                self.assertEqual(
                    m.key_bias_vector.grad.shape,
                    m.key_bias_vector.shape,
                )
                self.assertIsNotNone(m.value_bias_vector.grad)
                self.assertEqual(
                    m.value_bias_vector.grad.shape,
                    m.value_bias_vector.shape,
                )

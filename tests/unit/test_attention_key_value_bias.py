import unittest

import torch
from emperor.attention.core.handlers.bias import KeyValueBias
from emperor.attention.core.runtime import QKV, AttentionMasks, AttentionRuntimeShape

from support.attention import build_attention_config


class TestKeyValueBias(unittest.TestCase):
    def test_rejects_attention_ready_projection_with_invalid_branch_count(self):
        cfg = build_attention_config(
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            query_key_projection_dim=4,
            value_projection_dim=4,
            add_key_value_bias_flag=True,
        )
        model = KeyValueBias(cfg)
        invalid = torch.randn(3, 2, 2)

        with self.assertRaises(RuntimeError) as caught:
            model.add_kv_learnable_bias_vectors(
                QKV(query=invalid, key=invalid, value=invalid),
                AttentionMasks(),
            )
        self.assertEqual(
            str(caught.exception),
            "Attention-ready key/value projections must have a leading dimension "
            "divisible by batch_size * num_heads.",
        )

    def test_runtime_batch_and_branch_multiplier_preserve_head_order(self):
        cfg = build_attention_config(
            batch_size=7,
            num_heads=2,
            embedding_dim=4,
            query_key_projection_dim=4,
            value_projection_dim=6,
            add_key_value_bias_flag=True,
        )
        model = KeyValueBias(cfg)
        with torch.no_grad():
            model.key_bias_vector.copy_(torch.arange(4).view(1, 1, 4))
            model.value_bias_vector.copy_(torch.arange(10, 16).view(1, 1, 6))
        runtime_shape = AttentionRuntimeShape(
            batch_size=2,
            target_sequence_length=3,
            source_sequence_length=5,
        )
        branch_multiplier = 3
        branch_count = runtime_shape.batch_size * branch_multiplier * cfg.num_heads
        qkv = QKV(
            query=torch.zeros(branch_count, 3, 2),
            key=torch.zeros(branch_count, 5, 2),
            value=torch.zeros(branch_count, 5, 3),
        )

        output, _ = model.add_kv_learnable_bias_vectors(
            qkv,
            AttentionMasks(),
            runtime_shape,
        )

        expected_key = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).repeat(
            runtime_shape.batch_size * branch_multiplier,
            1,
        )
        expected_value = torch.tensor([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]).repeat(
            runtime_shape.batch_size * branch_multiplier, 1
        )
        torch.testing.assert_close(output.key[:, -1], expected_key)
        torch.testing.assert_close(output.value[:, -1], expected_value)

    def test_init(self):
        bias_options = [True, False]
        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for add_key_value_bias_flag in bias_options:
            for embedding_dim, qk_dim, v_dim in dimension_options:
                with self.subTest(
                    add_key_value_bias_flag=add_key_value_bias_flag,
                    embedding_dim=embedding_dim,
                    qk_dim=qk_dim,
                    v_dim=v_dim,
                ):
                    cfg = build_attention_config(
                        embedding_dim=embedding_dim,
                        query_key_projection_dim=qk_dim,
                        value_projection_dim=v_dim,
                        add_key_value_bias_flag=add_key_value_bias_flag,
                    )
                    m = KeyValueBias(cfg)

                    self.assertEqual(m.batch_size, cfg.batch_size)
                    self.assertEqual(m.add_key_value_bias_flag, add_key_value_bias_flag)
                    self.assertEqual(m.query_key_projection_dim, qk_dim)
                    self.assertEqual(m.value_projection_dim, v_dim)

                    if add_key_value_bias_flag:
                        self.assertIsInstance(m.key_bias_vector, torch.Tensor)
                        self.assertIsInstance(m.value_bias_vector, torch.Tensor)
                        self.assertEqual(m.key_bias_vector.shape, (1, 1, qk_dim))
                        self.assertEqual(m.value_bias_vector.shape, (1, 1, v_dim))
                    else:
                        self.assertIsNone(m.key_bias_vector)
                        self.assertIsNone(m.value_bias_vector)

    def test_forward_no_bias(self):
        batch_size = 2
        source_seq_len = 10

        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for embedding_dim, qk_dim, v_dim in dimension_options:
            with self.subTest(
                embedding_dim=embedding_dim,
                qk_dim=qk_dim,
                v_dim=v_dim,
            ):
                cfg = build_attention_config(
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=qk_dim,
                    value_projection_dim=v_dim,
                    batch_size=batch_size,
                    source_sequence_length=source_seq_len,
                    add_key_value_bias_flag=False,
                )
                m = KeyValueBias(cfg)

                key_projections = torch.randn(source_seq_len, batch_size, qk_dim)
                value_projections = torch.randn(source_seq_len, batch_size, v_dim)
                query = torch.randn(source_seq_len, batch_size, embedding_dim)
                qkv = QKV(
                    query=query,
                    key=key_projections,
                    value=value_projections,
                )
                masks = AttentionMasks()

                output_qkv, output_masks = m.add_kv_learnable_bias_vectors(
                    qkv,
                    masks,
                )

                self.assertIs(output_qkv, qkv)
                self.assertIs(output_masks, masks)
                self.assertIs(output_qkv.query, query)

    def test_forward_no_bias_with_masks(self):
        batch_size = 4
        num_heads = 4
        source_seq_len = 10
        target_seq_len = 18

        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for embedding_dim, qk_dim, v_dim in dimension_options:
            with self.subTest(
                embedding_dim=embedding_dim,
                qk_dim=qk_dim,
                v_dim=v_dim,
            ):
                cfg = build_attention_config(
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

                key_projections = torch.randn(source_seq_len, batch_size, qk_dim)
                value_projections = torch.randn(source_seq_len, batch_size, v_dim)
                key_padding_mask = torch.randint(0, 2, (batch_size, source_seq_len))
                attention_mask = torch.randn(
                    batch_size * num_heads, target_seq_len, source_seq_len
                )
                qkv = QKV(
                    query=torch.randn(target_seq_len, batch_size, embedding_dim),
                    key=key_projections,
                    value=value_projections,
                )
                masks = AttentionMasks(
                    key_padding_mask=key_padding_mask,
                    attention_mask=attention_mask,
                )

                output_qkv, output_masks = m.add_kv_learnable_bias_vectors(qkv, masks)

                self.assertIs(output_qkv, qkv)
                self.assertIs(output_masks, masks)

    def test_forward_with_bias_no_masks(self):
        batch_size = 4
        num_heads = 2
        source_seq_len = 20

        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for embedding_dim, qk_dim, v_dim in dimension_options:
            with self.subTest(
                embedding_dim=embedding_dim,
                qk_dim=qk_dim,
                v_dim=v_dim,
            ):
                cfg = build_attention_config(
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=qk_dim,
                    value_projection_dim=v_dim,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    source_sequence_length=source_seq_len,
                    add_key_value_bias_flag=True,
                )
                m = KeyValueBias(cfg)

                branch_count = batch_size * num_heads
                key_head_dim = qk_dim // num_heads
                value_head_dim = v_dim // num_heads
                key_projections = torch.randn(
                    branch_count, source_seq_len, key_head_dim
                )
                value_projections = torch.randn(
                    branch_count, source_seq_len, value_head_dim
                )
                query = torch.randn(branch_count, source_seq_len, key_head_dim)
                qkv = QKV(
                    query=query,
                    key=key_projections,
                    value=value_projections,
                )
                masks = AttentionMasks()

                output_qkv, output_masks = m.add_kv_learnable_bias_vectors(qkv, masks)

                expected_seq_len = source_seq_len + 1
                self.assertIsNot(output_qkv, qkv)
                self.assertIsNot(output_masks, masks)
                self.assertIs(output_qkv.query, query)
                self.assertEqual(
                    output_qkv.key.shape,
                    (branch_count, expected_seq_len, key_head_dim),
                )
                self.assertEqual(
                    output_qkv.value.shape,
                    (branch_count, expected_seq_len, value_head_dim),
                )
                self.assertIsNone(output_masks.key_padding_mask)
                self.assertIsNone(output_masks.attention_mask)

    def test_forward_with_bias_and_masks(self):
        batch_size = 4
        num_heads = 2
        source_seq_len = 10
        target_seq_len = 18

        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for embedding_dim, qk_dim, v_dim in dimension_options:
            with self.subTest(
                embedding_dim=embedding_dim,
                qk_dim=qk_dim,
                v_dim=v_dim,
            ):
                cfg = build_attention_config(
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

                branch_count = batch_size * num_heads
                key_head_dim = qk_dim // num_heads
                value_head_dim = v_dim // num_heads
                key_projections = torch.randn(
                    branch_count, source_seq_len, key_head_dim
                )
                value_projections = torch.randn(
                    branch_count, source_seq_len, value_head_dim
                )
                key_padding_mask = torch.randint(0, 2, (batch_size, source_seq_len))
                attention_mask = torch.randn(
                    batch_size * num_heads,
                    target_seq_len,
                    source_seq_len,
                )
                query = torch.randn(branch_count, target_seq_len, key_head_dim)
                qkv = QKV(
                    query=query,
                    key=key_projections,
                    value=value_projections,
                )
                masks = AttentionMasks(
                    key_padding_mask=key_padding_mask,
                    attention_mask=attention_mask,
                )

                output_qkv, output_masks = m.add_kv_learnable_bias_vectors(qkv, masks)

                expected_seq_len = source_seq_len + 1
                self.assertIsNot(output_qkv, qkv)
                self.assertIsNot(output_masks, masks)
                self.assertIs(output_qkv.query, query)
                self.assertEqual(
                    output_qkv.key.shape,
                    (branch_count, expected_seq_len, key_head_dim),
                )
                self.assertEqual(
                    output_qkv.value.shape,
                    (branch_count, expected_seq_len, value_head_dim),
                )
                self.assertEqual(
                    output_masks.key_padding_mask.shape,
                    (batch_size, expected_seq_len),
                )
                self.assertEqual(
                    output_masks.attention_mask.shape,
                    (
                        batch_size * num_heads,
                        target_seq_len,
                        expected_seq_len,
                    ),
                )
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

    def test_gradients_flow_through_bias_vectors(self):
        batch_size = 4
        num_heads = 2
        source_seq_len = 20

        dimension_options = [(8, 8, 8), (12, 14, 16), (8, 12, 8)]

        for embedding_dim, qk_dim, v_dim in dimension_options:
            with self.subTest(
                embedding_dim=embedding_dim,
                qk_dim=qk_dim,
                v_dim=v_dim,
            ):
                cfg = build_attention_config(
                    embedding_dim=embedding_dim,
                    query_key_projection_dim=qk_dim,
                    value_projection_dim=v_dim,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    source_sequence_length=source_seq_len,
                    add_key_value_bias_flag=True,
                )
                m = KeyValueBias(cfg)

                branch_count = batch_size * num_heads
                key_head_dim = qk_dim // num_heads
                value_head_dim = v_dim // num_heads
                key_projections = torch.randn(
                    branch_count, source_seq_len, key_head_dim
                )
                value_projections = torch.randn(
                    branch_count, source_seq_len, value_head_dim
                )

                output_qkv, _ = m.add_kv_learnable_bias_vectors(
                    QKV(
                        query=torch.randn(
                            branch_count,
                            source_seq_len,
                            key_head_dim,
                        ),
                        key=key_projections,
                        value=value_projections,
                    ),
                    AttentionMasks(),
                )
                loss = output_qkv.key.sum() + output_qkv.value.sum()
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

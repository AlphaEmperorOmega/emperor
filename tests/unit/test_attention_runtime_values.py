import unittest
from dataclasses import FrozenInstanceError, replace

import torch

from emperor.attention._runtime import (
    QKV,
    AttentionMasks,
    AttentionRuntimeLayout,
    MultiHeadAttentionInputs,
)


class TestQKV(unittest.TestCase):
    def test_init_stores_exact_named_tensor_references(self):
        query = torch.randn(2, 3)
        key = torch.randn(2, 3)
        value = torch.randn(2, 3)

        qkv = QKV(query=query, key=key, value=value)

        self.assertIs(qkv.query, query)
        self.assertIs(qkv.key, key)
        self.assertIs(qkv.value, value)

    def test_init_requires_keyword_arguments(self):
        tensor = torch.randn(2, 3)

        with self.assertRaises(TypeError):
            QKV(tensor, tensor, tensor)

    def test_fields_cannot_be_rebound(self):
        tensor = torch.randn(2, 3)
        qkv = QKV(query=tensor, key=tensor, value=tensor)

        with self.assertRaises(FrozenInstanceError):
            qkv.query = torch.randn(2, 3)

    def test_replace_preserves_unchanged_references(self):
        query = torch.randn(2, 3)
        key = torch.randn(2, 3)
        value = torch.randn(2, 3)
        replacement_key = torch.randn(2, 3)
        qkv = QKV(query=query, key=key, value=value)

        replaced = replace(qkv, key=replacement_key)

        self.assertIs(replaced.query, query)
        self.assertIs(replaced.key, replacement_key)
        self.assertIs(replaced.value, value)

    def test_no_generated_tensor_equality(self):
        tensor = torch.randn(2, 3)
        first = QKV(query=tensor, key=tensor, value=tensor)
        second = QKV(query=tensor, key=tensor, value=tensor)

        self.assertIsNot(first, second)
        self.assertFalse(first == second)
        self.assertTrue(first == first)


class TestAttentionMasks(unittest.TestCase):
    def test_init_defaults_to_no_masks(self):
        masks = AttentionMasks()

        self.assertIsNone(masks.key_padding_mask)
        self.assertIsNone(masks.attention_mask)

    def test_init_stores_exact_named_mask_references(self):
        key_padding_mask = torch.zeros(2, 3, dtype=torch.bool)
        attention_mask = torch.zeros(3, 3, dtype=torch.bool)

        masks = AttentionMasks(
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )

        self.assertIs(masks.key_padding_mask, key_padding_mask)
        self.assertIs(masks.attention_mask, attention_mask)

    def test_init_requires_keyword_arguments(self):
        mask = torch.zeros(2, 3, dtype=torch.bool)

        with self.assertRaises(TypeError):
            AttentionMasks(mask, mask)

    def test_fields_cannot_be_rebound(self):
        masks = AttentionMasks()

        with self.assertRaises(FrozenInstanceError):
            masks.attention_mask = torch.zeros(2, 3, dtype=torch.bool)

    def test_replace_changes_only_named_mask(self):
        key_padding_mask = torch.zeros(2, 3, dtype=torch.bool)
        attention_mask = torch.zeros(3, 3, dtype=torch.bool)
        replacement_attention_mask = torch.ones(3, 3, dtype=torch.bool)
        masks = AttentionMasks(
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )

        replaced = replace(masks, attention_mask=replacement_attention_mask)

        self.assertIs(replaced.key_padding_mask, key_padding_mask)
        self.assertIs(replaced.attention_mask, replacement_attention_mask)


class TestMultiHeadAttentionInputs(unittest.TestCase):
    def test_defaults_store_query_key_value_and_create_empty_optional_fields(self):
        tensor = torch.randn(2, 3)

        attention_inputs = MultiHeadAttentionInputs(
            query=tensor,
            key=tensor,
            value=tensor,
        )

        self.assertIs(attention_inputs.query, tensor)
        self.assertIs(attention_inputs.key, tensor)
        self.assertIs(attention_inputs.value, tensor)
        self.assertIsNone(attention_inputs.key_padding_mask)
        self.assertIsNone(attention_inputs.attention_mask)
        self.assertIsNone(attention_inputs.static_key)
        self.assertIsNone(attention_inputs.static_value)
        self.assertIsNone(attention_inputs.runtime_layout)
        self.assertIsNone(attention_inputs.merged_attention_mask)

    def test_stores_exact_runtime_value_references(self):
        query = torch.randn(2, 3)
        key = torch.randn(2, 3)
        value = torch.randn(2, 3)
        static_key = torch.randn(2, 3)
        static_value = torch.randn(2, 3)
        key_padding_mask = torch.randn(2, 3)
        attention_mask = torch.randn(2, 3)
        merged_attention_mask = torch.randn(2, 3)
        runtime_layout = AttentionRuntimeLayout(
            batch_size=1,
            target_sequence_length=2,
            source_sequence_length=3,
        )

        attention_inputs = MultiHeadAttentionInputs(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
            static_key=static_key,
            static_value=static_value,
            runtime_layout=runtime_layout,
            merged_attention_mask=merged_attention_mask,
        )

        self.assertIs(attention_inputs.query, query)
        self.assertIs(attention_inputs.key, key)
        self.assertIs(attention_inputs.value, value)
        self.assertIs(attention_inputs.key_padding_mask, key_padding_mask)
        self.assertIs(attention_inputs.attention_mask, attention_mask)
        self.assertIs(attention_inputs.static_key, static_key)
        self.assertIs(attention_inputs.static_value, static_value)
        self.assertIs(attention_inputs.runtime_layout, runtime_layout)
        self.assertIs(
            attention_inputs.merged_attention_mask,
            merged_attention_mask,
        )

    def test_fields_cannot_be_rebound(self):
        tensor = torch.randn(2, 3)
        attention_inputs = MultiHeadAttentionInputs(
            query=tensor,
            key=tensor,
            value=tensor,
        )

        with self.assertRaises(FrozenInstanceError):
            attention_inputs.query = torch.randn(2, 3)

    def test_replace_preserves_unchanged_references(self):
        query = torch.randn(2, 3)
        key = torch.randn(2, 3)
        value = torch.randn(2, 3)
        replacement_key = torch.randn(2, 3)
        attention_inputs = MultiHeadAttentionInputs(
            query=query,
            key=key,
            value=value,
        )

        replaced = replace(attention_inputs, key=replacement_key)

        self.assertIs(replaced.query, query)
        self.assertIs(replaced.key, replacement_key)
        self.assertIs(replaced.value, value)

    def test_no_generated_tensor_equality(self):
        tensor = torch.randn(2, 3)
        first = MultiHeadAttentionInputs(query=tensor, key=tensor, value=tensor)
        second = MultiHeadAttentionInputs(query=tensor, key=tensor, value=tensor)

        self.assertIsNot(first, second)
        self.assertFalse(first == second)
        self.assertTrue(first == first)


if __name__ == "__main__":
    unittest.main()

import unittest
from dataclasses import FrozenInstanceError, replace

import torch
from emperor.attention.core.runtime import QKV, AttentionMasks


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


if __name__ == "__main__":
    unittest.main()

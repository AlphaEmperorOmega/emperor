import unittest

import torch


class HierarchicalTextBatchTests(unittest.TestCase):
    def test_empty_documents_and_unicode_byte_boundaries(self):
        from emperor.decoding.hierarchical import (
            HierarchicalLanguageModelBatch,
            HierarchicalTextCodec,
        )

        codec = HierarchicalTextCodec(4)
        self.assertEqual(list(codec.split_text("😀🌍é\n")), ["😀", "🌍", "é\n"])
        empty = HierarchicalLanguageModelBatch.collate(
            list(codec.training_windows("", 1))
        )
        self.assertEqual(empty.labels.tolist(), [257])
        self.assertEqual(empty.bos_mask.tolist(), [[True]])
        self.assertEqual(empty.byte_prefix_ids.shape, (1, 1, 0))
        self.assertEqual(empty.byte_count, 0)
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            list(codec.split_text("\ud800"))
        with self.assertRaisesRegex(ValueError, "character"):
            list(HierarchicalTextCodec(3).split_text("😀"))
        with self.assertRaisesRegex(TypeError, "string"):
            list(codec.split_text(None))
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            list(codec.training_windows("a", 0))
        with self.assertRaisesRegex(ValueError, "at least"):
            HierarchicalLanguageModelBatch.collate([])

    def test_lossless_splitting_and_shifted_packed_labels(self):
        from emperor.decoding.hierarchical import (
            HierarchicalLanguageModelBatch,
            HierarchicalTextCodec,
        )

        codec = HierarchicalTextCodec(max_token_bytes=8)
        text = "  Café\t😀 ab\x00cdefghi\n"
        tokens = list(codec.split_text(text))
        self.assertEqual("".join(tokens), text)
        self.assertTrue(all(len(token.encode()) <= 8 for token in tokens))
        windows = list(codec.training_windows("mat cat ", sequence_length=2))
        self.assertEqual(len(windows), 2)
        batch = HierarchicalLanguageModelBatch.collate(windows)
        self.assertEqual(batch.context_texts, (("", "mat "), ("cat ", "")))
        self.assertEqual(batch.bos_mask.tolist(), [[True, False], [False, False]])
        self.assertEqual(batch.attention_mask.tolist(), [[True, True], [True, False]])
        self.assertEqual(batch.byte_lengths.tolist(), [[4, 4], [0, 0]])
        self.assertEqual(
            batch.labels.tolist(), [109, 97, 116, 32, 256, 99, 97, 116, 32, 256, 257]
        )
        self.assertEqual(batch.byte_count, 8)
        self.assertEqual(batch.to("cpu").byte_prefix_ids.dtype, torch.long)

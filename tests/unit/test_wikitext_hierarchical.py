import tempfile
import unittest

from datasets import Dataset


class WikiTextHierarchicalTests(unittest.TestCase):
    def test_articles_subsections_splits_and_disk_cache(self):
        from emperor.datasets.text.language_modeling import WikiText103Hierarchical

        rows = [
            "\n",
            " = First = \n",
            "Café 😀\n",
            " = = Details = = \n",
            "Ab\x00cd\n",
            " = Second = \n",
            "Last",
        ]
        source = Dataset.from_dict({"text": rows})

        class LocalWikiText(WikiText103Hierarchical):
            def _dataset(self, split):
                return source

        with tempfile.TemporaryDirectory() as root:
            data = LocalWikiText(
                root=root,
                batch_size=2,
                sequence_length=3,
                max_token_bytes=8,
                num_workers=0,
            )
            data.setup("fit")
            examples = list(data.train)
            restored = "".join(
                target
                for row in examples
                for target in row["target_texts"]
                if target is not None
            )
            self.assertEqual(restored, "".join(rows))
            self.assertEqual(sum(sum(row["bos_mask"]) for row in examples), 2)
            self.assertEqual(
                sum(
                    target is None for row in examples for target in row["target_texts"]
                ),
                2,
            )
            self.assertTrue(data.train.cache_files)
            self.assertNotEqual(data.train.cache_files, data.val.cache_files)
            again = LocalWikiText(
                root=root, sequence_length=3, max_token_bytes=8, num_workers=0
            )
            again.setup("fit")
            self.assertEqual(data.train.cache_files, again.train.cache_files)
            batch = next(iter(data.val_dataloader()))
            self.assertGreater(batch.byte_count, 0)
            self.assertFalse(hasattr(data, "vocab"))

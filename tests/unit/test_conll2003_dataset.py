import unittest
from unittest.mock import call, patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.text.ner._conll2003 import CoNLL2003


def _sample(tokens: list[str], tags: list[int]) -> dict[str, list]:
    return {"tokens": tokens, "ner_tags": tags}


class TestCoNLLLifecycle(unittest.TestCase):
    def test_prepare_fit_samples_loaders_and_labels_are_literal(self) -> None:
        train_samples = [
            _sample(["zebra", "apple", "middle", "tail"], [1]),
            _sample(["apple"], [8, 7, 6, 5]),
        ]
        validation_samples = [_sample(["zebra", "unknown"], [3, 4])]

        def source(name: str, *, split: str):
            self.assertEqual(name, "conll2003")
            return train_samples if split == "train" else validation_samples

        data = CoNLL2003(batch_size=1, sequence_length=3)
        data.num_workers = 0
        with patch(
            "emperor.datasets.text.ner._conll2003.load_dataset",
            side_effect=source,
        ) as load_dataset:
            data.prepare_data()
            data.setup("fit")

        self.assertEqual(
            load_dataset.call_args_list,
            [
                call("conll2003", split="train"),
                call("conll2003", split="validation"),
                call("conll2003", split="train"),
                call("conll2003", split="validation"),
            ],
        )
        self.assertEqual(len(data.train), 2)
        train_tokens, train_tags = data.train[0]
        validation_tokens, validation_tags = data.val[0]
        torch.testing.assert_close(train_tokens, torch.tensor([5, 2, 3]))
        torch.testing.assert_close(train_tags, torch.tensor([1, 0, 0]))
        torch.testing.assert_close(validation_tokens, torch.tensor([5, 1, 0]))
        torch.testing.assert_close(validation_tags, torch.tensor([3, 4, 0]))
        self.assertEqual(train_tokens.dtype, torch.long)
        self.assertEqual(train_tags.dtype, torch.long)

        training_loader = data.train_dataloader()
        validation_loader = data.val_dataloader()
        self.assertIsInstance(training_loader.sampler, RandomSampler)
        self.assertIsInstance(validation_loader.sampler, SequentialSampler)
        self.assertTrue(training_loader.drop_last)
        self.assertTrue(validation_loader.drop_last)
        self.assertEqual(data._text_labels([0, 8]), ["O", "I-MISC"])
        with self.assertRaises(IndexError):
            CoNLL2003()._text_labels([9])

    def test_schema_guard_and_unsupported_test_stage_are_explicit(self) -> None:
        data = CoNLL2003(sequence_length=2)
        schema = data._build_schema([_sample(["train"], [0])])
        metadata = data.resolved_metadata

        returned_schema = data._build_schema([_sample(["replacement"], [0])])

        self.assertIs(returned_schema, schema)
        self.assertIs(data.schema, schema)
        self.assertIs(data.resolved_metadata, metadata)
        with self.assertRaisesRegex(
            NotImplementedError,
            "CoNLL2003 does not support the 'test' stage",
        ):
            data.setup("test")
        with self.assertRaisesRegex(
            NotImplementedError,
            "CoNLL2003 does not support the 'test' stage",
        ):
            data.test_dataloader()
        with self.assertRaisesRegex(RuntimeError, r"call setup\('fit'\)"):
            CoNLL2003().train_dataloader()

    def test_validation_failure_keeps_the_new_train_schema_installed(self) -> None:
        data = CoNLL2003(sequence_length=2)

        def source(_name: str, *, split: str):
            if split == "train":
                return [_sample(["train-token"], [0])]
            raise RuntimeError("validation unavailable")

        with (
            patch(
                "emperor.datasets.text.ner._conll2003.load_dataset",
                side_effect=source,
            ),
            self.assertRaisesRegex(RuntimeError, "validation unavailable"),
        ):
            data.setup("validate")

        self.assertIsNotNone(data.schema)
        self.assertEqual(data.schema.fingerprint, ("<pad>", "<unk>", "train-token"))
        self.assertFalse(hasattr(data, "val"))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import torch

from emperor.datasets.text.ner._conll2003 import CoNLL2003


def _sample(tokens: list[str], tags: list[int] | None = None) -> dict[str, list]:
    return {
        "tokens": tokens,
        "ner_tags": tags if tags is not None else [0] * len(tokens),
    }


class TestCoNLL2003SchemaOwnership(unittest.TestCase):
    def datasets(
        self,
        train_samples: list[dict[str, list]],
        validation_samples: list[dict[str, list]],
    ):
        def load_dataset(name: str, *, split: str):
            self.assertEqual(name, "conll2003")
            if split == "train":
                return train_samples
            if split == "validation":
                return validation_samples
            raise AssertionError(f"unexpected split: {split}")

        return load_dataset

    def test_fit_injects_one_train_schema_into_both_splits(self) -> None:
        train_samples = [_sample(["zebra", "apple"], [1, 2])]
        validation_samples = [_sample(["zebra", "novel"], [3, 4])]
        data = CoNLL2003(batch_size=1, sequence_length=4)

        with patch(
            "emperor.datasets.text.ner._conll2003.load_dataset",
            side_effect=self.datasets(train_samples, validation_samples),
        ):
            data._setup_fit()

        self.assertIs(data.train.schema, data.schema)
        self.assertIs(data.val.schema, data.schema)
        self.assertIs(data.train.schema, data.val.schema)
        train_tokens, train_tags = data.train[0]
        val_tokens, val_tags = data.val[0]
        torch.testing.assert_close(train_tokens, torch.tensor([3, 2, 0, 0]))
        torch.testing.assert_close(val_tokens, torch.tensor([3, 1, 0, 0]))
        torch.testing.assert_close(train_tags, torch.tensor([1, 2, 0, 0]))
        torch.testing.assert_close(val_tags, torch.tensor([3, 4, 0, 0]))
        self.assertEqual(train_tokens.dtype, torch.long)
        self.assertEqual(val_tokens.dtype, torch.long)

    def test_validation_only_reconstructs_the_training_schema(self) -> None:
        train_samples = [_sample(["zebra", "apple"])]
        validation_samples = [_sample(["zebra", "novel"])]
        fitted = CoNLL2003(sequence_length=2)
        restored = CoNLL2003(sequence_length=2)
        loader = self.datasets(train_samples, validation_samples)

        with patch(
            "emperor.datasets.text.ner._conll2003.load_dataset",
            side_effect=loader,
        ):
            fitted._setup_fit()
        with patch(
            "emperor.datasets.text.ner._conll2003.load_dataset",
            side_effect=loader,
        ) as load_dataset:
            restored._setup_validate()

        self.assertEqual(restored.schema.fingerprint, fitted.schema.fingerprint)
        self.assertEqual(load_dataset.call_args_list[0].kwargs["split"], "train")
        self.assertEqual(load_dataset.call_args_list[1].kwargs["split"], "validation")
        restored_tokens, _ = restored.val[0]
        fitted_tokens, _ = fitted.val[0]
        torch.testing.assert_close(restored_tokens, fitted_tokens)
        torch.testing.assert_close(restored_tokens, torch.tensor([3, 1]))

    def test_repeated_validation_reuses_the_existing_schema(self) -> None:
        train_samples = [_sample(["zebra", "apple"])]
        validation_samples = [_sample(["zebra", "novel"])]
        data = CoNLL2003(sequence_length=2)
        loader = self.datasets(train_samples, validation_samples)

        with patch(
            "emperor.datasets.text.ner._conll2003.load_dataset",
            side_effect=loader,
        ):
            data._setup_fit()
        original_schema = data.schema
        with patch(
            "emperor.datasets.text.ner._conll2003.load_dataset",
            side_effect=loader,
        ) as load_dataset:
            data._setup_validate()

        self.assertIs(data.schema, original_schema)
        self.assertEqual(load_dataset.call_count, 1)
        self.assertEqual(load_dataset.call_args.kwargs["split"], "validation")

    def test_instances_keep_independent_training_schemas(self) -> None:
        first = CoNLL2003(sequence_length=2)
        second = CoNLL2003(sequence_length=2)
        validation_samples = [_sample(["shared", "unknown"])]

        with patch(
            "emperor.datasets.text.ner._conll2003.load_dataset",
            side_effect=self.datasets(
                [_sample(["shared", "zebra"])],
                validation_samples,
            ),
        ):
            first._setup_fit()
        with patch(
            "emperor.datasets.text.ner._conll2003.load_dataset",
            side_effect=self.datasets(
                [_sample(["alpha", "shared", "beta"])],
                validation_samples,
            ),
        ):
            second._setup_fit()

        self.assertIsNot(first.schema, second.schema)
        self.assertNotEqual(first.schema.fingerprint, second.schema.fingerprint)
        first_tokens, _ = first.val[0]
        second_tokens, _ = second.val[0]
        torch.testing.assert_close(first_tokens, torch.tensor([2, 1]))
        torch.testing.assert_close(second_tokens, torch.tensor([4, 1]))


if __name__ == "__main__":
    unittest.main()

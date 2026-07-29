from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.text.masked_language_modeling._datasets import (
    PennTreebankMaskedLanguageModeling,
    WikiText2MaskedLanguageModeling,
    WikiText103MaskedLanguageModeling,
)


class _Vocabulary:
    def __init__(self, token_sequences, *, specials) -> None:
        tokens = {token for sequence in token_sequences for token in sequence}
        self._tokens = [*specials, *sorted(tokens.difference(specials))]
        self._indices = {token: index for index, token in enumerate(self._tokens)}

    def __call__(self, tokens: list[str]) -> list[int]:
        unknown_index = self._indices["[UNK]"]
        return [self._indices.get(token, unknown_index) for token in tokens]

    def __getitem__(self, token: str) -> int:
        return self._indices.get(token, self._indices["[UNK]"])

    def __len__(self) -> int:
        return len(self._tokens)

    def get_stoi(self) -> dict[str, int]:
        return dict(self._indices)

    def lookup_token(self, index: int) -> str:
        return self._tokens[index]

    def set_default_index(self, index: int) -> None:
        self.default_index = index


def _build_vocabulary(token_sequences, *, specials):
    return _Vocabulary(token_sequences, specials=specials)


class _MaskedCorpusSource:
    def __init__(self) -> None:
        self.requested_splits: list[str] = []
        self.corpus = {
            "train": ("zero one two three four five",),
            "valid": ("one two three four",),
            "test": ("two three four five",),
        }

    def __call__(self, *, root: str, split: str):
        self.requested_splits.append(split)
        return self.corpus[split]


class MaskedLanguageModelingDatasetTests(unittest.TestCase):
    dataset_types = (
        PennTreebankMaskedLanguageModeling,
        WikiText2MaskedLanguageModeling,
        WikiText103MaskedLanguageModeling,
    )

    def test_prepare_all_stages_and_loader_schema_are_consistent(self) -> None:
        for dataset_type in self.dataset_types:
            with (
                self.subTest(dataset=dataset_type.__name__),
                torch.random.fork_rng(devices=[]),
            ):
                torch.manual_seed(0)
                source = _MaskedCorpusSource()
                with (
                    patch.object(dataset_type, "torchtext_dataset", source),
                    patch(
                        "emperor.datasets.text.masked_language_modeling._datasets."
                        "build_vocab_from_iterator",
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset = dataset_type(
                        batch_size=2,
                        sequence_length=5,
                        mlm_probability=1.0,
                        num_workers=0,
                        drop_last=False,
                    )
                    dataset.prepare_data()
                    dataset.setup()

                self.assertEqual(
                    source.requested_splits,
                    [
                        "train",
                        "valid",
                        "test",
                        "train",
                        "train",
                        "valid",
                        "test",
                    ],
                )
                self.assertEqual(len(dataset.train), 2)
                self.assertEqual(len(dataset.val), 2)
                self.assertEqual(len(dataset.test), 2)
                self.assertIsInstance(dataset.train_dataloader().sampler, RandomSampler)
                self.assertIsInstance(
                    dataset.val_dataloader().sampler,
                    SequentialSampler,
                )
                self.assertIsInstance(
                    dataset.test_dataloader().sampler,
                    SequentialSampler,
                )

                for loader in (
                    dataset.train_dataloader(),
                    dataset.val_dataloader(),
                    dataset.test_dataloader(),
                ):
                    input_ids, labels, attention_mask = next(iter(loader))
                    self.assertEqual(input_ids.shape, torch.Size([2, 5]))
                    self.assertEqual(labels.shape, torch.Size([2, 5]))
                    self.assertEqual(attention_mask.shape, torch.Size([2, 5]))
                    self.assertEqual(input_ids.dtype, torch.long)
                    self.assertEqual(labels.dtype, torch.long)
                    self.assertEqual(attention_mask.dtype, torch.long)

                self.assertEqual(
                    dataset.resolved_metadata.vocab_size,
                    len(dataset.vocab),
                )
                self.assertEqual(
                    dataset.resolved_metadata.num_classes,
                    len(dataset.vocab),
                )
                self.assertEqual(
                    dataset.bert_special_token_ids(),
                    dataset.special_token_ids,
                )

    def test_dataset_build_requires_vocabulary_and_source_errors_propagate(
        self,
    ) -> None:
        for dataset_type in self.dataset_types:
            with self.subTest(dataset=dataset_type.__name__):
                source = _MaskedCorpusSource()
                with patch.object(dataset_type, "torchtext_dataset", source):
                    dataset = dataset_type(sequence_length=5, num_workers=0)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Vocabulary must be built before the dataset",
                ):
                    dataset._build_dataset(("alpha beta",))

                def unavailable(*, root: str, split: str):
                    raise RuntimeError(f"{split} unavailable")

                with (
                    patch.object(dataset_type, "torchtext_dataset", unavailable),
                    self.assertRaisesRegex(RuntimeError, "train unavailable"),
                ):
                    dataset.prepare_data()


if __name__ == "__main__":
    unittest.main()

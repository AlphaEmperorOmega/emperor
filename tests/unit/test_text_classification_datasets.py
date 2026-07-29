from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.text.classification._ag_news import AgNews
from emperor.datasets.text.classification._dbpedia import DBpedia
from emperor.datasets.text.classification._imdb import IMDB
from emperor.datasets.text.classification._yelp_review_full import YelpReviewFull


@dataclass(frozen=True)
class _ClassificationCase:
    dataset_type: type
    source_target: str
    vocabulary_target: str
    source_labels: tuple[object, object]
    expected_labels: tuple[int, int]
    expected_text_labels: tuple[str, str]


_CASES = (
    _ClassificationCase(
        AgNews,
        "emperor.datasets.text.classification._ag_news.AgNewsDataset",
        "emperor.datasets.text.classification._ag_news.build_vocab_from_iterator",
        (1, 4),
        (0, 3),
        ("World", "Sci/Tech"),
    ),
    _ClassificationCase(
        DBpedia,
        "emperor.datasets.text.classification._dbpedia.DBpediaDataset",
        "emperor.datasets.text.classification._dbpedia.build_vocab_from_iterator",
        (1, 14),
        (0, 13),
        ("Company", "Written Work"),
    ),
    _ClassificationCase(
        IMDB,
        "emperor.datasets.text.classification._imdb.IMDBDataset",
        "emperor.datasets.text.classification._imdb.build_vocab_from_iterator",
        ("neg", "pos"),
        (0, 1),
        ("negative", "positive"),
    ),
    _ClassificationCase(
        YelpReviewFull,
        "emperor.datasets.text.classification._yelp_review_full.YelpReviewFullDataset",
        "emperor.datasets.text.classification._yelp_review_full.build_vocab_from_iterator",
        (1, 5),
        (0, 4),
        ("1 star", "5 stars"),
    ),
)


class _Vocabulary:
    def __init__(self, token_sequences, *, specials) -> None:
        tokens = {token for sequence in token_sequences for token in sequence}
        self._tokens = [*specials, *sorted(tokens.difference(specials))]
        self._indices = {token: index for index, token in enumerate(self._tokens)}

    def __call__(self, tokens: list[str]) -> list[int]:
        unknown_index = self._indices["<unk>"]
        return [self._indices.get(token, unknown_index) for token in tokens]

    def __getitem__(self, token: str) -> int:
        return self._indices[token]

    def __len__(self) -> int:
        return len(self._tokens)

    def set_default_index(self, index: int) -> None:
        self.default_index = index


def _build_vocabulary(token_sequences, *, specials):
    return _Vocabulary(token_sequences, specials=specials)


class _TextSource:
    def __init__(self, case: _ClassificationCase) -> None:
        self.case = case
        self.calls: list[str] = []
        self.train = (
            (case.source_labels[0], "alpha beta"),
            (case.source_labels[1], "gamma delta"),
        )
        self.validation = (
            (case.source_labels[1], "alpha unseen"),
            (case.source_labels[0], "delta beta"),
        )

    def __call__(self, *, root: str, split: str):
        self.calls.append(split)
        return self.train if split == "train" else self.validation


class TextClassificationDatasetTests(unittest.TestCase):
    def test_prepare_fit_and_loader_contracts_are_consistent(self) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                source = _TextSource(case)
                dataset = case.dataset_type(batch_size=2, sequence_length=5)
                dataset.num_workers = 0
                with (
                    patch(case.source_target, side_effect=source),
                    patch(
                        case.vocabulary_target,
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset.prepare_data()
                    dataset.setup("fit")

                self.assertEqual(
                    source.calls,
                    ["train", "test", "train", "train", "test"],
                )
                train_inputs, train_labels = dataset.train.tensors
                validation_inputs, validation_labels = dataset.val.tensors
                self.assertEqual(train_inputs.shape, torch.Size([2, 5]))
                self.assertEqual(validation_inputs.shape, torch.Size([2, 5]))
                self.assertEqual(train_inputs.dtype, torch.long)
                self.assertEqual(train_labels.dtype, torch.long)
                torch.testing.assert_close(
                    train_labels,
                    torch.tensor(case.expected_labels),
                )
                torch.testing.assert_close(
                    validation_labels,
                    torch.tensor(case.expected_labels[::-1]),
                )
                self.assertEqual(
                    dataset.resolved_metadata.vocab_size,
                    len(dataset.vocab),
                )
                self.assertEqual(
                    dataset.resolved_metadata.num_classes,
                    dataset.num_classes,
                )
                self.assertIsInstance(dataset.train_dataloader().sampler, RandomSampler)
                self.assertIsInstance(
                    dataset.val_dataloader().sampler,
                    SequentialSampler,
                )
                self.assertEqual(
                    dataset._text_labels(case.expected_labels),
                    list(case.expected_text_labels),
                )

    def test_validation_rebuilds_training_schema_and_propagates_source_errors(
        self,
    ) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                source = _TextSource(case)
                dataset = case.dataset_type(batch_size=1, sequence_length=4)
                dataset.num_workers = 0
                with (
                    patch(case.source_target, side_effect=source),
                    patch(
                        case.vocabulary_target,
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset.setup("validate")

                self.assertEqual(source.calls, ["train", "test"])
                self.assertEqual(len(dataset.val_dataloader()), 2)

                with (
                    patch(case.source_target, side_effect=RuntimeError("unavailable")),
                    self.assertRaisesRegex(RuntimeError, "unavailable"),
                ):
                    case.dataset_type().setup("validate")


if __name__ == "__main__":
    unittest.main()

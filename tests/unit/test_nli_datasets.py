from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.text.nli._multi_nli import MultiNLI
from emperor.datasets.text.nli._snli import SNLI


@dataclass(frozen=True)
class _NLICase:
    dataset_type: type
    module_name: str
    source_name: str
    validation_split: str


_CASES = (
    _NLICase(SNLI, "emperor.datasets.text.nli._snli", "snli", "validation"),
    _NLICase(
        MultiNLI,
        "emperor.datasets.text.nli._multi_nli",
        "multi_nli",
        "validation_matched",
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


class _NLISource:
    def __init__(self, case: _NLICase) -> None:
        self.case = case
        self.calls: list[tuple[str, str]] = []
        self.train = (
            {"premise": "alpha beta", "hypothesis": "alpha", "label": 0},
            {"premise": "gamma", "hypothesis": "delta", "label": 2},
            {"premise": "ignored", "hypothesis": "sample", "label": -1},
        )
        self.validation = (
            {"premise": "alpha unseen", "hypothesis": "beta", "label": 1},
            {"premise": "ignored", "hypothesis": "again", "label": -1},
        )

    def __call__(self, name: str, *, split: str):
        self.calls.append((name, split))
        self._assert_name(name)
        return self.train if split == "train" else self.validation

    def _assert_name(self, name: str) -> None:
        if name != self.case.source_name:
            raise AssertionError(f"unexpected source: {name}")


class NLIDatasetTests(unittest.TestCase):
    def test_prepare_fit_filters_unlabelled_samples_and_preserves_batches(self) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                source = _NLISource(case)
                dataset = case.dataset_type(batch_size=1, sequence_length=4)
                dataset.num_workers = 0
                with (
                    patch(f"{case.module_name}.load_dataset", side_effect=source),
                    patch(
                        f"{case.module_name}.build_vocab_from_iterator",
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset.prepare_data()
                    dataset.setup("fit")

                self.assertEqual(
                    source.calls,
                    [
                        (case.source_name, "train"),
                        (case.source_name, case.validation_split),
                        (case.source_name, "train"),
                        (case.source_name, case.validation_split),
                    ],
                )
                self.assertEqual(len(dataset.train), 2)
                self.assertEqual(len(dataset.val), 1)
                premise, hypothesis, label = dataset.train[0]
                self.assertEqual(premise.shape, torch.Size([4]))
                self.assertEqual(hypothesis.shape, torch.Size([4]))
                self.assertEqual(premise.dtype, torch.long)
                self.assertEqual(hypothesis.dtype, torch.long)
                self.assertEqual(label.dtype, torch.long)
                self.assertEqual(label.item(), 0)
                self.assertEqual(dataset.resolved_metadata.vocab_size, len(dataset.vocab))
                self.assertEqual(dataset.resolved_metadata.num_classes, 3)
                self.assertIsInstance(dataset.train_dataloader().sampler, RandomSampler)
                self.assertIsInstance(
                    dataset.val_dataloader().sampler,
                    SequentialSampler,
                )
                self.assertEqual(
                    dataset._text_labels([0, 1, 2]),
                    ["entailment", "neutral", "contradiction"],
                )

    def test_validation_only_uses_validation_schema_and_propagates_errors(self) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                source = _NLISource(case)
                dataset = case.dataset_type(batch_size=1, sequence_length=3)
                dataset.num_workers = 0
                with (
                    patch(f"{case.module_name}.load_dataset", side_effect=source),
                    patch(
                        f"{case.module_name}.build_vocab_from_iterator",
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset.setup("validate")

                self.assertEqual(
                    source.calls,
                    [(case.source_name, case.validation_split)],
                )
                self.assertEqual(len(dataset.val_dataloader()), 1)

                with (
                    patch(
                        f"{case.module_name}.load_dataset",
                        side_effect=RuntimeError("unavailable"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "unavailable"),
                ):
                    case.dataset_type().setup("validate")


if __name__ == "__main__":
    unittest.main()

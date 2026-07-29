from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.text.summarization._cnn_dailymail import CnnDailyMail
from emperor.datasets.text.summarization._xsum import XSum


@dataclass(frozen=True)
class _SummarizationCase:
    dataset_type: type
    module_name: str
    source_args: tuple[str, ...]
    input_field: str
    target_field: str
    length_arguments: dict[str, int]


_CASES = (
    _SummarizationCase(
        CnnDailyMail,
        "emperor.datasets.text.summarization._cnn_dailymail",
        ("cnn_dailymail", "3.0.0"),
        "article",
        "highlights",
        {"article_length": 6, "summary_length": 5},
    ),
    _SummarizationCase(
        XSum,
        "emperor.datasets.text.summarization._xsum",
        ("xsum",),
        "document",
        "summary",
        {"document_length": 6, "summary_length": 5},
    ),
)


class _Vocabulary:
    def __init__(self, token_sequences, *, specials, max_tokens=None) -> None:
        tokens = {token for sequence in token_sequences for token in sequence}
        ordered = [*specials, *sorted(tokens.difference(specials))]
        self._tokens = ordered[:max_tokens] if max_tokens is not None else ordered
        self._indices = {token: index for index, token in enumerate(self._tokens)}

    def __call__(self, tokens: list[str]) -> list[int]:
        unknown_index = self._indices["<unk>"]
        return [self._indices.get(token, unknown_index) for token in tokens]

    def __getitem__(self, token: str) -> int:
        return self._indices[token]

    def __len__(self) -> int:
        return len(self._tokens)

    def lookup_token(self, index: int) -> str:
        return self._tokens[index]

    def set_default_index(self, index: int) -> None:
        self.default_index = index


def _build_vocabulary(token_sequences, *, specials, max_tokens=None):
    return _Vocabulary(
        token_sequences,
        specials=specials,
        max_tokens=max_tokens,
    )


class _SummarizationSource:
    def __init__(self, case: _SummarizationCase) -> None:
        self.case = case
        self.calls: list[tuple[tuple[str, ...], str]] = []
        self.train = (
            {case.input_field: "alpha beta gamma", case.target_field: "short alpha"},
            {case.input_field: "delta epsilon", case.target_field: "short delta"},
        )
        self.validation = (
            {case.input_field: "alpha unseen", case.target_field: "short beta"},
        )

    def __call__(self, *args: str, split: str):
        self.calls.append((args, split))
        if args != self.case.source_args:
            raise AssertionError(f"unexpected source: {args}")
        return self.train if split == "train" else self.validation


class SummarizationDatasetTests(unittest.TestCase):
    def test_prepare_fit_encodes_boundaries_and_preserves_loader_contract(self) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                source = _SummarizationSource(case)
                dataset = case.dataset_type(batch_size=1, **case.length_arguments)
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
                        (case.source_args, "train"),
                        (case.source_args, "validation"),
                        (case.source_args, "train"),
                        (case.source_args, "validation"),
                    ],
                )
                source_ids, target_ids = dataset.train[0]
                self.assertEqual(source_ids.shape, torch.Size([6]))
                self.assertEqual(target_ids.shape, torch.Size([5]))
                self.assertEqual(source_ids.dtype, torch.long)
                self.assertEqual(target_ids.dtype, torch.long)
                self.assertEqual(dataset._text_labels(source_ids[:2]), ["<bos>", "alpha"])
                self.assertEqual(
                    dataset._text_labels(target_ids[:3]),
                    ["<bos>", "short", "alpha"],
                )
                self.assertEqual(dataset._text_labels(source_ids[4:]), ["<eos>", "<pad>"])
                self.assertEqual(dataset.resolved_metadata.vocab_size, len(dataset.vocab))
                self.assertEqual(
                    dataset.resolved_metadata.num_classes,
                    len(dataset.vocab),
                )
                self.assertIsInstance(dataset.train_dataloader().sampler, RandomSampler)
                self.assertIsInstance(
                    dataset.val_dataloader().sampler,
                    SequentialSampler,
                )

    def test_validation_only_builds_local_schema_and_propagates_errors(self) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                source = _SummarizationSource(case)
                dataset = case.dataset_type(batch_size=1, **case.length_arguments)
                dataset.num_workers = 0
                with (
                    patch(f"{case.module_name}.load_dataset", side_effect=source),
                    patch(
                        f"{case.module_name}.build_vocab_from_iterator",
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset.setup("validate")

                self.assertEqual(source.calls, [(case.source_args, "validation")])
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

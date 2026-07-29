from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.text.language_modeling._wiki_text_103 import WikiText103


class _Vocabulary:
    def __init__(self, token_sequences, *, specials) -> None:
        tokens = {token for sequence in token_sequences for token in sequence}
        self._tokens = [*specials, *sorted(tokens.difference(specials))]
        self._indices = {token: index for index, token in enumerate(self._tokens)}

    def __getitem__(self, token: str) -> int:
        return self._indices.get(token, self._indices["<unk>"])

    def __len__(self) -> int:
        return len(self._tokens)

    def lookup_token(self, index: int) -> str:
        return self._tokens[index]

    def set_default_index(self, index: int) -> None:
        self.default_index = index


def _build_vocabulary(token_sequences, *, specials):
    return _Vocabulary(token_sequences, specials=specials)


class _WikiTextSource:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.corpus = {
            "train": ("zero one two three four five six seven",),
            "valid": ("one two three four five",),
        }

    def __call__(self, *, root: str, split: str):
        self.calls.append(split)
        return self.corpus[split]


class WikiText103DatasetTests(unittest.TestCase):
    source_target = (
        "emperor.datasets.text.language_modeling._wiki_text_103.WikiText103Dataset"
    )
    vocabulary_target = (
        "emperor.datasets.text.language_modeling._wiki_text_103."
        "build_vocab_from_iterator"
    )

    def test_prepare_fit_and_shifted_batch_contract_are_offline(self) -> None:
        source = _WikiTextSource()
        dataset = WikiText103(batch_size=1, sequence_length=3)
        dataset.num_workers = 0
        with (
            patch(self.source_target, side_effect=source),
            patch(self.vocabulary_target, side_effect=_build_vocabulary),
        ):
            dataset.prepare_data()
            dataset.setup("fit")

        self.assertEqual(
            source.calls,
            ["train", "valid", "train", "train", "valid"],
        )
        train_inputs, train_targets = dataset.train.tensors
        validation_inputs, validation_targets = dataset.val.tensors
        self.assertEqual(train_inputs.shape, torch.Size([2, 3]))
        self.assertEqual(train_targets.shape, torch.Size([2, 3]))
        self.assertEqual(validation_inputs.shape, torch.Size([1, 3]))
        self.assertEqual(validation_targets.shape, torch.Size([1, 3]))
        torch.testing.assert_close(train_inputs[:, 1:], train_targets[:, :-1])
        self.assertEqual(train_inputs.dtype, torch.long)
        self.assertEqual(dataset.resolved_metadata.vocab_size, len(dataset.vocab))
        self.assertEqual(
            dataset.resolved_metadata.flattened_input_dim,
            len(dataset.vocab),
        )
        self.assertEqual(dataset.resolved_metadata.num_classes, len(dataset.vocab))
        self.assertIsInstance(dataset.train_dataloader().sampler, RandomSampler)
        self.assertIsInstance(dataset.val_dataloader().sampler, SequentialSampler)
        self.assertEqual(dataset._text_labels(train_inputs[0, :2]), ["zero", "one"])

    def test_validation_reuses_training_vocabulary_source_and_propagates_errors(
        self,
    ) -> None:
        source = _WikiTextSource()
        dataset = WikiText103(batch_size=1, sequence_length=3)
        dataset.num_workers = 0
        with (
            patch(self.source_target, side_effect=source),
            patch(self.vocabulary_target, side_effect=_build_vocabulary),
        ):
            dataset.setup("validate")

        self.assertEqual(source.calls, ["train", "valid"])
        self.assertEqual(len(dataset.val_dataloader()), 1)

        with (
            patch(self.source_target, side_effect=RuntimeError("unavailable")),
            self.assertRaisesRegex(RuntimeError, "unavailable"),
        ):
            WikiText103().setup("validate")


if __name__ == "__main__":
    unittest.main()

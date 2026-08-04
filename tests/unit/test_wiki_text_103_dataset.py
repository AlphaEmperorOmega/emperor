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

    def test_literal_short_boundary_and_discarded_tail_windows(self) -> None:
        dataset = WikiText103(sequence_length=3)
        dataset.vocab = _Vocabulary(
            (("zero", "one", "two", "three", "four", "five", "six", "tail"),),
            specials=["<unk>"],
        )

        empty_inputs, empty_targets = dataset._build_dataset(iter(())).tensors
        exact_inputs, exact_targets = dataset._build_dataset(
            iter(("zero one two",))
        ).tensors
        one_inputs, one_targets = dataset._build_dataset(
            iter(("zero one two three",))
        ).tensors
        tail_inputs, tail_targets = dataset._build_dataset(
            iter(("zero one two three four five six tail",))
        ).tensors

        self.assertEqual(empty_inputs.shape, torch.Size([0, 3]))
        self.assertEqual(empty_targets.shape, torch.Size([0, 3]))
        self.assertEqual(exact_inputs.shape, torch.Size([0, 3]))
        self.assertEqual(exact_targets.shape, torch.Size([0, 3]))
        torch.testing.assert_close(one_inputs, torch.tensor([[8, 3, 7]]))
        torch.testing.assert_close(one_targets, torch.tensor([[3, 7, 6]]))
        torch.testing.assert_close(
            tail_inputs,
            torch.tensor([[8, 3, 7], [6, 2, 1]]),
        )
        torch.testing.assert_close(
            tail_targets,
            torch.tensor([[3, 7, 6], [2, 1, 4]]),
        )

    def test_repeated_stages_preserve_train_owned_vocabulary_and_metadata(self) -> None:
        source = _WikiTextSource()
        dataset = WikiText103(batch_size=1, sequence_length=3)
        dataset.num_workers = 0
        with (
            patch(self.source_target, side_effect=source),
            patch(
                self.vocabulary_target,
                side_effect=_build_vocabulary,
            ) as vocabulary_builder,
        ):
            dataset.setup("fit")
            vocabulary = dataset.vocab
            metadata = dataset.resolved_metadata
            dataset.setup("validate")
            dataset.setup("validate")

        self.assertEqual(
            source.calls,
            ["train", "train", "valid", "valid", "valid"],
        )
        self.assertEqual(vocabulary_builder.call_count, 1)
        self.assertIs(dataset.vocab, vocabulary)
        self.assertIs(dataset.resolved_metadata, metadata)
        self.assertEqual(dataset.vocab.default_index, 0)
        self.assertEqual(dataset.vocab["validation-only"], 0)
        self.assertTrue(dataset.train_dataloader().drop_last)
        self.assertTrue(dataset.val_dataloader().drop_last)

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

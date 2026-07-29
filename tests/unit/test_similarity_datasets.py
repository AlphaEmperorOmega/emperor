from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.text.similarity._stsb import STSb


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


class _SimilaritySource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.train = (
            {"sentence1": "alpha beta", "sentence2": "alpha", "score": 0.0},
            {"sentence1": "gamma", "sentence2": "gamma", "score": 5.0},
        )
        self.validation = (
            {"sentence1": "alpha unseen", "sentence2": "beta", "score": 2.5},
        )

    def __call__(self, name: str, subset: str, *, split: str):
        self.calls.append((name, subset, split))
        if (name, subset) != ("glue", "stsb"):
            raise AssertionError(f"unexpected source: {(name, subset)}")
        return self.train if split == "train" else self.validation


class SimilarityDatasetTests(unittest.TestCase):
    source_target = "emperor.datasets.text.similarity._stsb.load_dataset"
    vocabulary_target = (
        "emperor.datasets.text.similarity._stsb.build_vocab_from_iterator"
    )

    def test_prepare_fit_normalizes_scores_and_preserves_batch_contract(self) -> None:
        source = _SimilaritySource()
        dataset = STSb(batch_size=1, sequence_length=4)
        dataset.num_workers = 0
        with (
            patch(self.source_target, side_effect=source),
            patch(self.vocabulary_target, side_effect=_build_vocabulary),
        ):
            dataset.prepare_data()
            dataset.setup("fit")

        self.assertEqual(
            source.calls,
            [
                ("glue", "stsb", "train"),
                ("glue", "stsb", "validation"),
                ("glue", "stsb", "train"),
                ("glue", "stsb", "validation"),
            ],
        )
        first_sentence, second_sentence, low_score = dataset.train[0]
        _, _, high_score = dataset.train[1]
        _, _, middle_score = dataset.val[0]
        self.assertEqual(first_sentence.shape, torch.Size([4]))
        self.assertEqual(second_sentence.shape, torch.Size([4]))
        self.assertEqual(first_sentence.dtype, torch.long)
        self.assertEqual(low_score.dtype, torch.float32)
        torch.testing.assert_close(low_score, torch.tensor(0.0))
        torch.testing.assert_close(middle_score, torch.tensor(0.5))
        torch.testing.assert_close(high_score, torch.tensor(1.0))
        self.assertEqual(dataset.resolved_metadata.vocab_size, len(dataset.vocab))
        self.assertEqual(dataset.resolved_metadata.num_classes, 1)
        self.assertIsInstance(dataset.train_dataloader().sampler, RandomSampler)
        self.assertIsInstance(dataset.val_dataloader().sampler, SequentialSampler)
        self.assertEqual(dataset._text_labels([0.0, 0.5, 1.0]), ["0.00", "0.50", "1.00"])

    def test_validation_only_and_dependency_errors_are_explicit(self) -> None:
        source = _SimilaritySource()
        dataset = STSb(batch_size=1, sequence_length=3)
        dataset.num_workers = 0
        with (
            patch(self.source_target, side_effect=source),
            patch(self.vocabulary_target, side_effect=_build_vocabulary),
        ):
            dataset.setup("validate")

        self.assertEqual(source.calls, [("glue", "stsb", "validation")])
        self.assertEqual(len(dataset.val_dataloader()), 1)

        with (
            patch(self.source_target, side_effect=RuntimeError("unavailable")),
            self.assertRaisesRegex(RuntimeError, "unavailable"),
        ):
            STSb().setup("validate")


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from emperor.datasets.text.language_modeling import PennTreebank, WikiText2
from emperor.experiments.tasks import ExperimentTask, experiment_task_name


class _InMemoryCorpusMixin:
    corpus = {
        "train": ["zero one two three four five six"],
        "valid": ["one two three four"],
        "test": ["two three four five"],
    }

    def __init__(self, *args, **kwargs):
        self.requested_splits = []
        super().__init__(*args, **kwargs)

    def _dataset(self, split: str):
        self.requested_splits.append(split)
        return iter(self.corpus[split])


class InMemoryWikiText2(_InMemoryCorpusMixin, WikiText2):
    pass


class InMemoryPennTreebank(_InMemoryCorpusMixin, PennTreebank):
    pass


class FakeLegacyCorpus:
    calls = []

    @classmethod
    def splits(cls, field, root):
        cls.calls.append(root)
        return tuple(
            SimpleNamespace(
                examples=[SimpleNamespace(text=[split, "one", "two"])],
            )
            for split in ("train", "valid", "test")
        )


class TestCausalLanguageModelingDatasets(unittest.TestCase):
    def dataset_types(self):
        return (InMemoryWikiText2, InMemoryPennTreebank)

    def test_invalid_batch_and_sequence_lengths_are_rejected(self):
        for dataset_type in self.dataset_types():
            with self.subTest(dataset=dataset_type.__name__, field="batch_size"):
                with self.assertRaises(ValueError):
                    dataset_type(batch_size=0)
            with self.subTest(
                dataset=dataset_type.__name__,
                field="sequence_length",
            ):
                with self.assertRaises(ValueError):
                    dataset_type(sequence_length=0)

    def test_prepare_data_requests_all_three_corpus_splits(self):
        for dataset_type in self.dataset_types():
            with self.subTest(dataset=dataset_type.__name__):
                dataset = dataset_type(num_workers=0)
                dataset.prepare_data()
                self.assertEqual(dataset.requested_splits, ["train", "valid", "test"])

    def test_task_identity_uses_the_public_cli_name(self):
        self.assertEqual(
            experiment_task_name(ExperimentTask.CAUSAL_LANGUAGE_MODELING),
            "causal-language-modeling",
        )

    def test_setup_builds_train_validation_and_test_splits(self):
        for dataset_type in self.dataset_types():
            with self.subTest(dataset=dataset_type.__name__):
                dataset = dataset_type(
                    batch_size=1,
                    sequence_length=3,
                    num_workers=0,
                    drop_last=False,
                )
                dataset.setup("fit")
                self.assertEqual(len(dataset.train), 2)
                self.assertEqual(len(dataset.val), 1)
                dataset.setup("test")
                self.assertEqual(len(dataset.test), 1)
                inputs, targets = next(iter(dataset.test_dataloader()))
                self.assertEqual(tuple(inputs.shape), (1, 3))
                self.assertEqual(tuple(targets.shape), (1, 3))

    def test_setup_without_stage_prepares_every_split(self):
        for dataset_type in self.dataset_types():
            with self.subTest(dataset=dataset_type.__name__):
                dataset = dataset_type(sequence_length=3, num_workers=0)
                dataset.setup()
                self.assertEqual(len(dataset.train), 2)
                self.assertEqual(len(dataset.val), 1)
                self.assertEqual(len(dataset.test), 1)

    def test_legacy_torchtext_split_api_is_loaded_once_and_cached(self):
        cases = (
            (
                WikiText2,
                "emperor.datasets.text.language_modeling.wiki_text_2."
                "WikiText2Dataset",
                "emperor.datasets.text.language_modeling.wiki_text_2."
                "_legacy_text_field",
            ),
            (
                PennTreebank,
                "emperor.datasets.text.language_modeling.penn_treebank."
                "PennTreebankDataset",
                "emperor.datasets.text.language_modeling.penn_treebank."
                "_legacy_text_field",
            ),
        )
        for dataset_type, patch_target, field_patch_target in cases:
            with self.subTest(dataset=dataset_type.__name__):
                FakeLegacyCorpus.calls = []
                with (
                    patch(patch_target, FakeLegacyCorpus),
                    patch(field_patch_target, return_value=object()),
                ):
                    dataset = dataset_type(root="fake-root", num_workers=0)
                    self.assertEqual(
                        list(dataset._dataset("valid")),
                        ["valid", "one", "two"],
                    )
                    self.assertEqual(
                        list(dataset._dataset("test")),
                        ["test", "one", "two"],
                    )
                self.assertEqual(FakeLegacyCorpus.calls, ["fake-root"])

    def test_windows_are_shifted_by_exactly_one_token(self):
        for dataset_type in self.dataset_types():
            with self.subTest(dataset=dataset_type.__name__):
                dataset = dataset_type(sequence_length=3, num_workers=0)
                dataset._build_vocab()
                token_dataset = dataset._build_dataset(iter(dataset.corpus["train"]))
                inputs, targets = token_dataset.tensors

                torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])
                self.assertEqual(int(targets[0, -1]), int(inputs[1, 0]))

    def test_short_corpora_produce_empty_rank_two_tensors(self):
        for dataset_type in self.dataset_types():
            with self.subTest(dataset=dataset_type.__name__):
                dataset = dataset_type(sequence_length=8, num_workers=0)
                dataset._build_vocab()
                token_dataset = dataset._build_dataset(iter(["one two"]))
                inputs, targets = token_dataset.tensors
                self.assertEqual(tuple(inputs.shape), (0, 8))
                self.assertEqual(tuple(targets.shape), (0, 8))

    def test_vocabulary_helpers_round_trip_known_tokens(self):
        for dataset_type in self.dataset_types():
            with self.subTest(dataset=dataset_type.__name__):
                dataset = dataset_type(num_workers=0)
                token_ids = dataset.encode_text("one two three")
                self.assertEqual(dataset.decode_ids(token_ids), "one two three")
                self.assertEqual(
                    dataset.decode_batch([token_ids, token_ids]),
                    ["one two three", "one two three"],
                )


if __name__ == "__main__":
    unittest.main()

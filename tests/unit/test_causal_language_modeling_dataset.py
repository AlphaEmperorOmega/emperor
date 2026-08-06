import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

import emperor.datasets.text.language_modeling._penn_treebank as penn_module
import emperor.datasets.text.language_modeling._wiki_text_2 as wiki_module
import emperor.datasets.text.language_modeling._wiki_text_103 as wiki103_module
from emperor.datasets.text.language_modeling import (
    PennTreebank,
    WikiText2,
    WikiText103,
)
from emperor.experiments import (
    ExperimentTask,
    experiment_task_name,
)


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


class InMemoryWikiText103(_InMemoryCorpusMixin, WikiText103):
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


class _ModernVocabulary:
    def __init__(self) -> None:
        self.mapping = {"<unk>": 0, "one": 1, "two": 2}
        self.tokens = ["<unk>", "one", "two"]
        self.default_index = None

    def __getitem__(self, token: str) -> int:
        return self.mapping[token]

    def set_default_index(self, index: int) -> None:
        self.default_index = index

    def lookup_token(self, index: int) -> str:
        return self.tokens[index]


class TestCausalLanguageModelingDatasets(unittest.TestCase):
    def dataset_types(self):
        return (
            InMemoryWikiText2,
            InMemoryPennTreebank,
            InMemoryWikiText103,
        )

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
                "emperor.datasets.text.language_modeling._wiki_text_2.WikiText2Dataset",
                "emperor.datasets.text.language_modeling._wiki_text_2."
                "_legacy_text_field",
            ),
            (
                PennTreebank,
                "emperor.datasets.text.language_modeling._penn_treebank."
                "PennTreebankDataset",
                "emperor.datasets.text.language_modeling._penn_treebank."
                "_legacy_text_field",
            ),
            (
                WikiText103,
                "emperor.datasets.text.language_modeling._wiki_text_103."
                "WikiText103Dataset",
                "emperor.datasets.text.language_modeling._wiki_text_103."
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

    def test_modern_vocab_and_legacy_field_compatibility_helpers_are_literal(self):
        for module in (penn_module, wiki_module, wiki103_module):
            with self.subTest(module=module.__name__):
                vocabulary = _ModernVocabulary()

                module._set_unknown_default(vocabulary)
                decoded = module._lookup_token(vocabulary, 2)
                field = module._legacy_text_field(str.split)

                self.assertEqual(vocabulary.default_index, 0)
                self.assertEqual(decoded, "two")
                self.assertEqual(field.tokenize("one two"), ["one", "two"])

    def test_modern_provider_path_requests_the_literal_split(self):
        cases = (
            (PennTreebank, penn_module, "PennTreebankDataset"),
            (WikiText2, wiki_module, "WikiText2Dataset"),
            (WikiText103, wiki103_module, "WikiText103Dataset"),
        )
        for dataset_type, module, provider_name in cases:
            with self.subTest(dataset=dataset_type.__name__):
                calls: list[tuple[str, str]] = []

                def provider(*, root: str, split: str, calls=calls):
                    calls.append((root, split))
                    return iter(("one two",))

                with patch.object(module, provider_name, new=provider):
                    data = dataset_type(root="offline-root", num_workers=0)
                    text_units = tuple(data._dataset("valid"))

                self.assertEqual(text_units, ("one two",))
                self.assertEqual(calls, [("offline-root", "valid")])

    def test_wikitext_providers_use_the_maintained_hugging_face_source(self):
        cases = (
            (wiki_module, "WikiText2Dataset", "wikitext-2-v1"),
            (wiki103_module, "WikiText103Dataset", "wikitext-103-v1"),
        )
        for module, provider_name, config_name in cases:
            with self.subTest(config=config_name):
                with patch.object(
                    module,
                    "load_dataset",
                    return_value=({"text": "one two"}, {"text": "three four"}),
                ) as loader:
                    text_units = tuple(
                        getattr(module, provider_name)(
                            root="cache-root",
                            split="valid",
                        )
                    )

                self.assertEqual(text_units, ("one two", "three four"))
                loader.assert_called_once_with(
                    "Salesforce/wikitext",
                    config_name,
                    split="validation",
                    cache_dir="cache-root",
                )

    def test_validation_preconditions_and_encode_autobuild_are_exact(self):
        for dataset_type in self.dataset_types():
            with self.subTest(dataset=dataset_type.__name__):
                fresh = dataset_type(sequence_length=3, num_workers=0)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Vocabulary must be built before the dataset",
                ):
                    fresh._build_dataset(iter(("one two",)))
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Vocabulary must be built before decoding IDs",
                ):
                    fresh._text_labels([0])

                self.assertEqual(fresh.encode_text("one two three"), [3, 6, 5])
                vocabulary = fresh.vocab
                self.assertEqual(fresh.encode_text("unknown one"), [0, 3])
                self.assertIs(fresh.vocab, vocabulary)

                validating = dataset_type(sequence_length=3, num_workers=0)
                validating.setup("validate")
                self.assertEqual(validating.requested_splits, ["train", "valid"])
                self.assertEqual(len(validating.val), 1)

    def test_seeded_training_and_validation_loader_order_is_literal(self):
        for dataset_type in self.dataset_types():
            with self.subTest(dataset=dataset_type.__name__):
                data = dataset_type(
                    batch_size=2,
                    sequence_length=1,
                    num_workers=0,
                    drop_last=False,
                    seed=7,
                )
                literal_dataset = torch.utils.data.TensorDataset(torch.arange(4))
                data.train = literal_dataset
                data.val = literal_dataset

                training_loader = data.get_dataloader(train=True)
                validation_loader = data.get_dataloader(train=False)
                training_order = [
                    value for batch in training_loader for value in batch[0].tolist()
                ]
                validation_order = [
                    value for batch in validation_loader for value in batch[0].tolist()
                ]

                self.assertEqual(training_order, [1, 3, 0, 2])
                self.assertEqual(validation_order, [0, 1, 2, 3])
                self.assertIsInstance(training_loader.sampler, RandomSampler)
                self.assertIsInstance(validation_loader.sampler, SequentialSampler)
                self.assertFalse(training_loader.drop_last)
                self.assertFalse(validation_loader.drop_last)

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

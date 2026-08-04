import random
import unittest
from dataclasses import dataclass

import torch
from torch.utils.data import SequentialSampler

from emperor.datasets.text._bert_vocabulary import (
    BertSpecialTokenIds,
    get_bert_special_token_ids,
)
from emperor.datasets.text.bert_pretraining import (
    PennTreebankBertPretraining,
    WikiText2BertPretraining,
)
from emperor.datasets.text.bert_pretraining._collation import (
    BertPretrainingCollator,
)
from emperor.datasets.text.bert_pretraining._datasets import (
    _legacy_dataset_text_units,
    _split_on_whitespace,
)
from emperor.datasets.text.bert_pretraining._examples import (
    build_bert_pretraining_examples,
    build_bert_sentence_pair_inputs,
)
from emperor.datasets.text.bert_pretraining._next_sentence import (
    build_bert_next_sentence_pairs,
)
from emperor.datasets.text.bert_pretraining._tokenizer import (
    train_local_wordpiece_tokenizer,
)

OFFLINE_TEXT_SPLITS = {
    "train": (
        "a b c",
        "d e f",
        "g h i",
        "j k l",
        "m n o",
    ),
    "valid": (
        "a d g",
        "b e h",
        "c f i",
    ),
    "test": (
        "g j m",
        "h k n",
        "i l o",
    ),
}


@dataclass
class OfflineLegacyExample:
    text: list[str]


class OfflineLegacyDataset:
    def __init__(self, text_units: tuple[str, ...]):
        tokens = []
        for text_unit in text_units:
            tokens.extend(text_unit.split())
            tokens.append("<eos>")
        self.examples = [OfflineLegacyExample(tokens)]


class OfflineLegacyTextSource:
    @classmethod
    def splits(cls, text_field, root):
        return tuple(
            OfflineLegacyDataset(OFFLINE_TEXT_SPLITS[split])
            for split in ("train", "valid", "test")
        )


class OfflinePennTreebankBertPretraining(PennTreebankBertPretraining):
    torchtext_dataset = staticmethod(OfflineLegacyTextSource)


class OfflineWikiText2BertPretraining(WikiText2BertPretraining):
    torchtext_dataset = staticmethod(OfflineLegacyTextSource)


class OfflineModernTextSource:
    calls: list[tuple[str, str]] = []

    @classmethod
    def load(cls, *, root: str, split: str):
        cls.calls.append((root, split))
        return iter(OFFLINE_TEXT_SPLITS[split])


class OfflineModernPennTreebankBertPretraining(PennTreebankBertPretraining):
    torchtext_dataset = staticmethod(OfflineModernTextSource.load)


@dataclass(frozen=True)
class _LiteralEncoding:
    ids: list[int]


class _LiteralTokenizer:
    def __init__(self, token_ids_by_text: dict[str, list[int]]) -> None:
        self.token_ids_by_text = token_ids_by_text

    def encode(self, text: str) -> _LiteralEncoding:
        return _LiteralEncoding(self.token_ids_by_text[text])


class TestBertPretrainingDatasetHelpers(unittest.TestCase):
    def preset(self) -> BertSpecialTokenIds:
        return BertSpecialTokenIds(pad=0, unk=1, cls=2, sep=3, mask=4)

    def test_dataset_rejects_sequence_lengths_without_pair_capacity(self):
        with self.assertRaisesRegex(ValueError, "sequence_length must be at least 5"):
            OfflinePennTreebankBertPretraining(sequence_length=4)

    def test_prepare_requests_every_modern_provider_split(self):
        OfflineModernTextSource.calls.clear()
        data = OfflineModernPennTreebankBertPretraining(
            root="offline-root",
            num_workers=0,
        )

        data.prepare_data()

        self.assertEqual(
            OfflineModernTextSource.calls,
            [
                ("offline-root", "train"),
                ("offline-root", "valid"),
                ("offline-root", "test"),
            ],
        )

    def test_unseeded_dataset_delegates_random_state_to_callers(self):
        data = OfflineModernPennTreebankBertPretraining(seed=None, num_workers=0)

        self.assertIsNone(data._rng())
        self.assertIsNone(data._rng(2))
        self.assertIsNone(data._generator())
        self.assertIsNone(data._generator(2))

    def test_validate_and_test_stages_reuse_train_owned_tokenizer(self):
        OfflineModernTextSource.calls.clear()
        data = OfflineModernPennTreebankBertPretraining(
            batch_size=2,
            sequence_length=8,
            target_vocab_size=32,
            num_workers=0,
            drop_last=False,
            seed=7,
        )

        data.setup("validate")
        tokenizer = data.tokenizer
        validation_loader = data.val_dataloader()
        data.setup("test")
        test_loader = data.test_dataloader()

        self.assertEqual(
            [split for _, split in OfflineModernTextSource.calls],
            ["train", "valid", "test"],
        )
        self.assertIs(data.tokenizer, tokenizer)
        self.assertIsInstance(validation_loader.sampler, SequentialSampler)
        self.assertIsInstance(test_loader.sampler, SequentialSampler)
        self.assertEqual(next(iter(validation_loader))[0].shape, torch.Size([2, 8]))
        self.assertEqual(next(iter(test_loader))[0].shape, torch.Size([2, 8]))

    def test_dataset_build_requires_a_train_owned_tokenizer(self):
        data = OfflineModernPennTreebankBertPretraining(num_workers=0)

        with self.assertRaisesRegex(
            RuntimeError,
            "Tokenizer must be built before the dataset",
        ):
            data._build_dataset(iter(("alpha", "beta")), rng=None)

    def test_loader_build_requires_the_masking_collator(self):
        data = OfflineModernPennTreebankBertPretraining(num_workers=0)
        empty_dataset = torch.utils.data.TensorDataset(torch.empty((0, 8)))

        with self.assertRaisesRegex(
            RuntimeError,
            "Tokenizer must be built before creating loaders",
        ):
            data._dataloader(empty_dataset, train=False)

    def test_empty_text_units_build_ranked_long_dataset_tensors(self):
        data = OfflineModernPennTreebankBertPretraining(
            sequence_length=8,
            target_vocab_size=32,
            num_workers=0,
        )
        data.bert_special_token_ids()

        dataset = data._build_dataset(iter(("", "   ")), rng=None)

        self.assertEqual(
            tuple(tensor.shape for tensor in dataset.tensors),
            (torch.Size([0, 8]), torch.Size([0, 8]), torch.Size([0])),
        )
        self.assertTrue(all(tensor.dtype == torch.long for tensor in dataset.tensors))

    def test_special_ids_and_text_labels_auto_build_tokenizer_once(self):
        OfflineModernTextSource.calls.clear()
        data = OfflineModernPennTreebankBertPretraining(
            target_vocab_size=32,
            num_workers=0,
        )

        labels = data._text_labels([0, 1, 999])
        tokenizer = data.tokenizer
        token_ids = data.bert_special_token_ids()
        repeated_labels = data._text_labels([token_ids.pad, token_ids.unk])
        second_token_ids = data.bert_special_token_ids()

        self.assertEqual(token_ids, self.preset())
        self.assertEqual(second_token_ids, self.preset())
        self.assertEqual(labels, ["[PAD]", "[UNK]", "[UNK]"])
        self.assertEqual(repeated_labels, ["[PAD]", "[UNK]"])
        self.assertIs(data.tokenizer, tokenizer)
        self.assertEqual(
            [split for _, split in OfflineModernTextSource.calls],
            ["train"],
        )

    def test_legacy_text_helpers_skip_empty_eos_and_flush_final_buffer(self):
        legacy_dataset = OfflineLegacyDataset(())
        legacy_dataset.examples = [
            OfflineLegacyExample(["<eos>", "alpha", "<eos>", "<eos>", "beta", "gamma"])
        ]

        text_units = tuple(_legacy_dataset_text_units(legacy_dataset))

        self.assertEqual(_split_on_whitespace(" alpha\t beta \n"), ["alpha", "beta"])
        self.assertEqual(text_units, ("alpha", "beta gamma"))

    def test_local_wordpiece_tokenizer_uses_fixed_bert_special_ids(self):
        tokenizer = train_local_wordpiece_tokenizer(
            ["Hello world", "hello there world"],
            vocab_size=24,
        )

        token_ids = get_bert_special_token_ids(tokenizer)

        self.assertEqual(token_ids, self.preset())
        self.assertEqual(tokenizer.token_to_id("[PAD]"), 0)
        self.assertEqual(tokenizer.token_to_id("[MASK]"), 4)

    def test_local_wordpiece_tokenizer_uses_unknown_fallback_for_blank_corpus(self):
        for text_units in ([], ["", "   "]):
            with self.subTest(text_units=text_units):
                tokenizer = train_local_wordpiece_tokenizer(
                    text_units,
                    vocab_size=12,
                )

                self.assertEqual(get_bert_special_token_ids(tokenizer), self.preset())
                self.assertEqual(tokenizer.encode("unseen-token").ids, [1, 1, 1])

    def test_sentence_pair_inputs_have_cls_sep_token_types_and_padding(self):
        input_ids, token_type_ids = build_bert_sentence_pair_inputs(
            tokens_a=[5],
            tokens_b=[6],
            sequence_length=7,
            special_token_ids=self.preset(),
        )

        torch.testing.assert_close(
            input_ids,
            torch.tensor([2, 5, 3, 6, 3, 0, 0]),
        )
        torch.testing.assert_close(
            token_type_ids,
            torch.tensor([0, 0, 0, 1, 1, 0, 0]),
        )

    def test_sentence_pair_inputs_truncate_longest_first(self):
        input_ids, token_type_ids = build_bert_sentence_pair_inputs(
            tokens_a=[5, 6, 7],
            tokens_b=[8, 9],
            sequence_length=7,
            special_token_ids=self.preset(),
        )

        torch.testing.assert_close(
            input_ids,
            torch.tensor([2, 5, 6, 3, 8, 9, 3]),
        )
        torch.testing.assert_close(
            token_type_ids,
            torch.tensor([0, 0, 0, 0, 1, 1, 1]),
        )

    def test_sentence_pair_inputs_reject_an_empty_segment(self):
        for tokens_a, tokens_b in (([], [5]), ([5], [])):
            with self.subTest(tokens_a=tokens_a, tokens_b=tokens_b):
                with self.assertRaises(ValueError) as raised:
                    build_bert_sentence_pair_inputs(
                        tokens_a=tokens_a,
                        tokens_b=tokens_b,
                        sequence_length=5,
                        special_token_ids=self.preset(),
                    )
                self.assertEqual(
                    str(raised.exception),
                    "Both sentence-pair segments must contain tokens.",
                )

    def test_sentence_pair_inputs_require_room_for_both_segments(self):
        with self.assertRaises(ValueError) as raised:
            build_bert_sentence_pair_inputs(
                tokens_a=[5],
                tokens_b=[6],
                sequence_length=4,
                special_token_ids=self.preset(),
            )
        self.assertEqual(
            str(raised.exception),
            "sequence_length must be at least 5 for BERT pairs.",
        )

    def test_sentence_pair_inputs_truncate_the_longer_second_segment(self):
        input_ids, token_type_ids = build_bert_sentence_pair_inputs(
            tokens_a=[5, 6],
            tokens_b=[7, 8, 9],
            sequence_length=7,
            special_token_ids=self.preset(),
        )

        torch.testing.assert_close(input_ids, torch.tensor([2, 5, 6, 3, 7, 8, 3]))
        torch.testing.assert_close(
            token_type_ids,
            torch.tensor([0, 0, 0, 0, 1, 1, 1]),
        )

    def test_sentence_pair_input_tie_truncates_the_first_segment(self):
        input_ids, token_type_ids = build_bert_sentence_pair_inputs(
            tokens_a=[5, 6, 7],
            tokens_b=[8, 9, 10],
            sequence_length=8,
            special_token_ids=self.preset(),
        )

        torch.testing.assert_close(
            input_ids,
            torch.tensor([2, 5, 6, 3, 8, 9, 10, 3]),
        )
        torch.testing.assert_close(
            token_type_ids,
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        )

    def test_next_sentence_pairs_label_adjacent_and_random_next(self):
        pairs = build_bert_next_sentence_pairs(
            ["alpha", "beta", "gamma", "delta"],
            random_next_probability=1.0,
            rng=random.Random(0),
        )

        self.assertGreater(len(pairs), 0)
        for pair in pairs:
            with self.subTest(sentence_a_index=pair.sentence_a_index):
                self.assertEqual(pair.next_sentence_label, 1)
                self.assertNotEqual(pair.sentence_b_index, pair.sentence_a_index)
                self.assertNotEqual(pair.sentence_b_index, pair.sentence_a_index + 1)

        positive_pairs = build_bert_next_sentence_pairs(
            ["alpha", "beta", "gamma", "delta"],
            random_next_probability=0.0,
            rng=random.Random(0),
        )
        for pair in positive_pairs:
            with self.subTest(positive_sentence_a_index=pair.sentence_a_index):
                self.assertEqual(pair.next_sentence_label, 0)
                self.assertEqual(pair.sentence_b_index, pair.sentence_a_index + 1)

    def test_next_sentence_pairs_reject_invalid_random_probability(self):
        for probability in (-0.1, 1.1):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    ValueError,
                    "random_next_probability must be between 0.0 and 1.0",
                ):
                    build_bert_next_sentence_pairs(
                        ["alpha", "beta"],
                        random_next_probability=probability,
                    )

    def test_next_sentence_pairs_require_two_normalized_units(self):
        self.assertEqual(
            build_bert_next_sentence_pairs(
                ["  ", "only unit", ""],
                random_next_probability=1.0,
                rng=random.Random(0),
            ),
            [],
        )

    def test_build_bert_pretraining_examples_uses_tokenizer_and_pair_labels(self):
        tokenizer = train_local_wordpiece_tokenizer(
            ["alpha beta", "gamma delta", "epsilon zeta"],
            vocab_size=32,
        )
        examples = build_bert_pretraining_examples(
            ["alpha beta", "gamma delta", "epsilon zeta"],
            tokenizer=tokenizer,
            sequence_length=8,
            special_token_ids=get_bert_special_token_ids(tokenizer),
            random_next_probability=0.0,
            rng=random.Random(0),
        )

        self.assertGreater(len(examples), 0)
        self.assertEqual(examples[0].input_ids.shape, torch.Size([8]))
        self.assertEqual(examples[0].token_type_ids.shape, torch.Size([8]))
        self.assertEqual(examples[0].next_sentence_label.item(), 0)

    def test_example_builder_skips_pairs_with_an_empty_tokenizer_encoding(self):
        examples = build_bert_pretraining_examples(
            ["empty", "tokens"],
            tokenizer=_LiteralTokenizer({"empty": [], "tokens": [5]}),
            sequence_length=8,
            special_token_ids=self.preset(),
            random_next_probability=0.0,
            rng=random.Random(0),
        )

        self.assertEqual(examples, [])

    def test_example_builder_skips_pairs_that_cannot_fit_the_sequence(self):
        examples = build_bert_pretraining_examples(
            ["alpha", "beta"],
            tokenizer=_LiteralTokenizer({"alpha": [5], "beta": [6]}),
            sequence_length=4,
            special_token_ids=self.preset(),
            random_next_probability=0.0,
            rng=random.Random(0),
        )

        self.assertEqual(examples, [])

    def test_bert_pretraining_collator_outputs_canonical_batch(self):
        token_ids = self.preset()
        collator = BertPretrainingCollator(
            special_token_ids=token_ids,
            vocab_size=12,
            mlm_probability=1.0,
            mask_replace_probability=1.0,
            random_replace_probability=0.0,
        )

        batch = [
            (
                torch.tensor([2, 5, 3, 6, 3, 0]),
                torch.tensor([0, 0, 0, 1, 1, 0]),
                torch.tensor(1),
            )
        ]

        (
            input_ids,
            mlm_labels,
            attention_mask,
            token_type_ids,
            next_sentence_labels,
        ) = collator(batch)

        torch.testing.assert_close(input_ids, torch.tensor([[2, 4, 3, 4, 3, 0]]))
        torch.testing.assert_close(
            mlm_labels,
            torch.tensor([[-100, 5, -100, 6, -100, -100]]),
        )
        torch.testing.assert_close(attention_mask, torch.tensor([[1, 1, 1, 1, 1, 0]]))
        torch.testing.assert_close(token_type_ids, torch.tensor([[0, 0, 0, 1, 1, 0]]))
        torch.testing.assert_close(next_sentence_labels, torch.tensor([1]))

    def test_bert_pretraining_collator_rejects_an_empty_batch(self):
        collator = BertPretrainingCollator(
            special_token_ids=self.preset(),
            vocab_size=12,
        )

        with self.assertRaisesRegex(
            ValueError,
            "BertPretrainingCollator received no samples",
        ):
            collator([])

    def test_bert_pretraining_collator_rejects_malformed_samples(self):
        collator = BertPretrainingCollator(
            special_token_ids=self.preset(),
            vocab_size=12,
        )

        for sample in (object(), (torch.tensor([5]), torch.tensor([0]))):
            with self.subTest(sample=sample):
                with self.assertRaisesRegex(
                    TypeError,
                    "BertPretrainingCollator expects samples containing",
                ):
                    collator([sample])

    def test_dataset_classes_expose_pretraining_metadata(self):
        for dataset_cls in (PennTreebankBertPretraining, WikiText2BertPretraining):
            with self.subTest(dataset_cls=dataset_cls.__name__):
                data = dataset_cls(batch_size=2, sequence_length=8, num_workers=0)

                self.assertEqual(data.batch_size, 2)
                self.assertEqual(data.sequence_length, 8)
                self.assertIsNotNone(data.torchtext_dataset)

    def test_legacy_source_preserves_text_units(self):
        for dataset_cls in (
            OfflinePennTreebankBertPretraining,
            OfflineWikiText2BertPretraining,
        ):
            with self.subTest(dataset_cls=dataset_cls.__name__):
                data = dataset_cls(num_workers=0)

                text_units = tuple(data._dataset("train"))

                self.assertEqual(text_units, OFFLINE_TEXT_SPLITS["train"])

    def test_catalogued_adapters_produce_deterministic_offline_batch(self):
        for dataset_cls in (
            OfflinePennTreebankBertPretraining,
            OfflineWikiText2BertPretraining,
        ):
            with self.subTest(dataset_cls=dataset_cls.__name__):
                batches = []
                for _ in range(2):
                    data = dataset_cls(
                        batch_size=2,
                        sequence_length=10,
                        target_vocab_size=32,
                        num_workers=0,
                        drop_last=False,
                        seed=11,
                    )
                    data.setup("fit")
                    batches.append(next(iter(data.train_dataloader())))

                for first, second in zip(*batches, strict=True):
                    torch.testing.assert_close(first, second)
                    self.assertEqual(first.dtype, torch.long)

                (
                    input_ids,
                    mlm_labels,
                    attention_mask,
                    token_type_ids,
                    next_sentence_labels,
                ) = batches[0]
                self.assertEqual(input_ids.shape, torch.Size([2, 10]))
                self.assertEqual(mlm_labels.shape, torch.Size([2, 10]))
                self.assertEqual(attention_mask.shape, torch.Size([2, 10]))
                self.assertEqual(token_type_ids.shape, torch.Size([2, 10]))
                self.assertEqual(next_sentence_labels.shape, torch.Size([2]))


if __name__ == "__main__":
    unittest.main()

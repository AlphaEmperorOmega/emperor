import torch
import unittest

from emperor.datasets.text.masked_language_modeling import (
    MaskedLanguageModelingCollator,
    PennTreebankMaskedLanguageModeling,
    WikiText2MaskedLanguageModeling,
    WikiText103MaskedLanguageModeling,
    build_mlm_token_windows,
)
from emperor.datasets.text.vocabulary import (
    BERT_SPECIAL_TOKENS,
    BertSpecialTokenIds,
    get_bert_special_token_ids,
    set_bert_default_index,
)


class FakeVocab:
    def __init__(self, token_to_index: dict[str, int]):
        self.token_to_index = token_to_index
        self.default_index = None

    def get_stoi(self):
        return self.token_to_index

    def set_default_index(self, index: int) -> None:
        self.default_index = index


class TestBertVocabularyHelpers(unittest.TestCase):
    def preset(self) -> BertSpecialTokenIds:
        return BertSpecialTokenIds(pad=0, unk=1, cls=2, sep=3, mask=4)

    def test_get_bert_special_token_ids_reads_vocab_mapping(self):
        vocab = FakeVocab(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "hello": 5,
            }
        )

        token_ids = get_bert_special_token_ids(vocab)

        self.assertEqual(token_ids, self.preset())

    def test_set_bert_default_index_uses_unk_token(self):
        vocab = FakeVocab({token: index for index, token in enumerate(BERT_SPECIAL_TOKENS)})

        token_ids = set_bert_default_index(vocab)

        self.assertEqual(token_ids.unk, 1)
        self.assertEqual(vocab.default_index, 1)

    def test_missing_bert_special_token_raises_key_error(self):
        vocab = FakeVocab({"[PAD]": 0, "[UNK]": 1})

        with self.assertRaises(KeyError):
            get_bert_special_token_ids(vocab)


class TestMaskedLanguageModelingCollator(unittest.TestCase):
    def preset(self) -> BertSpecialTokenIds:
        return BertSpecialTokenIds(pad=0, unk=1, cls=2, sep=3, mask=4)

    def collator(
        self,
        mlm_probability: float = 1.0,
        mask_replace_probability: float = 1.0,
        random_replace_probability: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> MaskedLanguageModelingCollator:
        return MaskedLanguageModelingCollator(
            special_token_ids=self.preset(),
            vocab_size=12,
            mlm_probability=mlm_probability,
            mask_replace_probability=mask_replace_probability,
            random_replace_probability=random_replace_probability,
            generator=generator,
        )

    def test_special_tokens_are_excluded_from_masking(self):
        token_ids = self.preset()
        collator = self.collator()
        tokens = torch.tensor(
            [[token_ids.cls, 5, 6, token_ids.sep, token_ids.pad, token_ids.unk, token_ids.mask, 7]]
        )

        input_ids, labels, attention_mask = collator(tokens)

        torch.testing.assert_close(
            labels,
            torch.tensor(
                [[-100, 5, 6, -100, -100, -100, -100, 7]]
            ),
        )
        torch.testing.assert_close(
            input_ids,
            torch.tensor(
                [[token_ids.cls, token_ids.mask, token_ids.mask, token_ids.sep, token_ids.pad, token_ids.unk, token_ids.mask, token_ids.mask]]
            ),
        )
        torch.testing.assert_close(
            attention_mask,
            torch.tensor([[1, 1, 1, 1, 0, 1, 1, 1]]),
        )

    def test_unmasked_labels_are_ignore_index(self):
        collator = self.collator(mlm_probability=1.0)
        tokens = torch.tensor([[5, 6, 7]])

        _, labels, _ = collator(tokens)

        self.assertTrue(torch.all(labels != -100))

        collator = self.collator(mlm_probability=0.0)
        _, labels, _ = collator(tokens)

        self.assertEqual((labels != -100).sum().item(), 1)
        self.assertEqual((labels == -100).sum().item(), 2)

    def test_labels_preserve_original_token_ids_at_masked_positions(self):
        collator = self.collator()
        tokens = torch.tensor([[5, 6, 7, 8]])

        input_ids, labels, _ = collator(tokens)

        torch.testing.assert_close(labels, tokens)
        self.assertTrue(torch.all(input_ids == self.preset().mask))

    def test_mask_replacement_is_deterministic_with_seed(self):
        tokens = torch.tensor([[5, 6, 7, 8, 9, 10]])
        first = self.collator(
            mlm_probability=0.5,
            generator=torch.Generator().manual_seed(7),
        )
        second = self.collator(
            mlm_probability=0.5,
            generator=torch.Generator().manual_seed(7),
        )

        first_input_ids, first_labels, first_attention_mask = first(tokens)
        second_input_ids, second_labels, second_attention_mask = second(tokens)

        torch.testing.assert_close(first_input_ids, second_input_ids)
        torch.testing.assert_close(first_labels, second_labels)
        torch.testing.assert_close(first_attention_mask, second_attention_mask)
        self.assertTrue(torch.any(first_labels != -100))
        self.assertTrue(torch.all(first_input_ids[first_labels != -100] == self.preset().mask))

    def test_random_replacement_avoids_special_tokens(self):
        collator = self.collator(
            mlm_probability=1.0,
            mask_replace_probability=0.0,
            random_replace_probability=1.0,
            generator=torch.Generator().manual_seed(11),
        )
        tokens = torch.tensor([[5, 6, 7, 8]])

        input_ids, labels, _ = collator(tokens)

        self.assertTrue(torch.all(labels == tokens))
        self.assertTrue(torch.all(input_ids >= 5))

    def test_attention_mask_matches_pad_positions(self):
        token_ids = self.preset()
        collator = self.collator()
        tokens = torch.tensor([[5, token_ids.pad, 6, token_ids.pad]])

        _, _, attention_mask = collator(tokens)

        torch.testing.assert_close(attention_mask, torch.tensor([[1, 0, 1, 0]]))

    def test_at_least_one_token_is_masked_when_possible(self):
        collator = self.collator(mlm_probability=0.0)
        tokens = torch.tensor([[5, 6, 7]])

        input_ids, labels, _ = collator(tokens)

        masked_positions = labels != -100
        self.assertEqual(masked_positions.sum().item(), 1)
        self.assertTrue(torch.all(input_ids[masked_positions] == self.preset().mask))

    def test_special_only_batch_has_no_masked_labels(self):
        token_ids = self.preset()
        collator = self.collator(mlm_probability=1.0)
        tokens = torch.tensor(
            [[token_ids.pad, token_ids.unk, token_ids.cls, token_ids.sep, token_ids.mask]]
        )

        _, labels, _ = collator(tokens)

        self.assertTrue(torch.all(labels == -100))

    def test_rejects_invalid_probability_configuration(self):
        with self.assertRaises(ValueError):
            self.collator(mask_replace_probability=0.8, random_replace_probability=0.3)


class TestMaskedLanguageModelingDatasetPath(unittest.TestCase):
    def preset(self) -> BertSpecialTokenIds:
        return BertSpecialTokenIds(pad=0, unk=1, cls=2, sep=3, mask=4)

    def test_build_mlm_token_windows_preserves_unshifted_tokens(self):
        windows = build_mlm_token_windows(
            [5, 6, 7, 8, 9],
            sequence_length=6,
            special_token_ids=self.preset(),
        )

        torch.testing.assert_close(
            windows,
            torch.tensor(
                [
                    [2, 5, 6, 7, 8, 3],
                    [2, 9, 3, 0, 0, 0],
                ]
            ),
        )

    def test_build_mlm_token_windows_can_skip_special_insertion(self):
        windows = build_mlm_token_windows(
            [5, 6, 7, 8, 9],
            sequence_length=3,
            special_token_ids=self.preset(),
            add_special_tokens=False,
        )

        torch.testing.assert_close(
            windows,
            torch.tensor(
                [
                    [5, 6, 7],
                    [8, 9, 0],
                ]
            ),
        )

    def test_dataset_classes_provide_mlm_path_without_changing_legacy_datasets(self):
        for dataset_cls in (
            PennTreebankMaskedLanguageModeling,
            WikiText2MaskedLanguageModeling,
            WikiText103MaskedLanguageModeling,
        ):
            with self.subTest(dataset_cls=dataset_cls.__name__):
                data = dataset_cls(batch_size=2, sequence_length=8, num_workers=0)

                self.assertEqual(data.batch_size, 2)
                self.assertEqual(data.sequence_length, 8)
                self.assertIsNotNone(data.torchtext_dataset)


if __name__ == "__main__":
    unittest.main()

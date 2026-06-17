import random
import unittest

import torch

from emperor.datasets.text.bert_pretraining import (
    BertPretrainingCollator,
    PennTreebankBertPretraining,
    WikiText2BertPretraining,
    build_bert_next_sentence_pairs,
    build_bert_pretraining_examples,
    build_bert_sentence_pair_inputs,
    train_local_wordpiece_tokenizer,
)
from emperor.datasets.text.vocabulary import (
    BertSpecialTokenIds,
    get_bert_special_token_ids,
)


class TestBertPretrainingDatasetHelpers(unittest.TestCase):
    def preset(self) -> BertSpecialTokenIds:
        return BertSpecialTokenIds(pad=0, unk=1, cls=2, sep=3, mask=4)

    def test_local_wordpiece_tokenizer_uses_fixed_bert_special_ids(self):
        tokenizer = train_local_wordpiece_tokenizer(
            ["Hello world", "hello there world"],
            vocab_size=24,
        )

        token_ids = get_bert_special_token_ids(tokenizer)

        self.assertEqual(token_ids, self.preset())
        self.assertEqual(tokenizer.token_to_id("[PAD]"), 0)
        self.assertEqual(tokenizer.token_to_id("[MASK]"), 4)

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

    def test_dataset_classes_expose_pretraining_metadata(self):
        for dataset_cls in (PennTreebankBertPretraining, WikiText2BertPretraining):
            with self.subTest(dataset_cls=dataset_cls.__name__):
                data = dataset_cls(batch_size=2, sequence_length=8, num_workers=0)

                self.assertEqual(data.batch_size, 2)
                self.assertEqual(data.sequence_length, 8)
                self.assertIsNotNone(data.torchtext_dataset)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import call, patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchtext.data.utils import get_tokenizer

from emperor.datasets.text.question_answering._adapter import (
    _QuestionAnswerEncoder,
    _QuestionAnsweringAdapter,
    _SQuADv1Dataset,
    _SQuADv2Dataset,
)
from emperor.datasets.text.question_answering._answer_spans import _AnswerSpanAligner
from emperor.datasets.text.question_answering._squad_v1 import SQuADv1
from emperor.datasets.text.question_answering._squad_v2 import SQuADv2


class _Vocabulary:
    def __init__(self, token_sequences) -> None:
        tokens = {token for sequence in token_sequences for token in sequence}
        self._tokens = ["<unk>", "<pad>", *sorted(tokens)]
        self._indices = {token: index for index, token in enumerate(self._tokens)}

    def __call__(self, tokens: list[str]) -> list[int]:
        unknown_index = self._indices["<unk>"]
        return [self._indices.get(token, unknown_index) for token in tokens]

    def __getitem__(self, token: str) -> int:
        return self._indices[token]

    def lookup_token(self, index: int) -> str:
        return self._tokens[index]

    def __len__(self) -> int:
        return len(self._tokens)

    def set_default_index(self, index: int) -> None:
        self.default_index = index


def _build_vocabulary(token_sequences, *, specials):
    del specials
    return _Vocabulary(token_sequences)


class SQuADDatasetTestCase(unittest.TestCase):
    dataset_type = _SQuADv1Dataset

    def setUp(self) -> None:
        self.tokenizer = get_tokenizer("basic_english")

    def sample(
        self,
        context: str,
        *,
        answer_starts: list[int],
        answer_texts: list[str],
        question: str = "Where is the answer?",
    ) -> dict[str, object]:
        return {
            "context": context,
            "question": question,
            "answers": {
                "answer_start": answer_starts,
                "text": answer_texts,
            },
        }

    def dataset(
        self,
        samples: list[dict[str, object]],
        *,
        context_length: int = 16,
        question_length: int = 8,
    ):
        token_sequences = (
            self.tokenizer(text)
            for sample in samples
            for text in (str(sample["context"]), str(sample["question"]))
        )
        vocab = _Vocabulary(token_sequences)
        return self.dataset_type(
            samples,
            _QuestionAnswerEncoder(
                self.tokenizer,
                vocab,
                context_length,
                question_length,
            ),
            _AnswerSpanAligner(self.tokenizer, context_length),
        )

    def assert_span(
        self,
        dataset,
        *,
        expected_start: int,
        expected_end: int,
        expected_tokens: list[str],
    ) -> None:
        context, _, start, end = dataset[0]

        self.assertEqual(start.item(), expected_start)
        self.assertEqual(end.item(), expected_end)
        actual_tokens = [
            dataset.vocab.lookup_token(int(token_id))
            for token_id in context[expected_start : expected_end + 1]
        ]
        self.assertEqual(actual_tokens, expected_tokens)


class TestSQuADv1Dataset(SQuADDatasetTestCase):
    def test_character_offset_anchors_repeated_answer_to_exact_token(self) -> None:
        context = "Alpha, beta! Alpha is final."
        dataset = self.dataset(
            [
                self.sample(
                    context,
                    answer_starts=[context.rindex("Alpha")],
                    answer_texts=["Alpha"],
                )
            ]
        )

        self.assert_span(
            dataset,
            expected_start=4,
            expected_end=4,
            expected_tokens=["alpha"],
        )

    def test_punctuation_and_whitespace_map_multi_token_answer(self) -> None:
        context = "  Prefix: New York, then suffix."
        dataset = self.dataset(
            [
                self.sample(
                    context,
                    answer_starts=[context.index("New")],
                    answer_texts=["New York"],
                )
            ]
        )

        self.assert_span(
            dataset,
            expected_start=1,
            expected_end=2,
            expected_tokens=["new", "york"],
        )

    def test_first_alignable_source_answer_is_selected(self) -> None:
        context = "alpha beta gamma"
        dataset = self.dataset(
            [
                self.sample(
                    context,
                    answer_starts=[999, context.index("beta")],
                    answer_texts=["missing", "beta"],
                )
            ]
        )

        self.assert_span(
            dataset,
            expected_start=1,
            expected_end=1,
            expected_tokens=["beta"],
        )

    def test_examples_without_an_in_window_answer_are_excluded(self) -> None:
        truncated_context = "zero one two three four"
        valid_context = "alpha beta"
        dataset = self.dataset(
            [
                self.sample(
                    truncated_context,
                    answer_starts=[truncated_context.index("four")],
                    answer_texts=["four"],
                ),
                self.sample(
                    valid_context,
                    answer_starts=[valid_context.index("alpha")],
                    answer_texts=["alpha"],
                ),
            ],
            context_length=4,
        )

        self.assertEqual(len(dataset), 1)
        self.assert_span(
            dataset,
            expected_start=0,
            expected_end=0,
            expected_tokens=["alpha"],
        )

    def test_tensor_schema_and_boundary_spans_remain_stable(self) -> None:
        context = "first middle last"
        dataset = self.dataset(
            [
                self.sample(
                    context,
                    answer_starts=[context.index("last")],
                    answer_texts=["last"],
                    question="Which word?",
                )
            ],
            context_length=3,
            question_length=5,
        )

        context_ids, question_ids, start, end = dataset[0]

        self.assertEqual(context_ids.shape, torch.Size([3]))
        self.assertEqual(question_ids.shape, torch.Size([5]))
        self.assertEqual(start.shape, torch.Size([]))
        self.assertEqual(end.shape, torch.Size([]))
        self.assertEqual(context_ids.dtype, torch.long)
        self.assertEqual(question_ids.dtype, torch.long)
        self.assertEqual(start.dtype, torch.long)
        self.assertEqual(end.dtype, torch.long)
        self.assertEqual((start.item(), end.item()), (2, 2))


class TestSQuADv2Dataset(SQuADDatasetTestCase):
    dataset_type = _SQuADv2Dataset

    def test_answerable_example_uses_character_anchored_token_span(self) -> None:
        context = "Before: answer, after."
        dataset = self.dataset(
            [
                self.sample(
                    context,
                    answer_starts=[context.index("answer")],
                    answer_texts=["answer"],
                )
            ]
        )

        self.assert_span(
            dataset,
            expected_start=1,
            expected_end=1,
            expected_tokens=["answer"],
        )

    def test_unanswerable_example_remains_negative_one_span(self) -> None:
        dataset = self.dataset(
            [self.sample("No answer here.", answer_starts=[], answer_texts=[])]
        )

        _, _, start, end = dataset[0]

        self.assertEqual((start.item(), end.item()), (-1, -1))

    def test_answer_outside_truncated_context_is_negative_one_span(self) -> None:
        context = "zero one two three four"
        dataset = self.dataset(
            [
                self.sample(
                    context,
                    answer_starts=[context.index("four")],
                    answer_texts=["four"],
                )
            ],
            context_length=4,
        )

        _, _, start, end = dataset[0]

        self.assertEqual((start.item(), end.item()), (-1, -1))


class TestQuestionAnsweringAdapter(unittest.TestCase):
    cases = (
        (SQuADv1, "squad", _SQuADv1Dataset),
        (SQuADv2, "squad_v2", _SQuADv2Dataset),
    )

    def sample(self, context: str, answer: str) -> dict[str, object]:
        return {
            "context": context,
            "question": "Which token?",
            "answers": {
                "answer_start": [context.index(answer)],
                "text": [answer],
            },
        }

    def test_leaves_declare_only_source_and_answer_policy_variation(self) -> None:
        for dataset_type, source_name, item_dataset_type in self.cases:
            with self.subTest(dataset=dataset_type.__name__):
                self.assertTrue(issubclass(dataset_type, _QuestionAnsweringAdapter))
                self.assertEqual(dataset_type._source_name, source_name)
                self.assertIs(dataset_type._item_dataset_type, item_dataset_type)

    def test_prepare_and_fit_preserve_source_schema_and_loader_policy(self) -> None:
        train_samples = [self.sample("train answer", "answer")]
        validation_samples = [self.sample("validation answer", "answer")]

        for dataset_type, source_name, item_dataset_type in self.cases:
            with self.subTest(dataset=dataset_type.__name__):
                def source(
                    name: str,
                    *,
                    split: str,
                    expected_source_name: str = source_name,
                ):
                    self.assertEqual(name, expected_source_name)
                    return train_samples if split == "train" else validation_samples

                dataset = dataset_type(
                    batch_size=1,
                    context_length=4,
                    question_length=4,
                )
                with (
                    patch(
                        "emperor.datasets.text.question_answering._adapter.load_dataset",
                        side_effect=source,
                    ) as load_dataset,
                    patch(
                        "emperor.datasets.text.question_answering._adapter."
                        "build_vocab_from_iterator",
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset.prepare_data()
                    dataset.setup("fit")

                self.assertEqual(
                    load_dataset.call_args_list,
                    [
                        call(source_name, split="train"),
                        call(source_name, split="validation"),
                        call(source_name, split="train"),
                        call(source_name, split="validation"),
                    ],
                )
                self.assertIsInstance(dataset.train, item_dataset_type)
                self.assertIsInstance(dataset.val, item_dataset_type)
                self.assertEqual(
                    dataset.resolved_metadata.vocab_size,
                    len(dataset.vocab),
                )
                self.assertEqual(dataset.resolved_metadata.num_classes, 2)
                validation_context, _, _, _ = dataset.val[0]
                self.assertEqual(int(validation_context[0]), 0)
                self.assertIsInstance(dataset.train_dataloader().sampler, RandomSampler)
                self.assertIsInstance(dataset.val_dataloader().sampler, SequentialSampler)
                self.assertTrue(dataset.train_dataloader().drop_last)
                self.assertTrue(dataset.val_dataloader().drop_last)

    def test_fresh_validation_preserves_validation_owned_vocabulary(self) -> None:
        validation_samples = [self.sample("validation answer", "answer")]

        for dataset_type, source_name, _ in self.cases:
            with self.subTest(dataset=dataset_type.__name__):
                dataset = dataset_type(context_length=4, question_length=4)
                with (
                    patch(
                        "emperor.datasets.text.question_answering._adapter.load_dataset",
                        return_value=validation_samples,
                    ) as load_dataset,
                    patch(
                        "emperor.datasets.text.question_answering._adapter."
                        "build_vocab_from_iterator",
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset.setup("validate")

                load_dataset.assert_called_once_with(
                    source_name,
                    split="validation",
                )
                self.assertEqual(dataset._text_labels([0, 1]), ["<unk>", "<pad>"])

if __name__ == "__main__":
    unittest.main()

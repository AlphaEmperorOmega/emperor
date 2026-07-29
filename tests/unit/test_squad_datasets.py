import unittest

import torch
from torchtext.data.utils import get_tokenizer

from emperor.datasets.text.question_answering._squad_v1 import (
    _QADataset as SQuADv1Dataset,
)
from emperor.datasets.text.question_answering._squad_v2 import (
    _QADataset as SQuADv2Dataset,
)


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


class SQuADDatasetTestCase(unittest.TestCase):
    dataset_type = SQuADv1Dataset

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
            self.tokenizer,
            vocab,
            context_length,
            question_length,
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
    dataset_type = SQuADv2Dataset

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


if __name__ == "__main__":
    unittest.main()

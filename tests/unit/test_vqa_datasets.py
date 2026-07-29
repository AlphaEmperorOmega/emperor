from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.multimodal.vqa._gqa import GQA


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


class GQADatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_root = self.root / "images"
        self.image_root.mkdir()
        for image_id, color in (("1", (255, 0, 0)), ("2", (0, 255, 0))):
            Image.new("RGB", (10, 12), color=color).save(
                self.image_root / f"{image_id}.jpg"
            )

        self.train_questions = {
            "train-red-1": {
                "imageId": "1",
                "question": "What color is shown?",
                "answer": "red",
            },
            "train-red-2": {
                "imageId": "2",
                "question": "Which color appears?",
                "answer": "red",
            },
            "train-circle": {
                "imageId": "1",
                "question": "Which shape appears?",
                "answer": "circle",
            },
        }
        self.validation_questions = {
            "validation-red": {
                "imageId": "1",
                "question": "What color appears?",
                "answer": "red",
            },
            "validation-unknown": {
                "imageId": "2",
                "question": "What is missing?",
                "answer": "blue",
            },
        }
        self.train_path = self._write_questions("train.json", self.train_questions)
        self.validation_path = self._write_questions(
            "validation.json",
            self.validation_questions,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_questions(self, name: str, questions: dict) -> str:
        path = self.root / name
        path.write_text(json.dumps(questions), encoding="utf-8")
        return str(path)

    def _dataset(self, *, answer_classes: int = 1) -> GQA:
        dataset = GQA(
            batch_size=1,
            question_length=5,
            resize=(6, 8),
            train_questions_file=self.train_path,
            val_questions_file=self.validation_path,
            image_root=str(self.image_root),
            num_answer_classes=answer_classes,
        )
        dataset.num_workers = 0
        return dataset

    def test_fit_resolves_train_vocabulary_and_filters_unknown_answers(self) -> None:
        dataset = self._dataset()
        with patch(
            "emperor.datasets.multimodal.vqa._gqa.build_vocab_from_iterator",
            side_effect=_build_vocabulary,
        ):
            dataset.prepare_data()
            dataset.setup("fit")

        self.assertEqual(dataset.answer_vocab, {"red": 0})
        self.assertEqual(len(dataset.train), 2)
        self.assertEqual(len(dataset.val), 1)
        self.assertEqual(dataset.resolved_metadata.vocab_size, len(dataset.question_vocab))
        self.assertEqual(dataset.resolved_metadata.num_classes, 1)
        self.assertIsInstance(dataset.train_dataloader().sampler, RandomSampler)
        self.assertIsInstance(dataset.val_dataloader().sampler, SequentialSampler)

        images, questions, answers = next(iter(dataset.val_dataloader()))
        self.assertEqual(images.shape, torch.Size([1, 3, 6, 8]))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(questions.shape, torch.Size([1, 5]))
        self.assertEqual(questions.dtype, torch.long)
        self.assertEqual(answers.shape, torch.Size([1]))
        self.assertEqual(answers.dtype, torch.long)
        self.assertEqual(dataset._text_labels(answers), ["red"])

    def test_validation_only_owns_its_schema_and_missing_files_fail_clearly(self) -> None:
        dataset = self._dataset(answer_classes=2)
        with patch(
            "emperor.datasets.multimodal.vqa._gqa.build_vocab_from_iterator",
            side_effect=_build_vocabulary,
        ):
            dataset.setup("validate")

        self.assertEqual(dataset.answer_vocab, {"red": 0, "blue": 1})
        self.assertEqual(len(dataset.val), 2)
        self.assertEqual(dataset._text_labels([0, 1, 99]), ["red", "blue", "<unk>"])

        missing = GQA(val_questions_file=str(self.root / "missing.json"))
        with self.assertRaises(FileNotFoundError):
            missing.setup("validate")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from emperor.datasets.multimodal.vqa._vqa_v2 import VQAv2


class _Vocabulary:
    def __init__(self) -> None:
        self._indices = {
            "<unk>": 0,
            "<pad>": 1,
            "is": 2,
            "visible": 3,
            "?": 4,
        }

    def __call__(self, tokens: list[str]) -> list[int]:
        return [self._indices.get(token, self._indices["<unk>"]) for token in tokens]

    def __getitem__(self, token: str) -> int:
        return self._indices[token]


class _OfflineVQAv2(VQAv2):
    def _build_vocabs(self, questions: dict, annotations: dict) -> None:
        if self.question_vocab is not None:
            return
        self.question_vocab = _Vocabulary()
        self.answer_vocab = {"yes": 0}


class TestVQAv2Dataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.train_image_root = self.root / "images" / "train"
        self.val_image_root = self.root / "images" / "validation"
        self.train_image_root.mkdir(parents=True)
        self.val_image_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_json(self, name: str, payload: dict) -> str:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def write_image(self, root: Path, name: str, color: tuple[int, int, int]) -> None:
        Image.new("RGB", (3, 2), color=color).save(root / name)

    def data_module(
        self,
        *,
        train_questions: list[dict] | None = None,
        train_annotations: list[dict] | None = None,
        val_questions: list[dict] | None = None,
        val_annotations: list[dict] | None = None,
    ) -> _OfflineVQAv2:
        train_questions = train_questions or [
            {"question_id": 1, "image_id": 1, "question": "Is visible?"}
        ]
        train_annotations = train_annotations or [
            {"question_id": 1, "multiple_choice_answer": "yes"}
        ]
        val_questions = val_questions or [
            {"question_id": 2, "image_id": 2, "question": "Is visible?"}
        ]
        val_annotations = val_annotations or [
            {"question_id": 2, "multiple_choice_answer": "yes"}
        ]
        return _OfflineVQAv2(
            batch_size=1,
            question_length=6,
            resize=(4, 6),
            train_questions_file=self.write_json(
                "train_questions.json",
                {"questions": train_questions},
            ),
            train_annotations_file=self.write_json(
                "train_annotations.json",
                {"annotations": train_annotations},
            ),
            val_questions_file=self.write_json(
                "val_questions.json",
                {"questions": val_questions},
            ),
            val_annotations_file=self.write_json(
                "val_annotations.json",
                {"annotations": val_annotations},
            ),
            train_image_root=str(self.train_image_root),
            val_image_root=str(self.val_image_root),
        )

    def test_fit_uses_explicit_train_and_validation_image_prefixes(self) -> None:
        train_name = "COCO_train2014_000000000001.jpg"
        val_name = "COCO_val2014_000000000002.jpg"
        self.write_image(self.train_image_root, train_name, (255, 0, 0))
        self.write_image(self.val_image_root, val_name, (0, 255, 0))
        data = self.data_module()

        data._setup_fit()

        self.assertEqual(data.train.samples[0][0], train_name)
        self.assertEqual(data.val.samples[0][0], val_name)
        train_image, train_question, train_answer = data.train[0]
        val_image, val_question, val_answer = data.val[0]
        self.assertEqual(train_image.shape, torch.Size([3, 4, 6]))
        self.assertEqual(val_image.shape, torch.Size([3, 4, 6]))
        self.assertEqual(train_question.dtype, torch.long)
        self.assertEqual(val_question.dtype, torch.long)
        self.assertEqual(train_answer.dtype, torch.long)
        self.assertEqual(val_answer.dtype, torch.long)
        torch.testing.assert_close(train_question, val_question)
        self.assertEqual(train_answer.item(), 0)
        self.assertEqual(val_answer.item(), 0)

    def test_standalone_validation_uses_validation_image_prefix(self) -> None:
        val_name = "COCO_val2014_000000000002.jpg"
        self.write_image(self.val_image_root, val_name, (0, 0, 255))
        data = self.data_module()

        data._setup_validate()

        self.assertEqual(data.val.samples[0][0], val_name)
        image, question, answer = data.val[0]
        self.assertEqual(image.shape, torch.Size([3, 4, 6]))
        self.assertEqual(question.shape, torch.Size([6]))
        self.assertEqual(answer.item(), 0)

    def test_sample_names_pad_ids_and_preserve_filtering(self) -> None:
        train_questions = [
            {"question_id": 10, "image_id": 0, "question": "Is visible?"},
            {
                "question_id": 11,
                "image_id": 123456789012,
                "question": "Is visible?",
            },
            {"question_id": 12, "image_id": 12, "question": "Is visible?"},
            {"question_id": 13, "image_id": 13, "question": "Is visible?"},
        ]
        train_annotations = [
            {"question_id": 10, "multiple_choice_answer": "yes"},
            {"question_id": 11, "multiple_choice_answer": "yes"},
            {"question_id": 12, "multiple_choice_answer": "no"},
        ]
        data = self.data_module(
            train_questions=train_questions,
            train_annotations=train_annotations,
        )

        data._setup_fit()

        self.assertEqual(
            [sample[0] for sample in data.train.samples],
            [
                "COCO_train2014_000000000000.jpg",
                "COCO_train2014_123456789012.jpg",
            ],
        )


if __name__ == "__main__":
    unittest.main()

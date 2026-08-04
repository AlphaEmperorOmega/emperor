import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.multimodal.vqa._vqa_v2 import VQAv2, _VQADataset


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

    def test_dataset_length_and_second_sample_are_literal(self) -> None:
        Image.new("RGB", (3, 2), color=(1, 2, 3)).save(
            self.train_image_root / "first.jpg",
            format="PNG",
        )
        Image.new("RGB", (3, 2), color=(4, 5, 6)).save(
            self.train_image_root / "second.jpg",
            format="PNG",
        )
        first_tokens = torch.tensor([1, 2])
        second_tokens = torch.tensor([3, 4])
        first_answer = torch.tensor(0)
        second_answer = torch.tensor(1)
        dataset = _VQADataset(
            [
                ("first.jpg", first_tokens, first_answer),
                ("second.jpg", second_tokens, second_answer),
            ],
            self.train_image_root,
            lambda image: torch.tensor(image.getpixel((0, 0))),
        )

        image, tokens, answer = dataset[1]

        self.assertEqual(len(dataset), 2)
        torch.testing.assert_close(image, torch.tensor([4, 5, 6]))
        self.assertIs(tokens, second_tokens)
        self.assertIs(answer, second_answer)

    def test_json_vocab_guard_loaders_and_labels_follow_owned_contracts(self) -> None:
        direct_mapping = {"literal": {"question": "already indexed"}}
        mapping_path = self.write_json("direct_mapping.json", direct_mapping)
        data = VQAv2(batch_size=2)
        data.num_workers = 0
        question_vocab = {"<unk>": 0, "train": 1}
        answer_vocab = {"yes": 0, "no": 1}
        data.question_vocab = question_vocab
        data.answer_vocab = answer_vocab
        metadata = data.resolved_metadata
        training = torch.utils.data.TensorDataset(torch.tensor([0, 1, 2]))
        validation = torch.utils.data.TensorDataset(torch.tensor([3, 4, 5]))
        data.train = training
        data.val = validation

        data.prepare_data()
        loaded_mapping = data._load_json(mapping_path)
        data._build_vocabs(object(), object())
        training_loader = data.get_dataloader(train=True)
        validation_loader = data.get_dataloader(train=False)

        self.assertEqual(loaded_mapping, direct_mapping)
        self.assertIs(data.question_vocab, question_vocab)
        self.assertIs(data.answer_vocab, answer_vocab)
        self.assertIs(data.resolved_metadata, metadata)
        self.assertIsInstance(training_loader.sampler, RandomSampler)
        self.assertIsInstance(validation_loader.sampler, SequentialSampler)
        self.assertEqual(training_loader.batch_size, 2)
        self.assertEqual(validation_loader.batch_size, 2)
        self.assertTrue(training_loader.drop_last)
        self.assertTrue(validation_loader.drop_last)
        self.assertEqual(data._text_labels([0, 1, 99]), ["yes", "no", "<unk>"])

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

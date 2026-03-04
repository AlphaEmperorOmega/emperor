import json
import torch
import torch.utils.data
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision.transforms.transforms import Compose

from emperor.base.utils import DataModule


def _yield_question_tokens(questions, tokenizer):
    for question in questions.values():
        yield tokenizer(question["question"])


def _encode(text, tokenizer, vocab, sequence_length):
    tokens = vocab(tokenizer(text))[:sequence_length]
    padding = [vocab["<pad>"]] * (sequence_length - len(tokens))
    return tokens + padding


class _VQADataset(torch.utils.data.Dataset):
    def __init__(self, samples, image_root, transform):
        self.samples = samples
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, question_tokens, answer_label = self.samples[idx]
        image = Image.open(self.image_root / image_path).convert("RGB")
        image = self.transform(image)
        return image, question_tokens, answer_label


class VQAv2(DataModule):
    default_width: int = 224
    default_height: int = 224
    num_channels: int = 3
    flattened_input_dim: int = default_width * default_height * num_channels
    num_classes: int = 3129  # standard top-3129 VQA answer classes
    question_length: int = 20
    vocab_size: int = 19901  # approximate VQA v2 question vocab size

    def __init__(
        self,
        batch_size: int = 64,
        question_length: int = 20,
        resize: tuple = (224, 224),
        train_questions_file: str = "data/vqa_v2/v2_Questions_Train_mscoco.json",
        train_annotations_file: str = "data/vqa_v2/v2_Annotations_Train_mscoco.json",
        val_questions_file: str = "data/vqa_v2/v2_Questions_Val_mscoco.json",
        val_annotations_file: str = "data/vqa_v2/v2_Annotations_Val_mscoco.json",
        train_image_root: str = "data/coco/train2014",
        val_image_root: str = "data/coco/val2014",
        num_answer_classes: int = 3129,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.question_length = question_length
        self.resize = resize
        self.train_questions_file = train_questions_file
        self.train_annotations_file = train_annotations_file
        self.val_questions_file = val_questions_file
        self.val_annotations_file = val_annotations_file
        self.train_image_root = train_image_root
        self.val_image_root = val_image_root
        self.num_answer_classes = num_answer_classes
        self.tokenizer = get_tokenizer("basic_english")
        self.question_vocab = None
        self.answer_vocab = None

    def prepare_data(self) -> None:
        pass  # VQA v2 requires manual download from vqa.org

    def _setup_fit(self) -> None:
        train_questions = self._load_json(self.train_questions_file)
        train_annotations = self._load_json(self.train_annotations_file)
        val_questions = self._load_json(self.val_questions_file)
        val_annotations = self._load_json(self.val_annotations_file)
        self._build_vocabs(train_questions, train_annotations)
        self.train = _VQADataset(
            self._build_samples(train_questions, train_annotations, self.train_image_root),
            self.train_image_root,
            self._get_transforms(),
        )
        self.val = _VQADataset(
            self._build_samples(val_questions, val_annotations, self.val_image_root),
            self.val_image_root,
            self._get_transforms(),
        )

    def _setup_validate(self) -> None:
        val_questions = self._load_json(self.val_questions_file)
        val_annotations = self._load_json(self.val_annotations_file)
        self._build_vocabs(val_questions, val_annotations)
        self.val = _VQADataset(
            self._build_samples(val_questions, val_annotations, self.val_image_root),
            self.val_image_root,
            self._get_transforms(),
        )

    def _load_json(self, path: str) -> dict:
        with open(path) as f:
            data = json.load(f)
        # index questions by question_id
        if "questions" in data:
            return {q["question_id"]: q for q in data["questions"]}
        if "annotations" in data:
            return {a["question_id"]: a for a in data["annotations"]}
        return data

    def _build_vocabs(self, questions: dict, annotations: dict) -> None:
        if self.question_vocab is not None:
            return
        # question vocab
        self.question_vocab = build_vocab_from_iterator(
            _yield_question_tokens(questions, self.tokenizer),
            specials=["<unk>", "<pad>"],
        )
        self.question_vocab.set_default_index(self.question_vocab["<unk>"])
        VQAv2.vocab_size = len(self.question_vocab)
        # answer vocab — top-N most common answers
        answer_counts = Counter(
            a["multiple_choice_answer"] for a in annotations.values()
        )
        top_answers = [ans for ans, _ in answer_counts.most_common(self.num_answer_classes)]
        self.answer_vocab = {ans: idx for idx, ans in enumerate(top_answers)}
        VQAv2.num_classes = len(self.answer_vocab)

    def _build_samples(self, questions: dict, annotations: dict, image_root: str) -> list:
        samples = []
        for qid, question in questions.items():
            if qid not in annotations:
                continue
            answer = annotations[qid]["multiple_choice_answer"]
            if answer not in self.answer_vocab:
                continue
            image_id = question["image_id"]
            image_path = f"COCO_train2014_{image_id:012d}.jpg"
            question_tokens = torch.tensor(
                _encode(question["question"], self.tokenizer, self.question_vocab, self.question_length),
                dtype=torch.long,
            )
            answer_label = torch.tensor(self.answer_vocab[answer], dtype=torch.long)
            samples.append((image_path, question_tokens, answer_label))
        return samples

    def _get_transforms(self) -> Compose:
        return transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def _text_labels(self, indices) -> list:
        inv_vocab = {v: k for k, v in self.answer_vocab.items()}
        return [inv_vocab.get(int(i), "<unk>") for i in indices]

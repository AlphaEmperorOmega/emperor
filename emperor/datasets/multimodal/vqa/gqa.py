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


class _GQADataset(torch.utils.data.Dataset):
    def __init__(self, samples, image_root, transform):
        self.samples = samples
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, question_tokens, answer_label = self.samples[idx]
        image = Image.open(self.image_root / f"{image_id}.jpg").convert("RGB")
        image = self.transform(image)
        return image, question_tokens, answer_label


class GQA(DataModule):
    default_width: int = 224
    default_height: int = 224
    num_channels: int = 3
    flattened_input_dim: int = default_width * default_height * num_channels
    num_classes: int = 1852  # approximate number of unique GQA answers
    question_length: int = 20
    vocab_size: int = 3097  # approximate GQA question vocab size

    def __init__(
        self,
        batch_size: int = 64,
        question_length: int = 20,
        resize: tuple = (224, 224),
        train_questions_file: str = "data/gqa/train_balanced_questions.json",
        val_questions_file: str = "data/gqa/val_balanced_questions.json",
        image_root: str = "data/gqa/images",
        num_answer_classes: int = 1852,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.question_length = question_length
        self.resize = resize
        self.train_questions_file = train_questions_file
        self.val_questions_file = val_questions_file
        self.image_root = image_root
        self.num_answer_classes = num_answer_classes
        self.tokenizer = get_tokenizer("basic_english")
        self.question_vocab = None
        self.answer_vocab = None

    def prepare_data(self) -> None:
        pass  # GQA requires manual download from cs.stanford.edu/people/dorarad/gqa

    def _setup_fit(self) -> None:
        train_questions = self._load_json(self.train_questions_file)
        val_questions = self._load_json(self.val_questions_file)
        self._build_vocabs(train_questions)
        self.train = _GQADataset(
            self._build_samples(train_questions),
            self.image_root,
            self._get_transforms(),
        )
        self.val = _GQADataset(
            self._build_samples(val_questions),
            self.image_root,
            self._get_transforms(),
        )

    def _setup_validate(self) -> None:
        val_questions = self._load_json(self.val_questions_file)
        self._build_vocabs(val_questions)
        self.val = _GQADataset(
            self._build_samples(val_questions),
            self.image_root,
            self._get_transforms(),
        )

    def _load_json(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def _build_vocabs(self, questions: dict) -> None:
        if self.question_vocab is not None:
            return
        # question vocab
        self.question_vocab = build_vocab_from_iterator(
            _yield_question_tokens(questions, self.tokenizer),
            specials=["<unk>", "<pad>"],
        )
        self.question_vocab.set_default_index(self.question_vocab["<unk>"])
        GQA.vocab_size = len(self.question_vocab)
        # answer vocab — top-N most common answers
        answer_counts = Counter(q["answer"] for q in questions.values())
        top_answers = [ans for ans, _ in answer_counts.most_common(self.num_answer_classes)]
        self.answer_vocab = {ans: idx for idx, ans in enumerate(top_answers)}
        GQA.num_classes = len(self.answer_vocab)

    def _build_samples(self, questions: dict) -> list:
        samples = []
        for question in questions.values():
            answer = question["answer"]
            if answer not in self.answer_vocab:
                continue
            question_tokens = torch.tensor(
                _encode(question["question"], self.tokenizer, self.question_vocab, self.question_length),
                dtype=torch.long,
            )
            answer_label = torch.tensor(self.answer_vocab[answer], dtype=torch.long)
            samples.append((question["imageId"], question_tokens, answer_label))
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

import torch
import torch.utils.data
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from emperor.datasets._base import DataModule
from emperor.datasets.text.question_answering._answer_spans import (
    _AnswerSpanAligner,
    _TokenSpan,
)


def _yield_tokens(samples, tokenizer):
    for sample in samples:
        yield tokenizer(sample["context"])
        yield tokenizer(sample["question"])


def _encode(text, tokenizer, vocab, sequence_length):
    tokens = vocab(tokenizer(text))[:sequence_length]
    padding = [vocab["<pad>"]] * (sequence_length - len(tokens))
    return tokens + padding


class _QADataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, vocab, context_length, question_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.context_length = context_length
        self.question_length = question_length
        aligner = _AnswerSpanAligner(tokenizer, context_length)
        self._sample_spans: list[tuple[int, _TokenSpan]] = []
        for sample_index, sample in enumerate(samples):
            span = aligner.first_in_window(
                sample["context"],
                sample["answers"]["answer_start"],
                sample["answers"]["text"],
            )
            if span is not None:
                self._sample_spans.append((sample_index, span))

    def __len__(self):
        return len(self._sample_spans)

    def __getitem__(self, idx):
        sample_index, span = self._sample_spans[idx]
        sample = self.samples[sample_index]
        context = torch.tensor(
            _encode(sample["context"], self.tokenizer, self.vocab, self.context_length),
            dtype=torch.long,
        )
        question = torch.tensor(
            _encode(
                sample["question"], self.tokenizer, self.vocab, self.question_length
            ),
            dtype=torch.long,
        )
        return (
            context,
            question,
            torch.tensor(span.start, dtype=torch.long),
            torch.tensor(span.end, dtype=torch.long),
        )


class SQuADv1(DataModule):
    vocab_size: int = 87599  # approximate SQuAD v1 vocab size
    num_classes: int = 2  # start + end span positions
    context_length: int = 384
    question_length: int = 64

    def __init__(
        self,
        batch_size: int = 32,
        context_length: int = 384,
        question_length: int = 64,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.context_length = context_length
        self.question_length = question_length
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None

    def prepare_data(self) -> None:
        load_dataset("squad", split="train")
        load_dataset("squad", split="validation")

    def _setup_fit(self) -> None:
        train_data = load_dataset("squad", split="train")
        val_data = load_dataset("squad", split="validation")
        self._build_vocab(train_data)
        self.train = _QADataset(
            train_data,
            self.tokenizer,
            self.vocab,
            self.context_length,
            self.question_length,
        )
        self.val = _QADataset(
            val_data,
            self.tokenizer,
            self.vocab,
            self.context_length,
            self.question_length,
        )

    def _setup_validate(self) -> None:
        val_data = load_dataset("squad", split="validation")
        self._build_vocab(val_data)
        self.val = _QADataset(
            val_data,
            self.tokenizer,
            self.vocab,
            self.context_length,
            self.question_length,
        )

    def _build_vocab(self, data) -> None:
        if self.vocab is not None:
            return
        self.vocab = build_vocab_from_iterator(
            _yield_tokens(data, self.tokenizer),
            specials=["<unk>", "<pad>"],
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        self._resolve_metadata(
            vocab_size=len(self.vocab),
            num_classes=self.num_classes,
        )

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
        return [self.vocab.lookup_token(int(i)) for i in indices]

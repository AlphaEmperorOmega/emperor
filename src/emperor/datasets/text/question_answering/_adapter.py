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


class _QuestionAnswerEncoder:
    def __init__(
        self,
        tokenizer,
        vocab,
        context_length: int,
        question_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.context_length = context_length
        self.question_length = question_length

    def encode(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._encode_text(sample["context"], self.context_length)
        question = self._encode_text(sample["question"], self.question_length)
        return context, question

    def _encode_text(self, text: str, sequence_length: int) -> torch.Tensor:
        tokens = self.vocab(self.tokenizer(text))[:sequence_length]
        padding = [self.vocab["<pad>"]] * (sequence_length - len(tokens))
        return torch.tensor(tokens + padding, dtype=torch.long)


class _SQuADv1Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples,
        encoder: _QuestionAnswerEncoder,
        answer_span_aligner: _AnswerSpanAligner,
    ) -> None:
        self.samples = samples
        self._encoder = encoder
        self.vocab = encoder.vocab
        self._sample_spans: list[tuple[int, _TokenSpan]] = []
        for sample_index, sample in enumerate(samples):
            span = answer_span_aligner.first_in_window(
                sample["context"],
                sample["answers"]["answer_start"],
                sample["answers"]["text"],
            )
            if span is not None:
                self._sample_spans.append((sample_index, span))

    def __len__(self) -> int:
        return len(self._sample_spans)

    def __getitem__(self, index: int):
        sample_index, span = self._sample_spans[index]
        context, question = self._encoder.encode(self.samples[sample_index])
        return (
            context,
            question,
            torch.tensor(span.start, dtype=torch.long),
            torch.tensor(span.end, dtype=torch.long),
        )


class _SQuADv2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples,
        encoder: _QuestionAnswerEncoder,
        answer_span_aligner: _AnswerSpanAligner,
    ) -> None:
        self.samples = samples
        self._encoder = encoder
        self.vocab = encoder.vocab
        self._answer_span_aligner = answer_span_aligner

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        context, question = self._encoder.encode(sample)
        span = self._answer_span_aligner.first_in_window(
            sample["context"],
            sample["answers"]["answer_start"],
            sample["answers"]["text"],
        )
        start, end = (-1, -1) if span is None else (span.start, span.end)
        return (
            context,
            question,
            torch.tensor(start, dtype=torch.long),
            torch.tensor(end, dtype=torch.long),
        )


class _QuestionAnsweringAdapter(DataModule):
    num_classes: int = 2
    context_length: int = 384
    question_length: int = 64
    _source_name: str
    _item_dataset_type: type[_SQuADv1Dataset] | type[_SQuADv2Dataset]

    def __init__(
        self,
        batch_size: int = 32,
        context_length: int = 384,
        question_length: int = 64,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.context_length = context_length
        self.question_length = question_length
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None

    def prepare_data(self) -> None:
        self._source("train")
        self._source("validation")

    def _setup_fit(self) -> None:
        train_data = self._source("train")
        validation_data = self._source("validation")
        self._build_vocab(train_data)
        self.train = self._build_dataset(train_data)
        self.val = self._build_dataset(validation_data)

    def _setup_validate(self) -> None:
        validation_data = self._source("validation")
        self._build_vocab(validation_data)
        self.val = self._build_dataset(validation_data)

    def _source(self, split: str):
        return load_dataset(self._source_name, split=split)

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

    def _build_dataset(self, data):
        encoder = _QuestionAnswerEncoder(
            self.tokenizer,
            self.vocab,
            self.context_length,
            self.question_length,
        )
        answer_span_aligner = _AnswerSpanAligner(
            self.tokenizer,
            self.context_length,
        )
        return self._item_dataset_type(data, encoder, answer_span_aligner)

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
        return [self.vocab.lookup_token(int(index)) for index in indices]

import torch
import torch.utils.data

from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from emperor.base.utils import DataModule


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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context = torch.tensor(
            _encode(sample["context"], self.tokenizer, self.vocab, self.context_length),
            dtype=torch.long,
        )
        question = torch.tensor(
            _encode(sample["question"], self.tokenizer, self.vocab, self.question_length),
            dtype=torch.long,
        )
        # unanswerable questions (SQuAD v2) have empty answers — represented as (-1, -1)
        if sample["answers"]["answer_start"]:
            start = min(sample["answers"]["answer_start"][0], self.context_length - 1)
            end = min(start + len(self.tokenizer(sample["answers"]["text"][0])) - 1, self.context_length - 1)
        else:
            start, end = -1, -1
        return context, question, torch.tensor(start, dtype=torch.long), torch.tensor(end, dtype=torch.long)


class SQuADv2(DataModule):
    vocab_size: int = 97854  # approximate SQuAD v2 vocab size
    num_classes: int = 2  # start + end span positions (-1 for unanswerable)
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
        load_dataset("squad_v2", split="train")
        load_dataset("squad_v2", split="validation")

    def _setup_fit(self) -> None:
        train_data = load_dataset("squad_v2", split="train")
        val_data = load_dataset("squad_v2", split="validation")
        self._build_vocab(train_data)
        self.train = _QADataset(train_data, self.tokenizer, self.vocab, self.context_length, self.question_length)
        self.val = _QADataset(val_data, self.tokenizer, self.vocab, self.context_length, self.question_length)

    def _setup_validate(self) -> None:
        val_data = load_dataset("squad_v2", split="validation")
        self._build_vocab(val_data)
        self.val = _QADataset(val_data, self.tokenizer, self.vocab, self.context_length, self.question_length)

    def _build_vocab(self, data) -> None:
        if self.vocab is not None:
            return
        self.vocab = build_vocab_from_iterator(
            _yield_tokens(data, self.tokenizer),
            specials=["<unk>", "<pad>"],
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        SQuADv2.vocab_size = len(self.vocab)

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

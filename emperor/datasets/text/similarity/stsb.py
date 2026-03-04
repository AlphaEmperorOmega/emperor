import torch
import torch.utils.data

from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from emperor.base.utils import DataModule


def _yield_tokens(samples, tokenizer):
    for sample in samples:
        yield tokenizer(sample["sentence1"])
        yield tokenizer(sample["sentence2"])


def _encode(text, tokenizer, vocab, sequence_length):
    tokens = vocab(tokenizer(text))[:sequence_length]
    padding = [vocab["<pad>"]] * (sequence_length - len(tokens))
    return tokens + padding


class _STSDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, vocab, sequence_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sentence1 = torch.tensor(
            _encode(sample["sentence1"], self.tokenizer, self.vocab, self.sequence_length),
            dtype=torch.long,
        )
        sentence2 = torch.tensor(
            _encode(sample["sentence2"], self.tokenizer, self.vocab, self.sequence_length),
            dtype=torch.long,
        )
        # score is a float in [0.0, 5.0]
        score = torch.tensor(sample["score"] / 5.0, dtype=torch.float32)
        return sentence1, sentence2, score


class STSb(DataModule):
    vocab_size: int = 19658  # approximate STS-B vocab size
    num_classes: int = 1  # regression: single similarity score in [0, 1]
    sequence_length: int = 64

    def __init__(
        self,
        batch_size: int = 64,
        sequence_length: int = 64,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None

    def prepare_data(self) -> None:
        load_dataset("glue", "stsb", split="train")
        load_dataset("glue", "stsb", split="validation")

    def _setup_fit(self) -> None:
        train_data = list(load_dataset("glue", "stsb", split="train"))
        val_data = list(load_dataset("glue", "stsb", split="validation"))
        self._build_vocab(train_data)
        self.train = _STSDataset(train_data, self.tokenizer, self.vocab, self.sequence_length)
        self.val = _STSDataset(val_data, self.tokenizer, self.vocab, self.sequence_length)

    def _setup_validate(self) -> None:
        val_data = list(load_dataset("glue", "stsb", split="validation"))
        self._build_vocab(val_data)
        self.val = _STSDataset(val_data, self.tokenizer, self.vocab, self.sequence_length)

    def _build_vocab(self, data: list) -> None:
        if self.vocab is not None:
            return
        self.vocab = build_vocab_from_iterator(
            _yield_tokens(data, self.tokenizer),
            specials=["<unk>", "<pad>"],
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        STSb.vocab_size = len(self.vocab)

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
        # scores are continuous — return formatted strings
        return [f"{float(i):.2f}" for i in indices]

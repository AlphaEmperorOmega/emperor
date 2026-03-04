import torch
import torch.utils.data

from torchtext.datasets import IMDB as IMDBDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from emperor.base.utils import DataModule


def _yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


def _encode(text, tokenizer, vocab, sequence_length):
    tokens = vocab(tokenizer(text))[:sequence_length]
    padding = [vocab["<pad>"]] * (sequence_length - len(tokens))
    return tokens + padding


_LABEL_MAP = {"neg": 0, "pos": 1}


class IMDB(DataModule):
    vocab_size: int = 101521  # approximate IMDB vocab size with basic_english tokenizer
    num_classes: int = 2
    flattened_input_dim: int = vocab_size
    sequence_length: int = 256

    def __init__(
        self,
        batch_size: int = 64,
        sequence_length: int = 256,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None

    def prepare_data(self) -> None:
        IMDBDataset(root=self.root, split="train")
        IMDBDataset(root=self.root, split="test")

    def _setup_fit(self) -> None:
        self._build_vocab()
        train_iter = IMDBDataset(root=self.root, split="train")
        val_iter = IMDBDataset(root=self.root, split="test")
        self.train = self._build_dataset(train_iter)
        self.val = self._build_dataset(val_iter)

    def _setup_validate(self) -> None:
        self._build_vocab()
        val_iter = IMDBDataset(root=self.root, split="test")
        self.val = self._build_dataset(val_iter)

    def _build_vocab(self) -> None:
        if self.vocab is not None:
            return
        train_iter = IMDBDataset(root=self.root, split="train")
        self.vocab = build_vocab_from_iterator(
            _yield_tokens(train_iter, self.tokenizer),
            specials=["<unk>", "<pad>"],
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        IMDB.vocab_size = len(self.vocab)
        IMDB.flattened_input_dim = len(self.vocab)

    def _build_dataset(self, data_iter) -> torch.utils.data.TensorDataset:
        inputs, labels = [], []
        for label, text in data_iter:
            inputs.append(_encode(text, self.tokenizer, self.vocab, self.sequence_length))
            labels.append(_LABEL_MAP[label])
        return torch.utils.data.TensorDataset(
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
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
        labels = ["negative", "positive"]
        return [labels[int(i)] for i in indices]

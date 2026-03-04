import torch
import torch.utils.data

from torchtext.datasets import PennTreebank as PennTreebankDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from emperor.base.utils import DataModule


def _yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)


class PennTreebank(DataModule):
    vocab_size: int = 10000  # standard PennTreebank vocab size
    num_classes: int = vocab_size
    flattened_input_dim: int = vocab_size
    sequence_length: int = 35

    def __init__(
        self,
        batch_size: int = 64,
        sequence_length: int = 35,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None

    def prepare_data(self) -> None:
        PennTreebankDataset(root=self.root, split="train")
        PennTreebankDataset(root=self.root, split="valid")

    def _setup_fit(self) -> None:
        self._build_vocab()
        train_iter = PennTreebankDataset(root=self.root, split="train")
        val_iter = PennTreebankDataset(root=self.root, split="valid")
        self.train = self._build_dataset(train_iter)
        self.val = self._build_dataset(val_iter)

    def _setup_validate(self) -> None:
        self._build_vocab()
        val_iter = PennTreebankDataset(root=self.root, split="valid")
        self.val = self._build_dataset(val_iter)

    def _build_vocab(self) -> None:
        if self.vocab is not None:
            return
        train_iter = PennTreebankDataset(root=self.root, split="train")
        self.vocab = build_vocab_from_iterator(
            _yield_tokens(train_iter, self.tokenizer),
            specials=["<unk>"],
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        PennTreebank.vocab_size = len(self.vocab)
        PennTreebank.flattened_input_dim = len(self.vocab)
        PennTreebank.num_classes = len(self.vocab)

    def _build_dataset(self, data_iter) -> torch.utils.data.TensorDataset:
        tokens = [self.vocab[t] for text in data_iter for t in self.tokenizer(text)]
        data = torch.tensor(tokens, dtype=torch.long)
        num_sequences = (data.size(0) - 1) // self.sequence_length
        data = data[: num_sequences * self.sequence_length + 1]
        inputs = data[:-1].view(-1, self.sequence_length)
        targets = data[1:].view(-1, self.sequence_length)
        return torch.utils.data.TensorDataset(inputs, targets)

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

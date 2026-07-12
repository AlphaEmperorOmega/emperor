from collections import Counter
from collections.abc import Iterable, Sequence

import torch
import torch.utils.data
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2 as WikiText2Dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator

from emperor.base.data import DataModule


def _yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)


def _build_compatible_vocab(token_iterator):
    try:
        return build_vocab_from_iterator(token_iterator, specials=["<unk>"])
    except TypeError:
        counter = Counter()
        for tokens in token_iterator:
            counter.update(tokens)
        return Vocab(counter, specials=["<unk>"])


def _set_unknown_default(vocab) -> None:
    if hasattr(vocab, "set_default_index"):
        vocab.set_default_index(vocab["<unk>"])


def _lookup_token(vocab, index: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(index)
    return vocab.itos[index]


def _legacy_text_field(tokenizer):
    from torchtext.data import Field

    return Field(tokenize=tokenizer)


class WikiText2(DataModule):
    vocab_size: int = (
        28782  # standard WikiText2 vocab size with basic_english tokenizer
    )
    num_classes: int = vocab_size
    flattened_input_dim: int = vocab_size
    sequence_length: int = 35

    def __init__(
        self,
        batch_size: int = 64,
        sequence_length: int = 35,
        root: str = "data",
        num_workers: int = 4,
        drop_last: bool = True,
    ) -> None:
        super().__init__(root=root, num_workers=num_workers)
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than 0.")
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.drop_last = drop_last
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None
        self._legacy_split_tokens: dict[str, tuple[str, ...]] | None = None

    def prepare_data(self) -> None:
        for split in ("train", "valid", "test"):
            next(iter(self._dataset(split)), None)

    def setup(self, stage: str | None = None) -> None:
        if stage is None:
            self._setup_fit()
            self._setup_test()
            return
        super().setup(stage)

    def _setup_fit(self) -> None:
        self._build_vocab()
        train_iter = self._dataset("train")
        val_iter = self._dataset("valid")
        self.train = self._build_dataset(train_iter)
        self.val = self._build_dataset(val_iter)

    def _setup_validate(self) -> None:
        self._build_vocab()
        val_iter = self._dataset("valid")
        self.val = self._build_dataset(val_iter)

    def _setup_test(self) -> None:
        self._build_vocab()
        self.test = self._build_dataset(self._dataset("test"))

    def _dataset(self, split: str):
        if isinstance(WikiText2Dataset, type) and hasattr(
            WikiText2Dataset,
            "splits",
        ):
            return iter(self._load_legacy_split_tokens()[split])
        return WikiText2Dataset(root=self.root, split=split)

    def _load_legacy_split_tokens(self) -> dict[str, tuple[str, ...]]:
        if self._legacy_split_tokens is not None:
            return self._legacy_split_tokens
        field = _legacy_text_field(self.tokenizer)
        datasets = WikiText2Dataset.splits(field, root=self.root)
        self._legacy_split_tokens = {
            split: tuple(
                token for example in dataset.examples for token in example.text
            )
            for split, dataset in zip(
                ("train", "valid", "test"),
                datasets,
                strict=True,
            )
        }
        return self._legacy_split_tokens

    def _build_vocab(self) -> None:
        if self.vocab is not None:
            return
        train_iter = self._dataset("train")
        self.vocab = _build_compatible_vocab(_yield_tokens(train_iter, self.tokenizer))
        _set_unknown_default(self.vocab)
        type(self).vocab_size = len(self.vocab)
        type(self).flattened_input_dim = len(self.vocab)
        type(self).num_classes = len(self.vocab)

    def _build_dataset(
        self,
        data_iter: Iterable[str],
    ) -> torch.utils.data.TensorDataset:
        if self.vocab is None:
            raise RuntimeError("Vocabulary must be built before the dataset.")
        tokens = [self.vocab[t] for text in data_iter for t in self.tokenizer(text)]
        data = torch.tensor(tokens, dtype=torch.long)
        num_sequences = (data.size(0) - 1) // self.sequence_length
        if num_sequences <= 0:
            inputs = torch.empty((0, self.sequence_length), dtype=torch.long)
            targets = torch.empty((0, self.sequence_length), dtype=torch.long)
        else:
            data = data[: num_sequences * self.sequence_length + 1]
            inputs = data[:-1].view(num_sequences, self.sequence_length)
            targets = data[1:].view(num_sequences, self.sequence_length)
        return torch.utils.data.TensorDataset(inputs, targets)

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def _get_test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def _text_labels(self, indices) -> list:
        if self.vocab is None:
            raise RuntimeError("Vocabulary must be built before decoding IDs.")
        return [_lookup_token(self.vocab, int(i)) for i in indices]

    def encode_text(self, text: str) -> list[int]:
        if self.vocab is None:
            self._build_vocab()
        return [int(self.vocab[token]) for token in self.tokenizer(text)]

    def decode_ids(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        return " ".join(self._text_labels(token_ids))

    def decode_batch(
        self,
        batch_token_ids: Sequence[Sequence[int]] | torch.Tensor,
    ) -> list[str]:
        return [self.decode_ids(token_ids) for token_ids in batch_token_ids]

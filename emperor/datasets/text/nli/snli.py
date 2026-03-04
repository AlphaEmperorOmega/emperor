import torch
import torch.utils.data

from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from emperor.base.utils import DataModule


# SNLI labels: entailment=0, neutral=1, contradiction=2  (-1 = unlabelled, filtered out)
_LABEL_MAP = {0: 0, 1: 1, 2: 2}

_LABEL_NAMES = ["entailment", "neutral", "contradiction"]


def _yield_tokens(samples, tokenizer):
    for sample in samples:
        yield tokenizer(sample["premise"])
        yield tokenizer(sample["hypothesis"])


def _encode(text, tokenizer, vocab, sequence_length):
    tokens = vocab(tokenizer(text))[:sequence_length]
    padding = [vocab["<pad>"]] * (sequence_length - len(tokens))
    return tokens + padding


class _NLIDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, vocab, sequence_length):
        self.samples = [s for s in samples if s["label"] != -1]
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        premise = torch.tensor(
            _encode(sample["premise"], self.tokenizer, self.vocab, self.sequence_length),
            dtype=torch.long,
        )
        hypothesis = torch.tensor(
            _encode(sample["hypothesis"], self.tokenizer, self.vocab, self.sequence_length),
            dtype=torch.long,
        )
        label = torch.tensor(_LABEL_MAP[sample["label"]], dtype=torch.long)
        return premise, hypothesis, label


class SNLI(DataModule):
    vocab_size: int = 36635  # approximate SNLI vocab size
    num_classes: int = 3  # entailment, neutral, contradiction
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
        load_dataset("snli", split="train")
        load_dataset("snli", split="validation")

    def _setup_fit(self) -> None:
        train_data = list(load_dataset("snli", split="train"))
        val_data = list(load_dataset("snli", split="validation"))
        self._build_vocab(train_data)
        self.train = _NLIDataset(train_data, self.tokenizer, self.vocab, self.sequence_length)
        self.val = _NLIDataset(val_data, self.tokenizer, self.vocab, self.sequence_length)

    def _setup_validate(self) -> None:
        val_data = list(load_dataset("snli", split="validation"))
        self._build_vocab(val_data)
        self.val = _NLIDataset(val_data, self.tokenizer, self.vocab, self.sequence_length)

    def _build_vocab(self, data: list) -> None:
        if self.vocab is not None:
            return
        self.vocab = build_vocab_from_iterator(
            _yield_tokens(data, self.tokenizer),
            specials=["<unk>", "<pad>"],
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        SNLI.vocab_size = len(self.vocab)

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
        return [_LABEL_NAMES[int(i)] for i in indices]

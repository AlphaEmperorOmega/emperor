import torch
import torch.utils.data

from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from emperor.base.utils import DataModule


def _yield_tokens(samples, tokenizer):
    for sample in samples:
        yield tokenizer(sample["document"])
        yield tokenizer(sample["summary"])


def _encode(text, tokenizer, vocab, sequence_length):
    tokens = [vocab["<bos>"]] + vocab(tokenizer(text))[:sequence_length - 2] + [vocab["<eos>"]]
    padding = [vocab["<pad>"]] * (sequence_length - len(tokens))
    return tokens + padding


class _SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, vocab, document_length, summary_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.document_length = document_length
        self.summary_length = summary_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        document = torch.tensor(
            _encode(sample["document"], self.tokenizer, self.vocab, self.document_length),
            dtype=torch.long,
        )
        summary = torch.tensor(
            _encode(sample["summary"], self.tokenizer, self.vocab, self.summary_length),
            dtype=torch.long,
        )
        return document, summary


class XSum(DataModule):
    vocab_size: int = 50000  # approximate
    num_classes: int = vocab_size  # generative: output is vocab distribution
    document_length: int = 512
    summary_length: int = 64  # XSum summaries are single sentences

    def __init__(
        self,
        batch_size: int = 16,
        document_length: int = 512,
        summary_length: int = 64,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.document_length = document_length
        self.summary_length = summary_length
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None

    def prepare_data(self) -> None:
        load_dataset("xsum", split="train")
        load_dataset("xsum", split="validation")

    def _setup_fit(self) -> None:
        train_data = load_dataset("xsum", split="train")
        val_data = load_dataset("xsum", split="validation")
        self._build_vocab(train_data)
        self.train = _SummarizationDataset(train_data, self.tokenizer, self.vocab, self.document_length, self.summary_length)
        self.val = _SummarizationDataset(val_data, self.tokenizer, self.vocab, self.document_length, self.summary_length)

    def _setup_validate(self) -> None:
        val_data = load_dataset("xsum", split="validation")
        self._build_vocab(val_data)
        self.val = _SummarizationDataset(val_data, self.tokenizer, self.vocab, self.document_length, self.summary_length)

    def _build_vocab(self, data) -> None:
        if self.vocab is not None:
            return
        self.vocab = build_vocab_from_iterator(
            _yield_tokens(data, self.tokenizer),
            max_tokens=50000,
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        XSum.vocab_size = len(self.vocab)
        XSum.num_classes = len(self.vocab)

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

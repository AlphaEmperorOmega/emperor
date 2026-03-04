import torch
import torch.utils.data

from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from emperor.base.utils import DataModule


def _yield_tokens(samples, tokenizer):
    for sample in samples:
        yield tokenizer(sample["article"])
        yield tokenizer(sample["highlights"])


def _encode(text, tokenizer, vocab, sequence_length):
    tokens = [vocab["<bos>"]] + vocab(tokenizer(text))[:sequence_length - 2] + [vocab["<eos>"]]
    padding = [vocab["<pad>"]] * (sequence_length - len(tokens))
    return tokens + padding


class _SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, vocab, article_length, summary_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.article_length = article_length
        self.summary_length = summary_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        article = torch.tensor(
            _encode(sample["article"], self.tokenizer, self.vocab, self.article_length),
            dtype=torch.long,
        )
        summary = torch.tensor(
            _encode(sample["highlights"], self.tokenizer, self.vocab, self.summary_length),
            dtype=torch.long,
        )
        return article, summary


class CnnDailyMail(DataModule):
    vocab_size: int = 50000  # approximate
    num_classes: int = vocab_size  # generative: output is vocab distribution
    article_length: int = 512
    summary_length: int = 128

    def __init__(
        self,
        batch_size: int = 16,
        article_length: int = 512,
        summary_length: int = 128,
        version: str = "3.0.0",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.article_length = article_length
        self.summary_length = summary_length
        self.version = version
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None

    def prepare_data(self) -> None:
        load_dataset("cnn_dailymail", self.version, split="train")
        load_dataset("cnn_dailymail", self.version, split="validation")

    def _setup_fit(self) -> None:
        train_data = load_dataset("cnn_dailymail", self.version, split="train")
        val_data = load_dataset("cnn_dailymail", self.version, split="validation")
        self._build_vocab(train_data)
        self.train = _SummarizationDataset(train_data, self.tokenizer, self.vocab, self.article_length, self.summary_length)
        self.val = _SummarizationDataset(val_data, self.tokenizer, self.vocab, self.article_length, self.summary_length)

    def _setup_validate(self) -> None:
        val_data = load_dataset("cnn_dailymail", self.version, split="validation")
        self._build_vocab(val_data)
        self.val = _SummarizationDataset(val_data, self.tokenizer, self.vocab, self.article_length, self.summary_length)

    def _build_vocab(self, data) -> None:
        if self.vocab is not None:
            return
        self.vocab = build_vocab_from_iterator(
            _yield_tokens(data, self.tokenizer),
            max_tokens=50000,
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        CnnDailyMail.vocab_size = len(self.vocab)
        CnnDailyMail.num_classes = len(self.vocab)

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

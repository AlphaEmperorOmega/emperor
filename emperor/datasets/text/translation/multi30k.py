import torch
import torch.utils.data

from torchtext.datasets import Multi30k as Multi30kDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from emperor.base.utils import DataModule


def _yield_tokens(data_iter, tokenizer, index):
    for pair in data_iter:
        yield tokenizer(pair[index])


def _encode(text, tokenizer, vocab, sequence_length):
    tokens = [vocab["<bos>"]] + vocab(tokenizer(text))[:sequence_length - 2] + [vocab["<eos>"]]
    padding = [vocab["<pad>"]] * (sequence_length - len(tokens))
    return tokens + padding


class _TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, src_length, tgt_length):
        self.pairs = pairs
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_length = src_length
        self.tgt_length = tgt_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src = torch.tensor(
            _encode(src_text, self.src_tokenizer, self.src_vocab, self.src_length),
            dtype=torch.long,
        )
        tgt = torch.tensor(
            _encode(tgt_text, self.tgt_tokenizer, self.tgt_vocab, self.tgt_length),
            dtype=torch.long,
        )
        return src, tgt


class Multi30k(DataModule):
    src_vocab_size: int = 7854   # approximate German vocab size
    tgt_vocab_size: int = 5893   # approximate English vocab size
    num_classes: int = tgt_vocab_size  # generative: output is target vocab distribution
    sequence_length: int = 64

    def __init__(
        self,
        batch_size: int = 128,
        sequence_length: int = 64,
        language_pair: tuple = ("de", "en"),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.language_pair = language_pair
        self.src_tokenizer = get_tokenizer("basic_english")
        self.tgt_tokenizer = get_tokenizer("basic_english")
        self.src_vocab = None
        self.tgt_vocab = None

    def prepare_data(self) -> None:
        Multi30kDataset(root=self.root, split="train", language_pair=self.language_pair)
        Multi30kDataset(root=self.root, split="valid", language_pair=self.language_pair)

    def _setup_fit(self) -> None:
        train_pairs = list(Multi30kDataset(root=self.root, split="train", language_pair=self.language_pair))
        val_pairs = list(Multi30kDataset(root=self.root, split="valid", language_pair=self.language_pair))
        self._build_vocabs(train_pairs)
        self.train = _TranslationDataset(train_pairs, self.src_tokenizer, self.tgt_tokenizer, self.src_vocab, self.tgt_vocab, self.sequence_length, self.sequence_length)
        self.val = _TranslationDataset(val_pairs, self.src_tokenizer, self.tgt_tokenizer, self.src_vocab, self.tgt_vocab, self.sequence_length, self.sequence_length)

    def _setup_validate(self) -> None:
        val_pairs = list(Multi30kDataset(root=self.root, split="valid", language_pair=self.language_pair))
        self._build_vocabs(val_pairs)
        self.val = _TranslationDataset(val_pairs, self.src_tokenizer, self.tgt_tokenizer, self.src_vocab, self.tgt_vocab, self.sequence_length, self.sequence_length)

    def _build_vocabs(self, pairs: list) -> None:
        if self.src_vocab is not None:
            return
        self.src_vocab = build_vocab_from_iterator(
            _yield_tokens(pairs, self.src_tokenizer, 0),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
        )
        self.src_vocab.set_default_index(self.src_vocab["<unk>"])
        self.tgt_vocab = build_vocab_from_iterator(
            _yield_tokens(pairs, self.tgt_tokenizer, 1),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"],
        )
        self.tgt_vocab.set_default_index(self.tgt_vocab["<unk>"])
        Multi30k.src_vocab_size = len(self.src_vocab)
        Multi30k.tgt_vocab_size = len(self.tgt_vocab)
        Multi30k.num_classes = len(self.tgt_vocab)

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
        return [self.tgt_vocab.lookup_token(int(i)) for i in indices]

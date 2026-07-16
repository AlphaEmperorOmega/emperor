import torch
import torch.utils.data
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import PennTreebank as PennTreebankDataset
from torchtext.datasets import WikiText2 as WikiText2Dataset
from torchtext.datasets import WikiText103 as WikiText103Dataset
from torchtext.vocab import build_vocab_from_iterator

from emperor.datasets._base import DataModule
from emperor.datasets.text._bert_vocabulary import (
    BERT_SPECIAL_TOKENS,
    BertSpecialTokenIds,
    get_bert_special_token_ids,
    set_bert_default_index,
)
from emperor.datasets.text.masked_language_modeling._collation import (
    MaskedLanguageModelingCollator,
)
from emperor.datasets.text.masked_language_modeling._windows import (
    build_mlm_token_windows,
)


def _yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)


class _TorchTextMaskedLanguageModeling(DataModule):
    vocab_size: int = 0
    num_classes: int = 0
    flattened_input_dim: int = 0
    sequence_length: int = 35
    torchtext_dataset = None
    train_split = "train"
    validation_split = "valid"
    test_split = "test"

    def __init__(
        self,
        batch_size: int = 64,
        sequence_length: int = 35,
        mlm_probability: float = 0.15,
        root: str = "data",
        num_workers: int = 4,
        drop_last: bool = True,
    ) -> None:
        super().__init__(root=root, num_workers=num_workers)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.mlm_probability = mlm_probability
        self.drop_last = drop_last
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None
        self.special_token_ids: BertSpecialTokenIds | None = None
        self.collator: MaskedLanguageModelingCollator | None = None

    def prepare_data(self) -> None:
        self._dataset(self.train_split)
        self._dataset(self.validation_split)
        self._dataset(self.test_split)

    def _setup_fit(self) -> None:
        self._build_vocab()
        self.train = self._build_dataset(self._dataset(self.train_split))
        self.val = self._build_dataset(self._dataset(self.validation_split))

    def _setup_validate(self) -> None:
        self._build_vocab()
        self.val = self._build_dataset(self._dataset(self.validation_split))

    def _setup_test(self) -> None:
        self._build_vocab()
        self.test = self._build_dataset(self._dataset(self.test_split))

    def _get_test_dataloader(self):
        return self._dataloader(self.test, train=False)

    def _dataset(self, split: str):
        return type(self).torchtext_dataset(root=self.root, split=split)

    def _build_vocab(self) -> None:
        if self.vocab is not None:
            return
        train_iter = self._dataset(self.train_split)
        self.vocab = build_vocab_from_iterator(
            _yield_tokens(train_iter, self.tokenizer),
            specials=BERT_SPECIAL_TOKENS,
        )
        self.special_token_ids = set_bert_default_index(self.vocab)
        type(self).vocab_size = len(self.vocab)
        type(self).flattened_input_dim = len(self.vocab)
        type(self).num_classes = len(self.vocab)
        self.collator = MaskedLanguageModelingCollator(
            special_token_ids=self.special_token_ids,
            vocab_size=len(self.vocab),
            mlm_probability=self.mlm_probability,
        )

    def _build_dataset(self, data_iter) -> torch.utils.data.TensorDataset:
        if self.special_token_ids is None:
            raise RuntimeError("Vocabulary must be built before the dataset.")
        tokens = [self.vocab[t] for text in data_iter for t in self.tokenizer(text)]
        windows = build_mlm_token_windows(
            tokens,
            sequence_length=self.sequence_length,
            special_token_ids=self.special_token_ids,
            add_special_tokens=True,
        )
        return torch.utils.data.TensorDataset(windows)

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return self._dataloader(data, train)

    def _dataloader(self, data, train: bool):
        if self.collator is None:
            raise RuntimeError("Vocabulary must be built before creating loaders.")
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=self.collator,
        )

    def _text_labels(self, indices) -> list:
        return [self.vocab.lookup_token(int(i)) for i in indices]

    def bert_special_token_ids(self) -> BertSpecialTokenIds:
        if self.vocab is None:
            self._build_vocab()
        return get_bert_special_token_ids(self.vocab)


class PennTreebankMaskedLanguageModeling(_TorchTextMaskedLanguageModeling):
    vocab_size: int = 10000
    num_classes: int = vocab_size
    flattened_input_dim: int = vocab_size
    sequence_length: int = 35
    torchtext_dataset = staticmethod(PennTreebankDataset)


class WikiText2MaskedLanguageModeling(_TorchTextMaskedLanguageModeling):
    vocab_size: int = 28782
    num_classes: int = vocab_size
    flattened_input_dim: int = vocab_size
    sequence_length: int = 35
    torchtext_dataset = staticmethod(WikiText2Dataset)


class WikiText103MaskedLanguageModeling(_TorchTextMaskedLanguageModeling):
    vocab_size: int = 267735
    num_classes: int = vocab_size
    flattened_input_dim: int = vocab_size
    sequence_length: int = 35
    torchtext_dataset = staticmethod(WikiText103Dataset)

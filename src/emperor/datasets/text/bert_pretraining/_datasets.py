import random
from collections.abc import Iterable

import torch
import torch.utils.data
from tokenizers import Tokenizer
from torchtext.datasets import PennTreebank as PennTreebankDataset
from torchtext.datasets import WikiText2 as WikiText2Dataset

from emperor.datasets._base import DataModule
from emperor.datasets.text._bert_vocabulary import (
    BERT_UNK_TOKEN,
    BertSpecialTokenIds,
    get_bert_special_token_ids,
)
from emperor.datasets.text.bert_pretraining._collation import (
    BertPretrainingCollator,
)
from emperor.datasets.text.bert_pretraining._examples import (
    build_bert_pretraining_examples,
)
from emperor.datasets.text.bert_pretraining._tokenizer import (
    BERT_PRETRAINING_TARGET_VOCAB_SIZE,
    train_local_wordpiece_tokenizer,
)


class _TorchTextBertPretraining(DataModule):
    target_vocab_size: int = BERT_PRETRAINING_TARGET_VOCAB_SIZE
    vocab_size: int = target_vocab_size
    num_classes: int = target_vocab_size
    flattened_input_dim: int = target_vocab_size
    sequence_length: int = 35
    torchtext_dataset = None
    train_split = "train"
    validation_split = "valid"
    test_split = "test"

    def __init__(
        self,
        batch_size: int = 64,
        sequence_length: int = 35,
        target_vocab_size: int = BERT_PRETRAINING_TARGET_VOCAB_SIZE,
        mlm_probability: float = 0.15,
        random_next_probability: float = 0.5,
        root: str = "data",
        num_workers: int = 4,
        drop_last: bool = True,
        seed: int | None = None,
    ) -> None:
        if sequence_length < 5:
            raise ValueError("sequence_length must be at least 5.")
        super().__init__(root=root, num_workers=num_workers)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_vocab_size = target_vocab_size
        self.mlm_probability = mlm_probability
        self.random_next_probability = random_next_probability
        self.drop_last = drop_last
        self.seed = None if seed is None else int(seed)
        self.tokenizer: Tokenizer | None = None
        self.special_token_ids: BertSpecialTokenIds | None = None
        self.collator: BertPretrainingCollator | None = None
        self.actual_vocab_size = 0
        self._legacy_split_text_units: dict[str, tuple[str, ...]] | None = None

    def prepare_data(self) -> None:
        self._dataset(self.train_split)
        self._dataset(self.validation_split)
        self._dataset(self.test_split)

    def _rng(self, offset: int = 0) -> random.Random | None:
        if self.seed is None:
            return None
        return random.Random(self.seed + offset)

    def _generator(self, offset: int = 0) -> torch.Generator | None:
        if self.seed is None:
            return None
        return torch.Generator().manual_seed(self.seed + offset)

    def _setup_fit(self) -> None:
        self._build_tokenizer()
        self.train = self._build_dataset(
            self._dataset(self.train_split),
            rng=self._rng(),
        )
        self.val = self._build_dataset(
            self._dataset(self.validation_split),
            rng=self._rng(1),
        )

    def _setup_validate(self) -> None:
        self._build_tokenizer()
        self.val = self._build_dataset(
            self._dataset(self.validation_split),
            rng=self._rng(1),
        )

    def _setup_test(self) -> None:
        self._build_tokenizer()
        self.test = self._build_dataset(
            self._dataset(self.test_split),
            rng=self._rng(2),
        )

    def _get_test_dataloader(self):
        return self._dataloader(self.test, train=False)

    def _dataset(self, split: str):
        dataset_source = type(self).torchtext_dataset
        if isinstance(dataset_source, type) and hasattr(dataset_source, "splits"):
            return iter(self._load_legacy_split_text_units()[split])
        return dataset_source(root=self.root, split=split)

    def _load_legacy_split_text_units(self) -> dict[str, tuple[str, ...]]:
        if self._legacy_split_text_units is not None:
            return self._legacy_split_text_units
        dataset_source = type(self).torchtext_dataset
        datasets = dataset_source.splits(_legacy_text_field(), root=self.root)
        self._legacy_split_text_units = {
            split: tuple(_legacy_dataset_text_units(dataset))
            for split, dataset in zip(
                (self.train_split, self.validation_split, self.test_split),
                datasets,
                strict=True,
            )
        }
        return self._legacy_split_text_units

    def _build_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return
        self.tokenizer = train_local_wordpiece_tokenizer(
            self._dataset(self.train_split),
            vocab_size=self.target_vocab_size,
        )
        self.special_token_ids = get_bert_special_token_ids(self.tokenizer)
        self.actual_vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)
        self.collator = BertPretrainingCollator(
            special_token_ids=self.special_token_ids,
            vocab_size=self.actual_vocab_size,
            mlm_probability=self.mlm_probability,
            generator=self._generator(),
        )

    def _build_dataset(
        self,
        data_iter,
        rng: random.Random | None,
    ) -> torch.utils.data.TensorDataset:
        if self.tokenizer is None or self.special_token_ids is None:
            raise RuntimeError("Tokenizer must be built before the dataset.")
        examples = build_bert_pretraining_examples(
            data_iter,
            tokenizer=self.tokenizer,
            sequence_length=self.sequence_length,
            special_token_ids=self.special_token_ids,
            random_next_probability=self.random_next_probability,
            rng=rng,
        )
        if not examples:
            return torch.utils.data.TensorDataset(
                torch.empty((0, self.sequence_length), dtype=torch.long),
                torch.empty((0, self.sequence_length), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )
        return torch.utils.data.TensorDataset(
            torch.stack([example.input_ids for example in examples]),
            torch.stack([example.token_type_ids for example in examples]),
            torch.stack([example.next_sentence_label for example in examples]),
        )

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return self._dataloader(data, train)

    def _dataloader(self, data, train: bool):
        if self.collator is None:
            raise RuntimeError("Tokenizer must be built before creating loaders.")
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=self.collator,
            generator=self._generator(0 if train else 1),
        )

    def _text_labels(self, indices) -> list:
        if self.tokenizer is None:
            self._build_tokenizer()
        labels = []
        for index in indices:
            token = self.tokenizer.id_to_token(int(index))
            labels.append(token if token is not None else BERT_UNK_TOKEN)
        return labels

    def bert_special_token_ids(self) -> BertSpecialTokenIds:
        if self.tokenizer is None:
            self._build_tokenizer()
        return get_bert_special_token_ids(self.tokenizer)


class PennTreebankBertPretraining(_TorchTextBertPretraining):
    torchtext_dataset = staticmethod(PennTreebankDataset)


class WikiText2BertPretraining(_TorchTextBertPretraining):
    torchtext_dataset = staticmethod(WikiText2Dataset)


def _legacy_text_field():
    from torchtext.data import Field

    return Field(tokenize=_split_on_whitespace)


def _split_on_whitespace(text: str) -> list[str]:
    return text.split()


def _legacy_dataset_text_units(dataset) -> Iterable[str]:
    for example in dataset.examples:
        text_unit_tokens = []
        for token in example.text:
            if token == "<eos>":
                if text_unit_tokens:
                    yield " ".join(text_unit_tokens)
                    text_unit_tokens = []
                continue
            text_unit_tokens.append(str(token))
        if text_unit_tokens:
            yield " ".join(text_unit_tokens)

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.utils.data
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer
from torch import Tensor
from torchtext.datasets import PennTreebank as PennTreebankDataset
from torchtext.datasets import WikiText2 as WikiText2Dataset

from emperor.base.data import DataModule
from emperor.datasets.text.masked_language_modeling import (
    MaskedLanguageModelingCollator,
)
from emperor.datasets.text.vocabulary import (
    BERT_SPECIAL_TOKENS,
    BERT_UNK_TOKEN,
    BertSpecialTokenIds,
    get_bert_special_token_ids,
)

BERT_PRETRAINING_TARGET_VOCAB_SIZE = 30522


@dataclass(frozen=True)
class BertNextSentencePair:
    sentence_a: str
    sentence_b: str
    next_sentence_label: int
    sentence_a_index: int
    sentence_b_index: int


@dataclass(frozen=True)
class BertPretrainingExample:
    input_ids: Tensor
    token_type_ids: Tensor
    next_sentence_label: Tensor


def train_local_wordpiece_tokenizer(
    text_units: Iterable[str],
    vocab_size: int = BERT_PRETRAINING_TARGET_VOCAB_SIZE,
) -> Tokenizer:
    training_units = [unit for unit in _normalise_text_units(text_units)]
    if not training_units:
        training_units = [BERT_UNK_TOKEN]

    tokenizer = Tokenizer(WordPiece(unk_token=BERT_UNK_TOKEN))
    tokenizer.normalizer = BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = BertPreTokenizer()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=BERT_SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(training_units, trainer=trainer)
    return tokenizer


def build_bert_next_sentence_pairs(
    text_units: Iterable[str],
    random_next_probability: float = 0.5,
    rng: random.Random | None = None,
) -> list[BertNextSentencePair]:
    if random_next_probability < 0.0 or random_next_probability > 1.0:
        raise ValueError("random_next_probability must be between 0.0 and 1.0.")

    units = list(_normalise_text_units(text_units))
    if len(units) < 2:
        return []

    rng = rng or random.Random()
    pairs = []
    for sentence_a_index in range(len(units) - 1):
        true_next_index = sentence_a_index + 1
        random_next_candidates = [
            index
            for index in range(len(units))
            if index != sentence_a_index and index != true_next_index
        ]
        use_random_next = (
            bool(random_next_candidates) and rng.random() < random_next_probability
        )
        if use_random_next:
            sentence_b_index = rng.choice(random_next_candidates)
            next_sentence_label = 1
        else:
            sentence_b_index = true_next_index
            next_sentence_label = 0

        pairs.append(
            BertNextSentencePair(
                sentence_a=units[sentence_a_index],
                sentence_b=units[sentence_b_index],
                next_sentence_label=next_sentence_label,
                sentence_a_index=sentence_a_index,
                sentence_b_index=sentence_b_index,
            )
        )
    return pairs


def build_bert_sentence_pair_inputs(
    tokens_a: Sequence[int],
    tokens_b: Sequence[int],
    sequence_length: int,
    special_token_ids: BertSpecialTokenIds,
) -> tuple[Tensor, Tensor]:
    if sequence_length < 5:
        raise ValueError("sequence_length must be at least 5 for BERT pairs.")

    tokens_a = list(tokens_a)
    tokens_b = list(tokens_b)
    if not tokens_a or not tokens_b:
        raise ValueError("Both sentence-pair segments must contain tokens.")

    _truncate_longest_first(tokens_a, tokens_b, sequence_length - 3)

    input_ids = [
        special_token_ids.cls,
        *tokens_a,
        special_token_ids.sep,
        *tokens_b,
        special_token_ids.sep,
    ]
    token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    padding_length = sequence_length - len(input_ids)
    if padding_length < 0:
        raise ValueError("Truncated sentence-pair input is still too long.")

    input_ids.extend([special_token_ids.pad] * padding_length)
    token_type_ids.extend([0] * padding_length)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
        token_type_ids,
        dtype=torch.long,
    )


def build_bert_pretraining_examples(
    text_units: Iterable[str],
    tokenizer: Tokenizer,
    sequence_length: int,
    special_token_ids: BertSpecialTokenIds,
    random_next_probability: float = 0.5,
    rng: random.Random | None = None,
) -> list[BertPretrainingExample]:
    examples = []
    pairs = build_bert_next_sentence_pairs(
        text_units,
        random_next_probability=random_next_probability,
        rng=rng,
    )
    for pair in pairs:
        tokens_a = tokenizer.encode(pair.sentence_a).ids
        tokens_b = tokenizer.encode(pair.sentence_b).ids
        if not tokens_a or not tokens_b:
            continue
        try:
            input_ids, token_type_ids = build_bert_sentence_pair_inputs(
                tokens_a=tokens_a,
                tokens_b=tokens_b,
                sequence_length=sequence_length,
                special_token_ids=special_token_ids,
            )
        except ValueError:
            continue
        examples.append(
            BertPretrainingExample(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                next_sentence_label=torch.tensor(
                    pair.next_sentence_label,
                    dtype=torch.long,
                ),
            )
        )
    return examples


class BertPretrainingCollator:
    def __init__(
        self,
        special_token_ids: BertSpecialTokenIds,
        vocab_size: int,
        mlm_probability: float = 0.15,
        mask_replace_probability: float = 0.8,
        random_replace_probability: float = 0.1,
        ignore_index: int = -100,
        generator: torch.Generator | None = None,
    ) -> None:
        self.mlm_collator = MaskedLanguageModelingCollator(
            special_token_ids=special_token_ids,
            vocab_size=vocab_size,
            mlm_probability=mlm_probability,
            mask_replace_probability=mask_replace_probability,
            random_replace_probability=random_replace_probability,
            ignore_index=ignore_index,
            generator=generator,
        )

    def __call__(
        self,
        batch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        input_ids, token_type_ids, next_sentence_labels = self._stack_batch(batch)
        input_ids, mlm_labels, attention_mask = self.mlm_collator(input_ids)
        return (
            input_ids,
            mlm_labels,
            attention_mask,
            token_type_ids,
            next_sentence_labels,
        )

    def _stack_batch(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        if len(batch) == 0:
            raise ValueError("BertPretrainingCollator received no samples.")

        input_ids = []
        token_type_ids = []
        next_sentence_labels = []
        for sample in batch:
            if not isinstance(sample, Sequence) or len(sample) != 3:
                raise TypeError(
                    "BertPretrainingCollator expects samples containing "
                    "(input_ids, token_type_ids, next_sentence_label)."
                )
            sample_input_ids, sample_token_type_ids, sample_next_sentence_label = sample
            input_ids.append(torch.as_tensor(sample_input_ids, dtype=torch.long))
            token_type_ids.append(
                torch.as_tensor(sample_token_type_ids, dtype=torch.long)
            )
            next_sentence_labels.append(
                torch.as_tensor(sample_next_sentence_label, dtype=torch.long)
            )

        return (
            torch.stack(input_ids),
            torch.stack(token_type_ids),
            torch.stack(next_sentence_labels).view(-1),
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


def _normalise_text_units(text_units: Iterable[str]) -> Iterable[str]:
    for unit in text_units:
        unit = str(unit).strip()
        if unit:
            yield unit


def _truncate_longest_first(
    tokens_a: list[int],
    tokens_b: list[int],
    max_content_length: int,
) -> None:
    while len(tokens_a) + len(tokens_b) > max_content_length:
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) > 1:
            tokens_a.pop()
        elif len(tokens_b) > 1:
            tokens_b.pop()
        elif len(tokens_a) > 1:
            tokens_a.pop()
        else:
            break

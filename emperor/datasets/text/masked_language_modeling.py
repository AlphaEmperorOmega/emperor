import torch
import torch.utils.data

from collections.abc import Sequence
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import PennTreebank as PennTreebankDataset
from torchtext.datasets import WikiText2 as WikiText2Dataset
from torchtext.datasets import WikiText103 as WikiText103Dataset
from torchtext.vocab import build_vocab_from_iterator

from emperor.base.utils import DataModule
from emperor.datasets.text.vocabulary import (
    BERT_SPECIAL_TOKENS,
    BertSpecialTokenIds,
    get_bert_special_token_ids,
    set_bert_default_index,
)


def _yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)


def build_mlm_token_windows(
    token_ids: Sequence[int] | Tensor,
    sequence_length: int,
    special_token_ids: BertSpecialTokenIds,
    add_special_tokens: bool = True,
) -> Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if add_special_tokens and sequence_length < 3:
        raise ValueError(
            "sequence_length must be at least 3 when adding [CLS] and [SEP]."
        )

    if isinstance(token_ids, Tensor):
        token_ids = token_ids.tolist()

    content_length = sequence_length - 2 if add_special_tokens else sequence_length
    windows = []
    for start in range(0, len(token_ids), content_length):
        chunk = list(token_ids[start : start + content_length])
        if not chunk:
            continue
        if add_special_tokens:
            window = [special_token_ids.cls, *chunk, special_token_ids.sep]
        else:
            window = chunk
        window.extend([special_token_ids.pad] * (sequence_length - len(window)))
        windows.append(window)

    if not windows:
        return torch.empty((0, sequence_length), dtype=torch.long)
    return torch.tensor(windows, dtype=torch.long)


class MaskedLanguageModelingCollator:
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
        self._validate_probabilities(
            mlm_probability,
            mask_replace_probability,
            random_replace_probability,
        )
        self.special_token_ids = special_token_ids
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability
        self.mask_replace_probability = mask_replace_probability
        self.random_replace_probability = random_replace_probability
        self.ignore_index = ignore_index
        self.generator = generator

        special_ids = set(special_token_ids.values())
        max_special_id = max(special_ids)
        if vocab_size <= max_special_id:
            raise ValueError(
                "vocab_size must be greater than every BERT special token id."
            )
        self.random_token_ids = torch.tensor(
            [index for index in range(vocab_size) if index not in special_ids],
            dtype=torch.long,
        )

    def __call__(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        token_ids = self._stack_batch(batch)
        input_ids = token_ids.clone()
        labels = token_ids.clone()
        attention_mask = (token_ids != self.special_token_ids.pad).long()

        masked_indices = self._sample_masked_indices(token_ids)
        labels[~masked_indices] = self.ignore_index

        replacement_sample = torch.rand(
            input_ids.shape,
            generator=self.generator,
            device=input_ids.device,
        )
        mask_replacement = masked_indices & (
            replacement_sample < self.mask_replace_probability
        )
        random_replacement = masked_indices & (
            replacement_sample >= self.mask_replace_probability
        )
        random_replacement &= (
            replacement_sample
            < self.mask_replace_probability + self.random_replace_probability
        )

        input_ids[mask_replacement] = self.special_token_ids.mask
        if self.random_token_ids.numel() > 0:
            random_token_ids = self.random_token_ids.to(input_ids.device)
            random_indices = torch.randint(
                random_token_ids.numel(),
                input_ids.shape,
                generator=self.generator,
                device=input_ids.device,
            )
            random_tokens = random_token_ids[random_indices]
            input_ids[random_replacement] = random_tokens[random_replacement]

        return input_ids, labels, attention_mask

    def _stack_batch(self, batch) -> Tensor:
        if isinstance(batch, Tensor):
            token_ids = batch
        else:
            if len(batch) == 0:
                raise ValueError("MaskedLanguageModelingCollator received no samples.")
            samples = []
            for sample in batch:
                if isinstance(sample, Tensor):
                    samples.append(sample)
                elif (
                    isinstance(sample, Sequence)
                    and len(sample) == 1
                    and isinstance(sample[0], Tensor)
                ):
                    samples.append(sample[0])
                else:
                    raise TypeError(
                        "MaskedLanguageModelingCollator expects tensors or "
                        "single-tensor samples."
                    )
            token_ids = torch.stack(samples)

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dim() != 2:
            raise ValueError(
                "MaskedLanguageModelingCollator expects a 2D token tensor."
            )
        return token_ids.long()

    def _sample_masked_indices(self, token_ids: Tensor) -> Tensor:
        probability_matrix = torch.full(
            token_ids.shape,
            self.mlm_probability,
            device=token_ids.device,
        )
        masked_indices = torch.bernoulli(
            probability_matrix,
            generator=self.generator,
        ).bool()
        masked_indices &= self._eligible_mask(token_ids)

        if not masked_indices.any():
            candidate_positions = self._eligible_mask(token_ids).nonzero(
                as_tuple=False
            )
            if candidate_positions.numel() > 0:
                selected_position = torch.randint(
                    candidate_positions.size(0),
                    (1,),
                    generator=self.generator,
                    device=token_ids.device,
                )
                row, column = candidate_positions[selected_position.item()]
                masked_indices[row, column] = True

        return masked_indices

    def _eligible_mask(self, token_ids: Tensor) -> Tensor:
        mask = torch.ones_like(token_ids, dtype=torch.bool)
        for special_token_id in self.special_token_ids.values():
            mask &= token_ids != special_token_id
        return mask

    def _validate_probabilities(
        self,
        mlm_probability: float,
        mask_replace_probability: float,
        random_replace_probability: float,
    ) -> None:
        probabilities = {
            "mlm_probability": mlm_probability,
            "mask_replace_probability": mask_replace_probability,
            "random_replace_probability": random_replace_probability,
        }
        for name, probability in probabilities.items():
            if probability < 0.0 or probability > 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0.")
        if mask_replace_probability + random_replace_probability > 1.0:
            raise ValueError(
                "mask_replace_probability + random_replace_probability "
                "must be <= 1.0."
            )


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

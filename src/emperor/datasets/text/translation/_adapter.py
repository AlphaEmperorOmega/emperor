from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch
from tokenizers import Tokenizer

from emperor.datasets._base import DataModule
from emperor.datasets.text.translation._download import (
    DownloadFunction,
    _download_file,
    _DownloadSupport,
    _exclusive_file_lock,
)
from emperor.datasets.text.translation._manifest import (
    BOS_ID,
    EOS_ID,
    FILES,
    PAD_ID,
    SOURCE_COMMIT,
    UNK_ID,
    UNK_TOKEN,
    VOCAB_SIZE,
)
from emperor.datasets.text.translation._tokenizer import _TokenizerSupport


class _TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        tokenizer: Tokenizer,
        source_sequence_length: int,
        target_sequence_length: int,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.source_sequence_length = source_sequence_length
        self.target_sequence_length = target_sequence_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self.pairs[index]
        return (
            self._encode(source, self.source_sequence_length),
            self._encode(target, self.target_sequence_length),
        )

    def _encode(self, text: str, maximum_length: int) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text).ids[: maximum_length - 2]
        token_ids = [BOS_ID, *token_ids, EOS_ID]
        token_ids.extend([PAD_ID] * (maximum_length - len(token_ids)))
        return torch.tensor(token_ids, dtype=torch.long)


class _Multi30k(_DownloadSupport, _TokenizerSupport, DataModule):
    """Pinned, shared-vocabulary Multi30k translation data module."""

    source_language = "de"
    target_language = "en"
    language_pair = (source_language, target_language)

    src_vocab_size = VOCAB_SIZE
    tgt_vocab_size = VOCAB_SIZE
    vocab_size = VOCAB_SIZE
    flattened_input_dim = VOCAB_SIZE
    num_classes = VOCAB_SIZE
    sequence_length = 64
    source_sequence_length = 64
    target_sequence_length = 64

    pad_token_id = PAD_ID
    unk_token_id = UNK_ID
    bos_token_id = BOS_ID
    eos_token_id = EOS_ID
    files = FILES

    def __init__(
        self,
        batch_size: int = 64,
        sequence_length: int | None = None,
        source_sequence_length: int | None = None,
        target_sequence_length: int | None = None,
        language_pair: tuple[str, str] | None = None,
        root: str | os.PathLike[str] = "data",
        num_workers: int = 0,
        downloader: DownloadFunction | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(root=str(root), num_workers=num_workers)
        requested_pair = language_pair or type(self).language_pair
        if tuple(requested_pair) != tuple(type(self).language_pair):
            raise ValueError(
                f"{type(self).__name__} only supports the language pair "
                f"{type(self).language_pair!r}; got {tuple(requested_pair)!r}."
            )
        broadcast_length = sequence_length or type(self).sequence_length
        self.batch_size = int(batch_size)
        self.source_sequence_length = int(source_sequence_length or broadcast_length)
        self.target_sequence_length = int(target_sequence_length or broadcast_length)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.source_sequence_length < 2 or self.target_sequence_length < 2:
            raise ValueError("source and target sequence lengths must be at least 2")
        self.sequence_length = max(
            self.source_sequence_length, self.target_sequence_length
        )
        self.language_pair = tuple(requested_pair)
        self.seed = None if seed is None else int(seed)
        self._downloader = downloader or _download_file
        self.cache_dir = Path(self.root) / "multi30k" / SOURCE_COMMIT
        self.archive_dir = self.cache_dir / "archives"
        self.corpus_dir = self.cache_dir / "corpus"
        self.tokenizer_path = self.cache_dir / "tokenizer.json"
        self.tokenizer_manifest_path = self.cache_dir / "tokenizer.manifest.json"
        self.tokenizer: Tokenizer | None = None
        self.train = None
        self.val = None
        self.test = None

    @property
    def _files(self):
        return type(self).files

    def prepare_data(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_dir / ".prepare.lock"
        with _exclusive_file_lock(lock_path):
            for file_spec in self._files:
                self._prepare_file(file_spec)
            self._prepare_tokenizer()

    def setup(self, stage: str | None = None) -> None:
        self.prepare_data()
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        if stage in (None, "fit"):
            self.train = self._dataset_for_split("train")
            self.val = self._dataset_for_split("val")
        elif stage == "validate":
            self.val = self._dataset_for_split("val")
        elif stage in ("test", "predict"):
            self.test = self._dataset_for_split("test")
        else:
            raise ValueError(f"Unsupported Multi30k setup stage: {stage!r}")

    def _dataset_for_split(self, split: str) -> _TranslationDataset:
        if self.tokenizer is None:
            raise RuntimeError("Multi30k tokenizer is not prepared")
        return _TranslationDataset(
            self._read_pairs(split),
            self.tokenizer,
            self.source_sequence_length,
            self.target_sequence_length,
        )

    def _read_pairs(self, split: str) -> list[tuple[str, str]]:
        source_path = (
            self.corpus_dir / f"{self._split_filename(split)}.{self.source_language}"
        )
        target_path = (
            self.corpus_dir / f"{self._split_filename(split)}.{self.target_language}"
        )
        source_lines = source_path.read_text(encoding="utf-8").splitlines()
        target_lines = target_path.read_text(encoding="utf-8").splitlines()
        if len(source_lines) != len(target_lines):
            raise RuntimeError(
                f"Multi30k {split} source/target line counts do not match: "
                f"{len(source_lines)} != {len(target_lines)}."
            )
        return list(zip(source_lines, target_lines, strict=True))

    def _split_filename(self, split: str) -> str:
        return "test_2016_flickr" if split == "test" else split

    def _loader(self, dataset, *, shuffle: bool) -> torch.utils.data.DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a Multi30k data loader")
        generator = (
            torch.Generator().manual_seed(self.seed)
            if shuffle and self.seed is not None
            else None
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            generator=generator,
        )

    def get_dataloader(self, train: bool):
        return self._loader(self.train if train else self.val, shuffle=train)

    def _get_test_dataloader(self):
        return self._loader(self.test, shuffle=False)

    def decode_ids(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        ids = [int(token_id) for token_id in token_ids]
        if EOS_ID in ids:
            ids = ids[: ids.index(EOS_ID)]
        ids = [token_id for token_id in ids if token_id not in (PAD_ID, BOS_ID)]
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def decode_batch(
        self, batch_token_ids: Sequence[Sequence[int]] | torch.Tensor
    ) -> list[str]:
        return [self.decode_ids(token_ids) for token_ids in batch_token_ids]

    def _text_labels(self, indices) -> list[str]:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        return [
            self.tokenizer.id_to_token(int(index)) or UNK_TOKEN for index in indices
        ]


class Multi30kDeEn(_Multi30k):
    source_language = "de"
    target_language = "en"
    language_pair = (source_language, target_language)


class Multi30kEnDe(_Multi30k):
    source_language = "en"
    target_language = "de"
    language_pair = (source_language, target_language)

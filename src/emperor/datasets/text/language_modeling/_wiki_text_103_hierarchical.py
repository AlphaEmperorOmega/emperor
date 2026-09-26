"""Raw WikiText article windows cached as memory-mapped Arrow records."""

import hashlib
import re
from pathlib import Path

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset

from emperor.datasets._base import DataModule
from emperor.decoding.hierarchical import (
    HierarchicalLanguageModelBatch,
    HierarchicalTextCodec,
)


def _articles(rows):
    pieces = []
    has_content = False
    for row in rows:
        text = row["text"]
        # WikiText renders subsections as '= = title = ='. Only level one
        # starts a document. Keep every supplied character, including blanks.
        heading = re.fullmatch(r"=\s+[^=\s].*?\s+=", text.strip())
        if heading and has_content:
            yield "".join(pieces)
            pieces = []
            has_content = False
        pieces.append(text)
        has_content = has_content or bool(text.strip())
    if pieces:
        yield "".join(pieces)


def _windows(source, sequence_length, max_token_bytes):
    codec = HierarchicalTextCodec(max_token_bytes)
    for article in _articles(source):
        yield from codec.training_windows(article, sequence_length)


class WikiText103Hierarchical(DataModule):
    """No vocabulary binding; labels are 256 bytes plus EOW and EOS."""

    flattened_input_dim = 256
    num_classes = 258
    sequence_length = 35
    hierarchical_language_model_flag = True

    def __init__(
        self,
        batch_size=64,
        sequence_length=35,
        max_token_bytes=64,
        root="data",
        num_workers=4,
        drop_last=False,
        seed=None,
    ):
        super().__init__(root=root, num_workers=num_workers)
        for name, value in (
            ("batch_size", batch_size),
            ("sequence_length", sequence_length),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        HierarchicalTextCodec(max_token_bytes)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_token_bytes = max_token_bytes
        self.drop_last = drop_last
        self.seed = seed

    def _dataset(self, split):
        return load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split=split,
            cache_dir=str(self.root),
        )

    def prepare_data(self):
        for split in ("train", "validation", "test"):
            self._dataset(split)

    def _build_split(self, split):
        source = self._dataset(split)
        identity = f"hat-raw-v1:{split}:{source._fingerprint}:{self.sequence_length}:{self.max_token_bytes}"
        fingerprint = hashlib.sha256(identity.encode()).hexdigest()
        return Dataset.from_generator(
            _windows,
            gen_kwargs={
                "source": source,
                "sequence_length": self.sequence_length,
                "max_token_bytes": self.max_token_bytes,
            },
            features=Features(
                {
                    "context_texts": Sequence(Value("string")),
                    "target_texts": Sequence(Value("string")),
                    "bos_mask": Sequence(Value("bool")),
                }
            ),
            fingerprint=fingerprint,
            cache_dir=str(Path(self.root) / "hierarchical_windows"),
            keep_in_memory=False,
        )

    def _setup_fit(self):
        self.train = self._build_split("train")
        self.val = self._build_split("validation")

    def _setup_validate(self):
        self.val = self._build_split("validation")

    def _setup_test(self):
        self.test = self._build_split("test")

    def _loader(self, data, train=False):
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train and len(data) > 0,
            num_workers=self.num_workers,
            drop_last=self.drop_last if train else False,
            collate_fn=HierarchicalLanguageModelBatch.collate,
            generator=torch.Generator().manual_seed(self.seed)
            if train and self.seed is not None
            else None,
        )

    def get_dataloader(self, train):
        return self._loader(self.train if train else self.val, train)

    def _get_test_dataloader(self):
        return self._loader(self.test)

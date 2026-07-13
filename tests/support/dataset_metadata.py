from __future__ import annotations

import gzip
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import emperor.datasets.text.translation.multi30k as multi30k_module
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.datasets.image.classification.mnist import Mnist
from emperor.datasets.text.bert_pretraining import (
    PennTreebankBertPretraining,
    WikiText2BertPretraining,
)
from emperor.datasets.text.language_modeling import PennTreebank, WikiText2
from emperor.datasets.text.translation import Multi30kDeEn, Multi30kEnDe
from emperor.datasets.text.translation.multi30k import Multi30kFile
from emperor.experiments.tasks import ExperimentTask
from PIL import Image
from torch.utils.data import Dataset

OFFLINE_TEXT_SPLITS = {
    "train": (
        "zero one two three four five",
        "six seven eight nine ten eleven",
        "twelve thirteen fourteen fifteen sixteen seventeen",
        "eighteen nineteen twenty twenty-one twenty-two twenty-three",
        "twenty-four twenty-five twenty-six twenty-seven twenty-eight twenty-nine",
    ),
    "valid": (
        "zero two four six eight ten",
        "one three five seven nine eleven",
        "twelve fourteen sixteen eighteen twenty twenty-two",
    ),
    "test": (
        "one four seven ten thirteen sixteen",
        "two five eight eleven fourteen seventeen",
        "three six nine twelve fifteen eighteen",
    ),
}

BERT_OFFLINE_TEXT_SPLITS = {
    "train": ("a b c", "d e f", "g h i", "j k l", "m n o"),
    "valid": ("a d g", "b e h", "c f i"),
    "test": ("g j m", "h k n", "i l o"),
}

DATASET_TASKS = {
    Mnist: ExperimentTask.IMAGE_CLASSIFICATION,
    FashionMNIST: ExperimentTask.IMAGE_CLASSIFICATION,
    Cifar10: ExperimentTask.IMAGE_CLASSIFICATION,
    Cifar100: ExperimentTask.IMAGE_CLASSIFICATION,
    PennTreebankBertPretraining: ExperimentTask.BERT_PRETRAINING,
    WikiText2BertPretraining: ExperimentTask.BERT_PRETRAINING,
    PennTreebank: ExperimentTask.CAUSAL_LANGUAGE_MODELING,
    WikiText2: ExperimentTask.CAUSAL_LANGUAGE_MODELING,
    Multi30kDeEn: ExperimentTask.TEXT_TRANSLATION,
    Multi30kEnDe: ExperimentTask.TEXT_TRANSLATION,
}


@dataclass(frozen=True)
class _VisionSpec:
    patch_target: str
    mode: str
    size: tuple[int, int]
    num_classes: int


_VISION_SPECS = {
    Mnist: _VisionSpec(
        "emperor.datasets.image.classification.mnist.datasets.MNIST",
        "L",
        (28, 28),
        10,
    ),
    FashionMNIST: _VisionSpec(
        "emperor.datasets.image.classification.fashion_mnist.datasets.FashionMNIST",
        "L",
        (28, 28),
        10,
    ),
    Cifar10: _VisionSpec(
        "emperor.datasets.image.classification.cifar_10.datasets.CIFAR10",
        "RGB",
        (32, 32),
        10,
    ),
    Cifar100: _VisionSpec(
        "emperor.datasets.image.classification.cifar_100.datasets.CIFAR100",
        "RGB",
        (32, 32),
        100,
    ),
}


class _OfflineVisionDataset(Dataset):
    def __init__(
        self,
        *,
        train: bool,
        transform,
        spec: _VisionSpec,
    ) -> None:
        self.train = train
        self.transform = transform
        self.spec = spec
        self.length = 40 if train else 12

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        pixel = (index * 5 + (0 if self.train else 1)) % 256
        color = pixel if self.spec.mode == "L" else (pixel, pixel, pixel)
        image = Image.new(self.spec.mode, self.spec.size, color=color)
        if self.transform is not None:
            image = self.transform(image)
        return image, index % self.spec.num_classes


@dataclass
class _OfflineLegacyExample:
    text: list[str]


class _OfflineLegacyDataset:
    def __init__(self, text_units: tuple[str, ...]) -> None:
        tokens = []
        for text_unit in text_units:
            tokens.extend(text_unit.split())
            tokens.append("<eos>")
        self.examples = [_OfflineLegacyExample(tokens)]


class _OfflineLegacyTextSource:
    @classmethod
    def splits(cls, _text_field, root):
        del root
        return tuple(
            _OfflineLegacyDataset(BERT_OFFLINE_TEXT_SPLITS[split])
            for split in ("train", "valid", "test")
        )


def _offline_modern_text_source(*, root, split):
    del root
    return iter(OFFLINE_TEXT_SPLITS[split])


def _causal_source_module(dataset_type: type) -> tuple[ModuleType, str]:
    module = __import__(dataset_type.__module__, fromlist=["*"])
    source_name = (
        "PennTreebankDataset" if dataset_type is PennTreebank else "WikiText2Dataset"
    )
    return module, source_name


def _translation_archives() -> tuple[dict[str, bytes], tuple[Multi30kFile, ...]]:
    text = {
        "train.de.gz": "Hallo Welt\nEin rotes Haus\nKleine Katze\nVogel fliegt\n",
        "train.en.gz": "Hello world\nA red house\nSmall cat\nBird flies\n",
        "val.de.gz": "Noch eins\nGuten Morgen\n",
        "val.en.gz": "One more\nGood morning\n",
        "test_2016_flickr.de.gz": "Test Satz\n",
        "test_2016_flickr.en.gz": "Test sentence\n",
    }
    archives = {
        name: gzip.compress(value.encode("utf-8"), mtime=0)
        for name, value in text.items()
    }
    split_by_name = {
        "train.de.gz": ("train", "de", 4),
        "train.en.gz": ("train", "en", 4),
        "val.de.gz": ("val", "de", 2),
        "val.en.gz": ("val", "en", 2),
        "test_2016_flickr.de.gz": ("test", "de", 1),
        "test_2016_flickr.en.gz": ("test", "en", 1),
    }
    files = tuple(
        Multi30kFile(
            split=split,
            language=language,
            filename=name,
            sha256=hashlib.sha256(archives[name]).hexdigest(),
            line_count=line_count,
        )
        for name, (split, language, line_count) in split_by_name.items()
    )
    return archives, files


@contextmanager
def offline_dataset_metadata(
    dataset_type: type,
    root: Path,
    *,
    seed: int | None,
) -> Iterator[object]:
    if dataset_type in _VISION_SPECS:
        spec = _VISION_SPECS[dataset_type]

        def source(*_args, train=True, transform=None, **_kwargs):
            return _OfflineVisionDataset(train=train, transform=transform, spec=spec)

        with patch(spec.patch_target, side_effect=source):
            dataset = dataset_type(batch_size=2, seed=seed)
            dataset.root = str(root)
            dataset.num_workers = 0
            yield dataset
        return

    if dataset_type in (PennTreebank, WikiText2):
        module, source_name = _causal_source_module(dataset_type)
        with patch.object(module, source_name, _offline_modern_text_source):
            yield dataset_type(
                batch_size=2,
                sequence_length=4,
                root=str(root),
                num_workers=0,
                drop_last=False,
                seed=seed,
            )
        return

    if dataset_type in (
        PennTreebankBertPretraining,
        WikiText2BertPretraining,
    ):
        with patch.object(dataset_type, "torchtext_dataset", _OfflineLegacyTextSource):
            yield dataset_type(
                batch_size=2,
                sequence_length=10,
                target_vocab_size=64,
                root=str(root),
                num_workers=0,
                drop_last=False,
                seed=seed,
            )
        return

    if dataset_type in (Multi30kDeEn, Multi30kEnDe):
        archives, files = _translation_archives()

        def downloader(url: str, destination: Path) -> None:
            destination.write_bytes(archives[url.rsplit("/", 1)[-1]])

        with patch.object(multi30k_module, "FILES", files):
            yield dataset_type(
                batch_size=2,
                source_sequence_length=7,
                target_sequence_length=6,
                root=root,
                num_workers=0,
                downloader=downloader,
                seed=seed,
            )
        return

    raise AssertionError(
        "Declared Dataset Metadata has no deterministic offline fixture: "
        f"{dataset_type.__module__}.{dataset_type.__qualname__}"
    )


__all__ = ["DATASET_TASKS", "offline_dataset_metadata"]

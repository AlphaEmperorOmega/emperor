from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import Normalize, RandomHorizontalFlip

from emperor.datasets.image.classification import (
    Cifar10,
    Cifar100,
    FashionMNIST,
    Mnist,
)


@dataclass(frozen=True)
class _DatasetCase:
    dataset_type: type
    patch_target: str
    mode: str
    source_size: tuple[int, int]
    channels: int
    default_size: tuple[int, int]
    num_classes: int
    first_label: str
    last_label: str
    normalization: tuple[tuple[float, ...], tuple[float, ...]]


_CASES = (
    _DatasetCase(
        Mnist,
        "emperor.datasets.image.classification._mnist.datasets.MNIST",
        "L",
        (28, 28),
        1,
        (28, 28),
        10,
        "0",
        "9",
        ((0.1307,), (0.3081,)),
    ),
    _DatasetCase(
        FashionMNIST,
        "emperor.datasets.image.classification._fashion_mnist.datasets.FashionMNIST",
        "L",
        (28, 28),
        1,
        (28, 28),
        10,
        "t-shirt",
        "ankle boot",
        ((0.2860,), (0.3530,)),
    ),
    _DatasetCase(
        Cifar10,
        "emperor.datasets.image.classification._cifar_10.datasets.CIFAR10",
        "RGB",
        (32, 32),
        3,
        (32, 32),
        10,
        "airplane",
        "truck",
        ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ),
    _DatasetCase(
        Cifar100,
        "emperor.datasets.image.classification._cifar_100.datasets.CIFAR100",
        "RGB",
        (32, 32),
        3,
        (32, 32),
        100,
        "apple",
        "worm",
        ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ),
)


class _TinyVisionDataset(Dataset):
    def __init__(
        self,
        *,
        train: bool,
        transform,
        mode: str,
        source_size: tuple[int, int],
        num_classes: int,
    ) -> None:
        self.train = train
        self.transform = transform
        self.mode = mode
        self.source_size = source_size
        self.num_classes = num_classes
        self.length = 21 if train else 5

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        color = index % 256 if self.mode == "L" else (index, index, index)
        image = Image.new(self.mode, self.source_size, color=color)
        if self.transform is not None:
            image = self.transform(image)
        return image, index % self.num_classes


class _DatasetFactory:
    def __init__(self, case: _DatasetCase) -> None:
        self.case = case
        self.datasets: list[_TinyVisionDataset] = []

    def __call__(
        self,
        *_args,
        train: bool,
        transform=None,
        **_kwargs,
    ) -> _TinyVisionDataset:
        dataset = _TinyVisionDataset(
            train=train,
            transform=transform,
            mode=self.case.mode,
            source_size=self.case.source_size,
            num_classes=self.case.num_classes,
        )
        self.datasets.append(dataset)
        return dataset


def _transform_signature(transform) -> tuple[type, ...]:
    return tuple(type(operation) for operation in transform.transforms)


class ImageClassificationDatasetTests(unittest.TestCase):
    def test_asymmetric_resize_matches_instance_metadata_and_emitted_geometry(
        self,
    ) -> None:
        resize = (6, 10)
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                catalog_dimensions = (
                    case.dataset_type.default_height,
                    case.dataset_type.default_width,
                    case.dataset_type.flattened_input_dim,
                )
                factory = _DatasetFactory(case)
                with patch(case.patch_target, side_effect=factory):
                    dataset = case.dataset_type(
                        batch_size=2,
                        resize=resize,
                        seed=0,
                    )
                    dataset.num_workers = 0
                    dataset.setup("test")

                image, label = dataset.test[0]
                expected_flattened_dim = case.channels * resize[0] * resize[1]
                self.assertEqual(image.shape, torch.Size([case.channels, 6, 10]))
                self.assertEqual(label, 0)
                self.assertEqual(dataset.default_height, 6)
                self.assertEqual(dataset.default_width, 10)
                self.assertEqual(dataset.flattened_input_dim, expected_flattened_dim)
                self.assertEqual(
                    dataset.resolved_metadata.flattened_input_dim,
                    expected_flattened_dim,
                )
                self.assertEqual(dataset.resolved_metadata.num_classes, case.num_classes)
                self.assertEqual(
                    (
                        case.dataset_type.default_height,
                        case.dataset_type.default_width,
                        case.dataset_type.flattened_input_dim,
                    ),
                    catalog_dimensions,
                )

    def test_fit_and_standalone_validation_use_the_same_deterministic_transform(
        self,
    ) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                factory = _DatasetFactory(case)
                with patch(case.patch_target, side_effect=factory):
                    fitted = case.dataset_type(batch_size=3, seed=0)
                    standalone = case.dataset_type(batch_size=3, seed=0)
                    fitted.setup("fit")
                    standalone.setup("validate")

                fitted_validation_transform = fitted.val.dataset.transform
                standalone_validation_transform = standalone.val.dataset.transform
                self.assertEqual(fitted.val.indices, standalone.val.indices)
                self.assertEqual(
                    _transform_signature(fitted_validation_transform),
                    _transform_signature(standalone_validation_transform),
                )
                self.assertNotIn(
                    RandomHorizontalFlip,
                    _transform_signature(fitted_validation_transform),
                )
                training_signature = _transform_signature(fitted.train.dataset.transform)
                self.assertEqual(
                    RandomHorizontalFlip in training_signature,
                    case.dataset_type is Cifar100,
                )

    def test_integer_resize_keeps_square_geometry_and_metadata(self) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                factory = _DatasetFactory(case)
                with patch(case.patch_target, side_effect=factory):
                    dataset = case.dataset_type(batch_size=2, resize=7)
                    dataset.setup("test")

                image, _ = dataset.test[0]
                expected_flattened_dim = case.channels * 7 * 7
                self.assertEqual(image.shape, torch.Size([case.channels, 7, 7]))
                self.assertEqual(dataset.default_height, 7)
                self.assertEqual(dataset.default_width, 7)
                self.assertEqual(dataset.flattened_input_dim, expected_flattened_dim)

    def test_evaluation_keeps_remainder_while_training_still_drops_it(self) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                factory = _DatasetFactory(case)
                with patch(case.patch_target, side_effect=factory):
                    dataset = case.dataset_type(batch_size=3, seed=0)
                    dataset.num_workers = 0
                    dataset.setup("fit")
                    dataset.setup("test")

                train_loader = dataset.train_dataloader()
                validation_loader = dataset.val_dataloader()
                test_loader = dataset.test_dataloader()
                self.assertTrue(train_loader.drop_last)
                self.assertFalse(validation_loader.drop_last)
                self.assertFalse(test_loader.drop_last)
                self.assertIsInstance(train_loader.sampler, RandomSampler)
                self.assertIsInstance(validation_loader.sampler, SequentialSampler)
                self.assertIsInstance(test_loader.sampler, SequentialSampler)
                self.assertEqual(
                    sum(batch[0].shape[0] for batch in validation_loader),
                    len(dataset.val),
                )
                self.assertEqual(
                    [batch[0].shape[0] for batch in test_loader],
                    [3, 2],
                )

    def test_seed_and_default_dataset_contracts_remain_stable(self) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                factory = _DatasetFactory(case)
                with patch(case.patch_target, side_effect=factory):
                    first = case.dataset_type(batch_size=2, seed=0)
                    second = case.dataset_type(batch_size=2, seed=0)
                    unseeded = case.dataset_type(batch_size=2, seed=None)
                    first.setup("fit")
                    second.setup("fit")

                self.assertEqual(first.val.indices, second.val.indices)
                self.assertEqual(first.seed, 0)
                self.assertIsNone(unseeded.seed)
                self.assertEqual(
                    (case.dataset_type.default_height, case.dataset_type.default_width),
                    case.default_size,
                )
                self.assertEqual(case.dataset_type.num_channels, case.channels)
                self.assertEqual(case.dataset_type.num_classes, case.num_classes)
                self.assertEqual(
                    first._text_labels([0, case.num_classes - 1]),
                    [case.first_label, case.last_label],
                )
                normalize = next(
                    operation
                    for operation in first.train.dataset.transform.transforms
                    if isinstance(operation, Normalize)
                )
                self.assertEqual(tuple(normalize.mean), case.normalization[0])
                self.assertEqual(tuple(normalize.std), case.normalization[1])


if __name__ == "__main__":
    unittest.main()

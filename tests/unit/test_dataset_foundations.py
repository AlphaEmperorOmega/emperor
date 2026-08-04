import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler, SequentialSampler

import emperor.datasets as datasets
from emperor.datasets._base import DataModule


class _ExampleDataModule(DataModule):
    batch_size = 2

    def __init__(self) -> None:
        super().__init__(root="example", num_workers=0)
        self.setup_calls: list[str] = []

    def _setup_fit(self) -> None:
        self.setup_calls.append("fit")
        self.train = "train"
        self.val = "validate"

    def _setup_validate(self) -> None:
        self.setup_calls.append("validate")
        self.val = "validate"

    def _setup_test(self) -> None:
        self.setup_calls.append("test")
        self.test = "test"

    def get_dataloader(self, train):
        return self.train if train else self.val

    def _get_test_dataloader(self):
        return self.test

    def _text_labels(self, indices) -> list[str]:
        return [str(index.item()) for index in indices]


class _ValidateAndTestDataModule(DataModule):
    def __init__(self) -> None:
        super().__init__(num_workers=0)
        self.setup_calls: list[str] = []

    def _setup_validate(self) -> None:
        self.setup_calls.append("validate")
        self.val = "validate"

    def _setup_test(self) -> None:
        self.setup_calls.append("test")
        self.test = "test"


class _TestOnlyDataModule(DataModule):
    def __init__(self) -> None:
        super().__init__(num_workers=0)
        self.setup_calls: list[str] = []

    def _setup_test(self) -> None:
        self.setup_calls.append("test")
        self.test = "test"


class DatasetFoundationTests(unittest.TestCase):
    def test_dataset_foundations_are_not_public_exports(self) -> None:
        self.assertFalse(hasattr(datasets, "DataModule"))
        self.assertFalse(hasattr(datasets, "show_images"))

    def test_setup_and_dataloader_dispatch_are_preserved(self) -> None:
        data = _ExampleDataModule()

        for stage in ("fit", "validate", "test"):
            data.setup(stage)

        self.assertEqual(data.setup_calls, ["fit", "validate", "test"])
        self.assertEqual(data.train_dataloader(), "train")
        self.assertEqual(data.val_dataloader(), "validate")
        self.assertEqual(data.test_dataloader(), "test")

    def test_setup_none_uses_validate_fallback_and_test_only_paths(self) -> None:
        validate_and_test = _ValidateAndTestDataModule()
        test_only = _TestOnlyDataModule()

        validate_and_test.setup(None)
        test_only.setup(None)

        self.assertEqual(validate_and_test.setup_calls, ["validate", "test"])
        self.assertEqual(test_only.setup_calls, ["test"])
        with self.assertRaisesRegex(
            NotImplementedError,
            "DataModule does not implement a dataset setup stage",
        ):
            DataModule().setup(None)
        with self.assertRaisesRegex(
            NotImplementedError,
            "_TestOnlyDataModule does not support validation data",
        ):
            test_only.val_dataloader()

    def test_unimplemented_base_hooks_raise_their_owned_errors(self) -> None:
        data = DataModule()

        for method_name, message in (
            ("_setup_fit", "'_setup_fit' must be implemented"),
            ("_setup_validate", "'_setup_validate' must be implemented"),
            ("_setup_test", "'_setup_test' must be implemented"),
            ("get_dataloader", None),
            ("_get_test_dataloader", "must implement '_get_test_dataloader'"),
            ("_text_labels", "'test_labels' method must be implemented"),
        ):
            with self.subTest(method=method_name):
                method = getattr(data, method_name)
                context = (
                    self.assertRaisesRegex(NotImplementedError, message)
                    if message is not None
                    else self.assertRaises(NotImplementedError)
                )
                with context:
                    if method_name == "get_dataloader":
                        method(False)
                    elif method_name == "_text_labels":
                        method([])
                    else:
                        method()

    def test_metadata_resolution_is_validated_frozen_and_atomic(self) -> None:
        data = _ExampleDataModule()
        data._resolve_metadata(
            vocab_size=3,
            num_classes=0,
            flattened_input_dim=4,
        )
        first = data.resolved_metadata

        data._resolve_metadata(vocab_size=5)

        resolved = data.resolved_metadata
        self.assertIsNot(resolved, first)
        self.assertEqual(resolved.vocab_size, 5)
        self.assertEqual(resolved.num_classes, 0)
        self.assertEqual(resolved.flattened_input_dim, 4)
        with self.assertRaises(FrozenInstanceError):
            resolved.vocab_size = 9

        invalid_dimensions = (
            ("vocab_size", 0, "positive"),
            ("vocab_size", True, "positive"),
            ("vocab_size", 1.5, "positive"),
            ("flattened_input_dim", -1, "positive"),
            ("num_classes", -1, "non-negative"),
            ("num_classes", False, "non-negative"),
            ("num_classes", "2", "non-negative"),
        )
        for field, value, qualifier in invalid_dimensions:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    f"Resolved dataset {field} must be a {qualifier} integer",
                ):
                    data._resolve_metadata(**{field: value})
                self.assertIs(data.resolved_metadata, resolved)

    def test_tensorloader_and_visualization_behavior_are_preserved(self) -> None:
        data = _ExampleDataModule()
        values = torch.arange(8).reshape(4, 2)
        labels = torch.arange(4)

        loader = data.get_tensorloader(
            (values, labels),
            train=False,
            indices=slice(1, 3),
        )
        loaded_values, loaded_labels = next(iter(loader))

        self.assertTrue(torch.equal(loaded_values, values[1:3]))
        self.assertTrue(torch.equal(loaded_labels, labels[1:3]))

        images = torch.zeros(2, 3, 3, 4)
        with patch("emperor.datasets._base.show_images") as render:
            data.visualize((images, torch.tensor([1, 2])), nrows=1, ncols=2)

        rendered_images, rows, columns = render.call_args.args[:3]
        self.assertEqual(rendered_images.shape, (2, 3, 4, 3))
        self.assertEqual((rows, columns), (1, 2))
        self.assertEqual(render.call_args.kwargs["titles"], ["1", "2"])

    def test_tensorloader_shuffles_only_training_and_never_drops_by_default(self):
        data = _ExampleDataModule()
        tensors = (torch.arange(8).reshape(4, 2), torch.arange(4))

        training_loader = data.get_tensorloader(tensors, train=True)
        validation_loader = data.get_tensorloader(tensors, train=False)

        self.assertIsInstance(training_loader.sampler, RandomSampler)
        self.assertIsInstance(validation_loader.sampler, SequentialSampler)
        self.assertFalse(training_loader.drop_last)
        self.assertFalse(validation_loader.drop_last)

    def test_visualization_keeps_explicit_labels_without_decoding(self) -> None:
        data = _ExampleDataModule()
        images = torch.zeros(1, 3, 2, 3)

        with patch("emperor.datasets._base.show_images") as render:
            data.visualize(
                (images, torch.tensor([1])),
                nrows=1,
                ncols=1,
                labels=["explicit"],
            )

        self.assertEqual(render.call_args.kwargs["titles"], ["explicit"])


if __name__ == "__main__":
    unittest.main()

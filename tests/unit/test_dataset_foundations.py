import unittest
from unittest.mock import patch

import torch

import emperor.datasets as datasets
from emperor.datasets._base import DataModule


class _ExampleDataModule(DataModule):
    batch_size = 2

    def __init__(self) -> None:
        super().__init__(root="example", num_workers=0)
        self.setup_calls: list[str] = []

    def _setup_fit(self) -> None:
        self.setup_calls.append("fit")

    def _setup_validate(self) -> None:
        self.setup_calls.append("validate")

    def _setup_test(self) -> None:
        self.setup_calls.append("test")

    def get_dataloader(self, train):
        return "train" if train else "validate"

    def _get_test_dataloader(self):
        return "test"

    def _text_labels(self, indices) -> list[str]:
        return [str(index.item()) for index in indices]


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


if __name__ == "__main__":
    unittest.main()

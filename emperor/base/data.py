import torch
from lightning import LightningDataModule

from emperor.base.visualization import show_images


class DataModule(LightningDataModule):
    """The base class of data."""

    def __init__(
        self,
        root="data",
        num_workers=4,
    ):
        super().__init__()
        self.root = root
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self._setup_fit()
            case "validate":
                self._setup_validate()
            case "test":
                self._setup_test()

    def _setup_fit(self) -> None:
        raise NotImplementedError(
            "The method '_setup_fit' must be implemented in the subclass."
        )

    def _setup_validate(self) -> None:
        raise NotImplementedError(
            "The method '_setup_validate' must be implemented in the subclass."
        )

    def _setup_test(self) -> None:
        raise NotImplementedError(
            "The method '_setup_test' must be implemented in the subclass."
        )

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def test_dataloader(self):
        return self._get_test_dataloader()

    def get_tensorloader(  # noqa: B008 - preserves the historical public default
        self,
        tensors,
        train,
        indices=slice(0, None),  # noqa: B008
    ):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    def visualize(self, batch, nrows=1, ncols=8, labels=None):
        X, y = batch
        if not labels:
            labels = self._text_labels(y)
        show_images(X.squeeze(1).permute(0, 2, 3, 1), nrows, ncols, titles=labels)

    def _text_labels(self, indices) -> list:
        raise NotImplementedError(
            "The 'test_labels' method must be implemented in the subclass."
        )


__all__ = ["DataModule"]

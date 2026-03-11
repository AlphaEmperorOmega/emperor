import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.transforms.transforms import Compose
from emperor.base.utils import DataModule


class FashionMNIST(DataModule):
    default_width: int = 28
    default_height: int = 28
    num_classes: int = 10
    num_channels: int = 1
    flattened_input_dim: int = default_width * default_height * num_channels

    def __init__(
        self,
        batch_size,
        resize=(28, 28),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize

    def prepare_data(self) -> None:
        datasets.FashionMNIST(root=self.root, train=True, download=True)
        datasets.FashionMNIST(root=self.root, train=False, download=True)

    val_split: float = 0.1

    def _setup_fit(self) -> None:
        full_train = datasets.FashionMNIST(
            root=self.root,
            train=True,
            transform=self.__get_train_transforms(),
        )
        val_size = int(len(full_train) * self.val_split)
        train_size = len(full_train) - val_size
        self.train, self.val = torch.utils.data.random_split(
            full_train, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    def _setup_validate(self) -> None:
        full_train = datasets.FashionMNIST(
            root=self.root,
            train=True,
            transform=self.__get_test_transforms(),
        )
        val_size = int(len(full_train) * self.val_split)
        train_size = len(full_train) - val_size
        _, self.val = torch.utils.data.random_split(
            full_train, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    def _setup_test(self) -> None:
        self.test = datasets.FashionMNIST(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
        )

    def __get_train_transforms(self) -> Compose:
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )

    def __get_test_transforms(self) -> Compose:
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )

    def _text_labels(self, indices) -> list:
        labels = [
            "t-shirt",
            "trouser",
            "pullover",
            "dress",
            "coat",
            "sandal",
            "shirt",
            "sneaker",
            "bag",
            "ankle boot",
        ]
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def _get_test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

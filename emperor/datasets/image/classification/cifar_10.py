import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.transforms.transforms import Compose
from emperor.base.utils import DataModule


class Cifar10(DataModule):
    default_width: int = 32
    default_height: int = 32
    num_classes: int = 10
    num_channels: int = 3
    flattened_input_dim: int = default_width * default_height * num_channels

    def __init__(
        self,
        batch_size=64,
        resize: tuple = (32, 32),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize
        width, height = resize
        self.flattened_input_dim: int = width * height * 3

    def prepare_data(self) -> None:
        datasets.CIFAR10(root=self.root, train=True, download=True)
        datasets.CIFAR10(root=self.root, train=False, download=True)

    def _setup_fit(self) -> None:
        self.train = datasets.CIFAR10(
            root=self.root,
            train=True,
            transform=self.__get_train_transforms(),
        )
        self.val = datasets.CIFAR10(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
        )

    def _setup_validate(self) -> None:
        self.val = datasets.CIFAR10(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
        )

    def __get_train_transforms(self) -> Compose:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __get_test_transforms(self) -> Compose:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _text_labels(self, indices) -> list:
        labels = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
        )

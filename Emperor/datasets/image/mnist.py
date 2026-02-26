import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Emperor.base.utils import DataModule
from torchvision.transforms.transforms import Compose


class Mnist(DataModule):
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
        datasets.MNIST(root=self.root, train=True, download=True)
        datasets.MNIST(root=self.root, train=False, download=True)

    def _setup_fit(self) -> None:
        self.train = datasets.MNIST(
            root=self.root,
            train=True,
            transform=self.__get_train_transforms(),
        )
        self.val = datasets.MNIST(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
        )

    def _setup_validate(self) -> None:
        self.val = datasets.MNIST(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
        )

    def __get_train_transforms(self) -> Compose:
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def __get_test_transforms(self) -> Compose:
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def _text_labels(self, indices) -> list:
        return [str(int(i)) for i in indices]

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
        )

import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.transforms.transforms import Compose
from Emperor.base.utils import DataModule


class Cifar10(DataModule):
    def __init__(
        self,
        batch_size=64,
        resize=(32, 32),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize

    def prepare_data(self) -> None:
        datasets.CIFAR10(root=self.root, train=True, download=True)
        datasets.CIFAR10(root=self.root, train=False, download=True)

    def setup(self, stage: str) -> None:
        train = datasets.CIFAR10(
            root=self.root,
            train=True,
            transform=self.__get_train_transforms(),
        )
        val = datasets.CIFAR10(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
        )
        self.train = self.get_dataset(train)
        self.val = self.get_dataset(val)

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
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
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

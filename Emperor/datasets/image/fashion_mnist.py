import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.transforms.transforms import Compose
from Emperor.base.utils import DataModule


class FashionMNIST(DataModule):
    def __init__(
        self,
        batch_size,
        resize=(28, 28),
        test_dataset_flag=False,
        test_dataset_num_samples=64,
    ):
        super().__init__(
            test_dataset_flag=test_dataset_flag,
            test_dataset_num_samples=test_dataset_num_samples,
        )
        self.batch_size = batch_size
        self.resize = resize

    def prepare_data(self) -> None:
        datasets.FashionMNIST(root=self.root, train=True, download=True)
        datasets.FashionMNIST(root=self.root, train=False, download=True)

    def setup(self, stage: str) -> None:
        train = datasets.FashionMNIST(
            root=self.root,
            train=True,
            transform=self.__get_train_transforms(),
        )
        val = datasets.FashionMNIST(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
        )
        self.train = self.get_dataset(train)
        self.val = self.get_dataset(val)

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
            "t-shirt", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle boot",
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

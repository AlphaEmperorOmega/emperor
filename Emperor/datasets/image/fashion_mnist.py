import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.transforms.transforms import Compose
from Emperor.base.utils import DataModule, show_images


class FashionMNIST(DataModule):
    def __init__(
        self,
        batch_size,
        resize=(28, 28),
        test_dataset_flag=False,
        test_dataset_num_samples=64,
    ):
        super().__init__()
        self.save_hyperparameters()
        train = datasets.FashionMNIST(
            root=self.root,
            train=True,
            transform=self.__get_train_transforms(),
            download=True,
        )
        val = datasets.FashionMNIST(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
            download=True,
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

    def _text_labels(self, indices):
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

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
        )

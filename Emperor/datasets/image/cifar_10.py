import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.transforms.transforms import Compose
from Emperor.base.utils import DataModule, show_images


class Cifar10(DataModule):
    def __init__(
        self,
        batch_size=64,
        resize=(32, 32),
    ):
        super().__init__()
        self.save_hyperparameters()

        train = datasets.CIFAR10(
            root=self.root,
            train=True,
            transform=self.__get_train_transforms(),
            download=True,
        )

        val = datasets.CIFAR10(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
            download=True,
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

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
        )

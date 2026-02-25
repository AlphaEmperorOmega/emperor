import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Emperor.base.utils import DataModule
from torchvision.transforms.transforms import Compose


class Mnist(DataModule):
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
        datasets.MNIST(root=self.root, train=True, download=True)
        datasets.MNIST(root=self.root, train=False, download=True)

    def setup(self, stage: str) -> None:
        train = datasets.MNIST(
            root=self.root,
            train=True,
            transform=self.__get_train_transforms(),
        )
        val = datasets.MNIST(
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

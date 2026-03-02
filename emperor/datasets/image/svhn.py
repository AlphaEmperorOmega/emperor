import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from emperor.base.utils import DataModule
from torchvision.transforms.transforms import Compose


class SVHN(DataModule):
    default_width: int = 32
    default_height: int = 32
    num_classes: int = 10
    num_channels: int = 3
    flattened_input_dim: int = default_width * default_height * num_channels

    def __init__(
        self,
        batch_size,
        resize=(32, 32),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize

    def prepare_data(self) -> None:
        datasets.SVHN(root=self.root, split="train", download=True)
        datasets.SVHN(root=self.root, split="test", download=True)

    def _setup_fit(self) -> None:
        self.train = datasets.SVHN(
            root=self.root,
            split="train",
            transform=self.__get_train_transforms(),
        )
        self.val = datasets.SVHN(
            root=self.root,
            split="test",
            transform=self.__get_test_transforms(),
        )

    def _setup_validate(self) -> None:
        self.val = datasets.SVHN(
            root=self.root,
            split="test",
            transform=self.__get_test_transforms(),
        )

    def __get_train_transforms(self) -> Compose:
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __get_test_transforms(self) -> Compose:
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _text_labels(self, indices):
        return [str(int(i)) for i in indices]

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
        )

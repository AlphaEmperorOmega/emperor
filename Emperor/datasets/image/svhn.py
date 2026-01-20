import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Emperor.base.utils import DataModule
from torchvision.transforms.transforms import Compose


class SVHN(DataModule):
    def __init__(
        self,
        batch_size,
        resize=(32, 32),
        test_dataset_flag=False,
        test_dataset_num_samples=64,
    ):
        super().__init__()
        self.save_hyperparameters()

        train = datasets.SVHN(
            root=self.root,
            split="train",
            transform=self.__get_train_transforms(),
            download=True,
        )

        val = datasets.SVHN(
            root=self.root,
            split="test",
            transform=self.__get_test_transforms(),
            download=True,
        )

        self.train = self.get_dataset(train)
        self.val = self.get_dataset(val)

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

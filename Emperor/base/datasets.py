# EXTERNAL
import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# LOCAL
from .utils import DataModule, show_images


class FashionMNIST(DataModule):
    """
    The Fashion-MNIST dataset.
    """

    def __init__(
        self,
        batch_size,
        resize=(28, 28),
        testDatasetFalg=False,
        testDatasetNumSamples=512,
    ):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        train = datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True
        )
        val = datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True
        )

        self.train = self.getDataset(train)
        self.val = self.getDataset(val)

    def text_labels(self, indices):
        """Return text labels."""
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

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), nrows, ncols, titles=labels)


class Cifar10(DataModule):
    """
    The Cifar10 Dataset
    """

    def __init__(self, batch_size=64, resize=(32, 32)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])

        self.train = datasets.CIFAR10(
            root=self.root, train=True, transform=trans, download=True
        )

        self.val = datasets.CIFAR10(
            root=self.root, train=False, transform=trans, download=True
        )

    def text_labels(self, indices):
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

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1).permute(0, 2, 3, 1), nrows, ncols, titles=labels)

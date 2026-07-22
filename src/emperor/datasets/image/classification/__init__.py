"""Public Interface for supported image-classification datasets."""

from emperor.datasets.image.classification._cifar_10 import Cifar10
from emperor.datasets.image.classification._cifar_100 import Cifar100
from emperor.datasets.image.classification._fashion_mnist import FashionMNIST
from emperor.datasets.image.classification._mnist import Mnist

__all__ = ("Mnist", "FashionMNIST", "Cifar10", "Cifar100")

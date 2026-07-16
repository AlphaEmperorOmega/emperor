"""Public Interface for supported image-classification datasets."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.datasets.image.classification._cifar_10 import Cifar10
    from emperor.datasets.image.classification._cifar_100 import Cifar100
    from emperor.datasets.image.classification._fashion_mnist import FashionMNIST
    from emperor.datasets.image.classification._mnist import Mnist

__all__ = ("Mnist", "FashionMNIST", "Cifar10", "Cifar100")

_LAZY_EXPORTS = {
    "Mnist": ("emperor.datasets.image.classification._mnist", "Mnist"),
    "FashionMNIST": (
        "emperor.datasets.image.classification._fashion_mnist",
        "FashionMNIST",
    ),
    "Cifar10": ("emperor.datasets.image.classification._cifar_10", "Cifar10"),
    "Cifar100": ("emperor.datasets.image.classification._cifar_100", "Cifar100"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from emperor.base.utils import DataModule
from torchvision.transforms.transforms import Compose


class Cifar100(DataModule):
    default_width: int = 32
    default_height: int = 32
    num_classes: int = 100
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
        datasets.CIFAR100(root=self.root, train=True, download=True)
        datasets.CIFAR100(root=self.root, train=False, download=True)

    def _setup_fit(self) -> None:
        self.train = datasets.CIFAR100(
            root=self.root,
            train=True,
            transform=self.__get_train_transforms(),
        )
        self.val = datasets.CIFAR100(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
        )

    def _setup_validate(self) -> None:
        self.val = datasets.CIFAR100(
            root=self.root,
            train=False,
            transform=self.__get_test_transforms(),
        )

    def __get_train_transforms(self) -> Compose:
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4865, 0.4409),
                    std=(0.2673, 0.2564, 0.2761),
                ),
            ]
        )

    def __get_test_transforms(self) -> Compose:
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4865, 0.4409),
                    std=(0.2673, 0.2564, 0.2761),
                ),
            ]
        )

    def _text_labels(self, indices) -> list:
        labels = [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
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

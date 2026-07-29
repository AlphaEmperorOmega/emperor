import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Compose

from emperor.datasets._base import DataModule
from emperor.datasets.image.segmentation._geometry import _SegmentationGeometry


class _SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, geometry):
        self.dataset = dataset
        self.geometry = geometry

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        return self.geometry.transform(image, mask)


class VOCSegmentation(DataModule):
    default_width: int = 256
    default_height: int = 256
    num_channels: int = 3
    flattened_input_dim: int = default_width * default_height * num_channels
    num_classes: int = 21  # 20 object classes + background

    def __init__(
        self,
        batch_size: int = 16,
        resize: tuple = (256, 256),
        year: str = "2012",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize
        self.year = year

    def prepare_data(self) -> None:
        datasets.VOCSegmentation(
            root=self.root, year=self.year, image_set="train", download=True
        )
        datasets.VOCSegmentation(
            root=self.root, year=self.year, image_set="val", download=True
        )

    def _setup_fit(self) -> None:
        self.train = _SegmentationDataset(
            datasets.VOCSegmentation(root=self.root, year=self.year, image_set="train"),
            _SegmentationGeometry(self._get_image_transforms()),
        )
        self.val = _SegmentationDataset(
            datasets.VOCSegmentation(root=self.root, year=self.year, image_set="val"),
            _SegmentationGeometry(self._get_image_transforms()),
        )

    def _setup_validate(self) -> None:
        self.val = _SegmentationDataset(
            datasets.VOCSegmentation(root=self.root, year=self.year, image_set="val"),
            _SegmentationGeometry(self._get_image_transforms()),
        )

    def _get_image_transforms(self) -> Compose:
        return transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def _text_labels(self, indices) -> list:
        labels = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        return [labels[int(i)] for i in indices]

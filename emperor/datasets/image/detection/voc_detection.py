import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.transforms.transforms import Compose

from emperor.base.utils import DataModule


_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]
_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(_CLASSES)}


def _collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class _DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_transform):
        self.dataset = dataset
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, annotation = self.dataset[idx]
        orig_w = annotation["annotation"]["size"]["width"]
        orig_h = annotation["annotation"]["size"]["height"]
        image = self.image_transform(image)
        _, new_h, new_w = image.shape
        target = self._build_target(annotation, orig_w, orig_h, new_w, new_h)
        return image, target

    def _build_target(self, annotation: dict, orig_w, orig_h, new_w, new_h) -> dict:
        boxes, labels = [], []
        objects = annotation["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        scale_x = int(new_w) / int(orig_w)
        scale_y = int(new_h) / int(orig_h)
        for obj in objects:
            label = _CLASS_TO_IDX.get(obj["name"], 0)
            bb = obj["bndbox"]
            x1 = float(bb["xmin"]) * scale_x
            y1 = float(bb["ymin"]) * scale_y
            x2 = float(bb["xmax"]) * scale_x
            y2 = float(bb["ymax"]) * scale_y
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
        if boxes:
            return {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.long),
        }


class VOCDetection(DataModule):
    default_width: int = 416
    default_height: int = 416
    num_channels: int = 3
    flattened_input_dim: int = default_width * default_height * num_channels
    num_classes: int = 21  # 20 object classes + background

    def __init__(
        self,
        batch_size: int = 16,
        resize: tuple = (416, 416),
        year: str = "2012",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize
        self.year = year

    def prepare_data(self) -> None:
        datasets.VOCDetection(root=self.root, year=self.year, image_set="train", download=True)
        datasets.VOCDetection(root=self.root, year=self.year, image_set="val", download=True)

    def _setup_fit(self) -> None:
        self.train = _DetectionDataset(
            datasets.VOCDetection(root=self.root, year=self.year, image_set="train"),
            self._get_image_transforms(),
        )
        self.val = _DetectionDataset(
            datasets.VOCDetection(root=self.root, year=self.year, image_set="val"),
            self._get_image_transforms(),
        )

    def _setup_validate(self) -> None:
        self.val = _DetectionDataset(
            datasets.VOCDetection(root=self.root, year=self.year, image_set="val"),
            self._get_image_transforms(),
        )

    def _get_image_transforms(self) -> Compose:
        return transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=_collate_fn,
        )

    def _text_labels(self, indices) -> list:
        return [_CLASSES[int(i)] for i in indices]

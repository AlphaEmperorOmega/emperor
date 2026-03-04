import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.transforms.transforms import Compose

from emperor.base.utils import DataModule


def _collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class _DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_transform):
        self.dataset = dataset
        self.image_transform = image_transform
        self.cat_ids = sorted(dataset.coco.getCatIds())
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}

    def __len__(self):
        return len(self.dataset.ids)

    def __getitem__(self, idx):
        image, annotations = self.dataset[idx]
        image = self.image_transform(image)
        target = self._build_target(annotations)
        return image, target

    def _build_target(self, annotations: list) -> dict:
        boxes, labels = [], []
        for ann in annotations:
            if "bbox" not in ann:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label.get(ann["category_id"], 0))
        if boxes:
            return {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.long),
        }


class CocoDetection(DataModule):
    default_width: int = 640
    default_height: int = 640
    num_channels: int = 3
    flattened_input_dim: int = default_width * default_height * num_channels
    num_classes: int = 81  # 80 object classes + background

    def __init__(
        self,
        batch_size: int = 8,
        resize: tuple = (640, 640),
        train_ann_file: str = "data/coco/annotations/instances_train2017.json",
        val_ann_file: str = "data/coco/annotations/instances_val2017.json",
        train_root: str = "data/coco/train2017",
        val_root: str = "data/coco/val2017",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize
        self.train_ann_file = train_ann_file
        self.val_ann_file = val_ann_file
        self.train_root = train_root
        self.val_root = val_root

    def prepare_data(self) -> None:
        pass  # COCO requires manual download

    def _setup_fit(self) -> None:
        self.train = _DetectionDataset(
            datasets.CocoDetection(root=self.train_root, annFile=self.train_ann_file),
            self._get_image_transforms(),
        )
        self.val = _DetectionDataset(
            datasets.CocoDetection(root=self.val_root, annFile=self.val_ann_file),
            self._get_image_transforms(),
        )

    def _setup_validate(self) -> None:
        self.val = _DetectionDataset(
            datasets.CocoDetection(root=self.val_root, annFile=self.val_ann_file),
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
        labels = [
            "background", "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
            "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "couch", "potted plant", "bed",
            "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush",
        ]
        return [labels[int(i)] for i in indices]

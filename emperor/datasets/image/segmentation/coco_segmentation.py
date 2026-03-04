import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

from torchvision.transforms.transforms import Compose

from emperor.base.utils import DataModule


class _SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_transform, mask_size):
        self.dataset = dataset
        self.image_transform = image_transform
        self.mask_size = mask_size
        self.cat_ids = sorted(dataset.coco.getCatIds())
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}

    def __len__(self):
        return len(self.dataset.ids)

    def __getitem__(self, idx):
        image, annotations = self.dataset[idx]
        image = self.image_transform(image)
        mask = self._build_mask(annotations)
        return image, mask

    def _build_mask(self, annotations: list) -> torch.Tensor:
        mask = np.zeros(self.mask_size, dtype=np.int64)
        for ann in annotations:
            if "segmentation" not in ann:
                continue
            cat_label = self.cat_id_to_label.get(ann["category_id"], 0)
            seg_mask = self.dataset.coco.annToMask(ann)
            seg_mask = torch.as_tensor(seg_mask, dtype=torch.uint8).numpy()
            mask[seg_mask > 0] = cat_label
        return torch.as_tensor(mask, dtype=torch.long)


class CocoSegmentation(DataModule):
    default_width: int = 256
    default_height: int = 256
    num_channels: int = 3
    flattened_input_dim: int = default_width * default_height * num_channels
    num_classes: int = 81  # 80 object classes + background

    def __init__(
        self,
        batch_size: int = 16,
        resize: tuple = (256, 256),
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
        self.train = _SegmentationDataset(
            datasets.CocoDetection(root=self.train_root, annFile=self.train_ann_file),
            self._get_image_transforms(),
            self.resize,
        )
        self.val = _SegmentationDataset(
            datasets.CocoDetection(root=self.val_root, annFile=self.val_ann_file),
            self._get_image_transforms(),
            self.resize,
        )

    def _setup_validate(self) -> None:
        self.val = _SegmentationDataset(
            datasets.CocoDetection(root=self.val_root, annFile=self.val_ann_file),
            self._get_image_transforms(),
            self.resize,
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

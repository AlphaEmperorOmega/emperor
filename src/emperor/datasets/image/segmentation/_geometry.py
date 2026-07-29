import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as functional_transforms


class _SegmentationGeometry:
    def __init__(self, image_transform) -> None:
        self._image_transform = image_transform

    def transform(self, image, mask) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._image_transform(image)
        target_height, target_width = image.shape[-2:]
        mask = functional_transforms.resize(
            mask,
            [target_height, target_width],
            interpolation=InterpolationMode.NEAREST,
        )
        mask = functional_transforms.pil_to_tensor(mask)
        if mask.shape[0] != 1:
            raise ValueError("segmentation masks must contain exactly one channel")
        return image, mask.squeeze(0).to(dtype=torch.long)

import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from emperor.datasets.image.detection._coco import (
    CocoDetection,
)
from emperor.datasets.image.detection._coco import (
    _collate_fn as coco_collate,
)
from emperor.datasets.image.detection._voc import VOCDetection
from emperor.datasets.image.segmentation._cityscapes import Cityscapes
from emperor.datasets.image.segmentation._coco import CocoSegmentation
from emperor.datasets.image.segmentation._voc import VOCSegmentation


def _image(width: int = 6, height: int = 4) -> Image.Image:
    return Image.new("RGB", (width, height), color=(64, 128, 192))


class _FakeCocoApi:
    def __init__(self, category_ids: list[int]) -> None:
        self._category_ids = category_ids

    def getCatIds(self) -> list[int]:
        return self._category_ids

    def annToMask(self, annotation: dict) -> np.ndarray:
        return np.asarray(annotation["segmentation_mask"], dtype=np.uint8)


class _FakeCocoDataset:
    def __init__(
        self,
        image: Image.Image,
        annotations: list[dict],
        *,
        category_ids: list[int],
    ) -> None:
        self.ids = [1]
        self.coco = _FakeCocoApi(category_ids)
        self._image = image
        self._annotations = annotations

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self._image.copy(), self._annotations


class _FakePairDataset:
    def __init__(self, image: Image.Image, target) -> None:
        self._image = image
        self._target = target

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        image = self._image.copy()
        target = self._target.copy() if hasattr(self._target, "copy") else self._target
        return image, target


class TestDetectionSpatialTargets(unittest.TestCase):
    def test_coco_boxes_scale_with_non_square_image_resize(self) -> None:
        dataset = _FakeCocoDataset(
            _image(width=10, height=20),
            [
                {"bbox": [1, 5, 3, 10], "category_id": 7},
                {"bbox": [0, 0, 0, 4], "category_id": 7},
                {"category_id": 7},
            ],
            category_ids=[7],
        )
        data = CocoDetection(resize=(8, 20))

        with patch(
            "emperor.datasets.image.detection._coco.datasets.CocoDetection",
            return_value=dataset,
        ):
            data._setup_validate()

        image, target = data.val[0]
        self.assertEqual(image.shape, torch.Size([3, 8, 20]))
        torch.testing.assert_close(
            target["boxes"],
            torch.tensor([[2.0, 2.0, 8.0, 6.0]]),
        )
        torch.testing.assert_close(target["labels"], torch.tensor([1]))
        self.assertEqual(target["boxes"].dtype, torch.float32)
        self.assertEqual(target["labels"].dtype, torch.long)

    def test_coco_empty_target_keeps_detection_schema(self) -> None:
        dataset = _FakeCocoDataset(
            _image(width=10, height=20),
            [{"bbox": [1, 2, -1, 3], "category_id": 7}],
            category_ids=[7],
        )
        data = CocoDetection(resize=(8, 20))

        with patch(
            "emperor.datasets.image.detection._coco.datasets.CocoDetection",
            return_value=dataset,
        ):
            data._setup_validate()

        _, target = data.val[0]
        self.assertEqual(target["boxes"].shape, torch.Size([0, 4]))
        self.assertEqual(target["labels"].shape, torch.Size([0]))
        self.assertEqual(target["boxes"].dtype, torch.float32)
        self.assertEqual(target["labels"].dtype, torch.long)

    def test_voc_boxes_retain_existing_non_square_scaling(self) -> None:
        annotation = {
            "annotation": {
                "size": {"width": "10", "height": "20"},
                "object": {
                    "name": "cat",
                    "bndbox": {
                        "xmin": "1",
                        "ymin": "5",
                        "xmax": "4",
                        "ymax": "15",
                    },
                },
            }
        }
        dataset = _FakePairDataset(_image(width=10, height=20), annotation)
        data = VOCDetection(resize=(8, 20))

        with patch(
            "emperor.datasets.image.detection._voc.datasets.VOCDetection",
            return_value=dataset,
        ):
            data._setup_validate()

        image, target = data.val[0]
        self.assertEqual(image.shape, torch.Size([3, 8, 20]))
        torch.testing.assert_close(
            target["boxes"],
            torch.tensor([[2.0, 2.0, 8.0, 6.0]]),
        )
        torch.testing.assert_close(target["labels"], torch.tensor([8]))

    def test_detection_collation_preserves_per_sample_lists(self) -> None:
        first = (torch.zeros(3, 2, 2), {"boxes": torch.zeros(0, 4)})
        second = (torch.ones(3, 2, 2), {"boxes": torch.ones(1, 4)})

        images, targets = coco_collate([first, second])

        self.assertIsInstance(images, list)
        self.assertIsInstance(targets, list)
        self.assertIs(images[0], first[0])
        self.assertIs(targets[1], second[1])


class TestSegmentationSpatialTargets(unittest.TestCase):
    def test_coco_mask_resizes_from_source_geometry_with_nearest_labels(self) -> None:
        first_mask = np.zeros((4, 6), dtype=np.uint8)
        first_mask[:, :2] = 1
        second_mask = np.zeros((4, 6), dtype=np.uint8)
        second_mask[2:, 4:] = 1
        dataset = _FakeCocoDataset(
            _image(),
            [
                {
                    "category_id": 7,
                    "segmentation": [[0, 0, 1, 0, 1, 3]],
                    "segmentation_mask": first_mask,
                },
                {
                    "category_id": 9,
                    "segmentation": [[4, 2, 5, 2, 5, 3]],
                    "segmentation_mask": second_mask,
                },
            ],
            category_ids=[7, 9],
        )
        data = CocoSegmentation(resize=(2, 3))

        with patch(
            "emperor.datasets.image.segmentation._coco.datasets.CocoDetection",
            return_value=dataset,
        ):
            data._setup_validate()

        image, mask = data.val[0]
        self.assertEqual(image.shape, torch.Size([3, 2, 3]))
        self.assertEqual(mask.shape, torch.Size([2, 3]))
        self.assertEqual(mask.dtype, torch.long)
        torch.testing.assert_close(mask, torch.tensor([[1, 0, 0], [1, 0, 2]]))

    def test_voc_mask_is_rank_two_and_preserves_ignore_label(self) -> None:
        source_mask = Image.fromarray(
            np.array(
                [
                    [0, 0, 1, 1, 2, 2],
                    [0, 0, 1, 1, 2, 2],
                    [3, 3, 4, 4, 255, 255],
                    [3, 3, 4, 4, 255, 255],
                ],
                dtype=np.uint8,
            )
        )
        dataset = _FakePairDataset(_image(), source_mask)
        data = VOCSegmentation(resize=(2, 3))

        with patch(
            "emperor.datasets.image.segmentation._voc.datasets.VOCSegmentation",
            return_value=dataset,
        ):
            data._setup_validate()

        image, mask = data.val[0]
        self.assertEqual(image.shape, torch.Size([3, 2, 3]))
        self.assertEqual(mask.shape, torch.Size([2, 3]))
        self.assertEqual(mask.dtype, torch.long)
        torch.testing.assert_close(mask, torch.tensor([[0, 1, 2], [3, 4, 255]]))

    def test_cityscapes_maps_source_ids_to_train_ids_and_ignore(self) -> None:
        source_mask = Image.fromarray(
            np.array(
                [
                    [7, 8, 11],
                    [0, 33, 29],
                ],
                dtype=np.uint8,
            )
        )
        dataset = _FakePairDataset(_image(width=3, height=2), source_mask)
        data = Cityscapes(resize=(2, 3))

        with patch(
            "emperor.datasets.image.segmentation._cityscapes.datasets.Cityscapes",
            return_value=dataset,
        ):
            data._setup_validate()

        image, mask = data.val[0]
        self.assertEqual(image.shape, torch.Size([3, 2, 3]))
        self.assertEqual(mask.shape, torch.Size([2, 3]))
        self.assertEqual(mask.dtype, torch.long)
        torch.testing.assert_close(
            mask,
            torch.tensor([[0, 1, 2], [255, 18, 255]]),
        )


if __name__ == "__main__":
    unittest.main()

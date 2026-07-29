from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from emperor.datasets.image.captioning._coco_captions import CocoCaptions
from emperor.datasets.image.captioning._flickr8k import Flickr8k
from emperor.datasets.image.captioning._flickr30k import Flickr30k


@dataclass(frozen=True)
class _CaptioningCase:
    dataset_type: type
    patch_target: str


_CASES = (
    _CaptioningCase(
        CocoCaptions,
        "emperor.datasets.image.captioning._coco_captions.datasets.CocoCaptions",
    ),
    _CaptioningCase(
        Flickr30k,
        "emperor.datasets.image.captioning._flickr30k.datasets.Flickr30k",
    ),
    _CaptioningCase(
        Flickr8k,
        "emperor.datasets.image.captioning._flickr8k.datasets.Flickr8k",
    ),
)


class _Vocabulary:
    def __init__(self, token_sequences, *, specials) -> None:
        tokens = {token for sequence in token_sequences for token in sequence}
        self._tokens = [*specials, *sorted(tokens.difference(specials))]
        self._indices = {token: index for index, token in enumerate(self._tokens)}
        self.default_index: int | None = None

    def __call__(self, tokens: list[str]) -> list[int]:
        unknown_index = self._indices["<unk>"]
        return [self._indices.get(token, unknown_index) for token in tokens]

    def __getitem__(self, token: str) -> int:
        return self._indices[token]

    def __len__(self) -> int:
        return len(self._tokens)

    def lookup_token(self, index: int) -> str:
        return self._tokens[index]

    def set_default_index(self, index: int) -> None:
        self.default_index = index


def _build_vocabulary(token_sequences, *, specials):
    return _Vocabulary(token_sequences, specials=specials)


class _OfflineCaptionSource:
    def __init__(self, transform, split: str) -> None:
        self.transform = transform
        self.split = split
        self.samples = (
            ("alpha beta", "gamma delta"),
            ("beta gamma", "delta alpha"),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image = Image.new("RGB", (10, 12), color=(index, 64, 128))
        if self.transform is not None:
            image = self.transform(image)
        return image, list(self.samples[index])


class _CaptionSourceFactory:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *_args, transform=None, **kwargs):
        self.calls.append(kwargs)
        annotation = kwargs.get("annFile", kwargs.get("ann_file", ""))
        split = "train" if "train" in str(annotation) else "validation"
        return _OfflineCaptionSource(transform, split)


class CaptioningDatasetTests(unittest.TestCase):
    def test_manual_prepare_is_offline_and_fit_exposes_canonical_batches(self) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                factory = _CaptionSourceFactory()
                dataset = case.dataset_type(
                    batch_size=2,
                    sequence_length=5,
                    resize=(6, 8),
                )
                dataset.num_workers = 0

                module_name = case.patch_target.rsplit(".datasets", 1)[0]

                with (
                    patch(case.patch_target, side_effect=factory),
                    patch(
                        f"{module_name}.build_vocab_from_iterator",
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset.prepare_data()
                    self.assertEqual(factory.calls, [])
                    dataset.setup("fit")

                self.assertEqual(len(factory.calls), 2)
                self.assertIsInstance(dataset.train_dataloader().sampler, RandomSampler)
                self.assertIsInstance(
                    dataset.val_dataloader().sampler,
                    SequentialSampler,
                )
                images, captions = next(iter(dataset.val_dataloader()))
                self.assertEqual(images.shape, torch.Size([2, 3, 6, 8]))
                self.assertEqual(images.dtype, torch.float32)
                self.assertEqual(captions.shape, torch.Size([2, 5]))
                self.assertEqual(captions.dtype, torch.long)
                self.assertEqual(
                    dataset._text_labels(captions[0, :2]),
                    ["alpha", "beta"],
                )
                self.assertEqual(
                    dataset.resolved_metadata.vocab_size,
                    len(dataset.vocab),
                )
                self.assertEqual(
                    dataset.resolved_metadata.num_classes,
                    len(dataset.vocab),
                )

    def test_training_selects_one_caption_and_validation_uses_the_first(self) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                factory = _CaptionSourceFactory()
                dataset = case.dataset_type(sequence_length=4)
                module_name = case.patch_target.rsplit(".datasets", 1)[0]
                with (
                    patch(case.patch_target, side_effect=factory),
                    patch(
                        f"{module_name}.build_vocab_from_iterator",
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset.setup("fit")

                with patch(
                    f"{module_name}.random.choice",
                    side_effect=lambda values: values[-1],
                ):
                    _, training_caption = dataset.train[0]
                _, validation_caption = dataset.val[0]

                self.assertEqual(
                    dataset._text_labels(training_caption[:2]),
                    ["gamma", "delta"],
                )
                self.assertEqual(
                    dataset._text_labels(validation_caption[:2]),
                    ["alpha", "beta"],
                )

    def test_validation_only_builds_a_local_schema_and_source_errors_propagate(
        self,
    ) -> None:
        for case in _CASES:
            with self.subTest(dataset=case.dataset_type.__name__):
                factory = _CaptionSourceFactory()
                dataset = case.dataset_type(batch_size=1, sequence_length=4)
                dataset.num_workers = 0
                module_name = case.patch_target.rsplit(".datasets", 1)[0]
                with (
                    patch(case.patch_target, side_effect=factory),
                    patch(
                        f"{module_name}.build_vocab_from_iterator",
                        side_effect=_build_vocabulary,
                    ),
                ):
                    dataset.setup("validate")

                self.assertEqual(len(factory.calls), 1)
                self.assertEqual(len(dataset.val_dataloader()), 2)
                self.assertIsNotNone(dataset.vocab)

                unavailable = case.dataset_type()
                with (
                    patch(case.patch_target, side_effect=FileNotFoundError("missing")),
                    self.assertRaisesRegex(FileNotFoundError, "missing"),
                ):
                    unavailable.setup("validate")


if __name__ == "__main__":
    unittest.main()

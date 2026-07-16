from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from emperor.monitoring._history import MonitorTensorHistory

DEFAULT_HISTOGRAM_MAX_ELEMENTS = 100_000
DEFAULT_IMAGE_MAX_RAW_BYTES = 1_000_000
DEFAULT_IMAGE_MAX_SIDE = 1_024


def _step_value(step: int | None = None, global_step: int | None = None) -> int:
    if global_step is not None:
        return int(global_step)
    return int(step or 0)


def _default_module_key(tag: str) -> str:
    return tag.partition("/")[0]


@dataclass
class MonitorEmissionPolicy:
    histogram_max_elements: int = DEFAULT_HISTOGRAM_MAX_ELEMENTS
    image_max_raw_bytes: int = DEFAULT_IMAGE_MAX_RAW_BYTES
    image_max_side: int = DEFAULT_IMAGE_MAX_SIDE
    media_every_n_steps: int = 1
    emitted_max_entries: int = 10_000
    _emitted: dict[tuple[str, str, str, int], None] = field(default_factory=dict)

    def clear(self) -> None:
        self._emitted.clear()

    def should_emit_media(
        self, step: int | None = None, global_step: int | None = None
    ) -> bool:
        cadence = max(1, int(self.media_every_n_steps))
        return _step_value(step, global_step) % cadence == 0

    def emit_histogram(
        self,
        experiment: object,
        tag: str,
        values: torch.Tensor,
        step: int,
        *,
        module_key: str | None = None,
    ) -> bool:
        add_histogram = getattr(experiment, "add_histogram", None)
        if not callable(add_histogram) or not self._claim_emission(
            "histogram",
            tag,
            step,
            module_key,
        ):
            return False
        add_histogram(tag, self._bounded_histogram(values), int(step))
        return True

    def emit_image(  # noqa: PLR0913
        self,
        experiment: object,
        tag: str,
        image: torch.Tensor,
        step: int | None = None,
        *,
        global_step: int | None = None,
        dataformats: str | None = None,
        module_key: str | None = None,
    ) -> bool:
        resolved_step = _step_value(step, global_step)
        add_image = getattr(experiment, "add_image", None)
        if (
            not callable(add_image)
            or not self.should_emit_media(resolved_step)
            or not self._claim_emission("image", tag, resolved_step, module_key)
        ):
            return False
        image = self._bounded_image(image, dataformats=dataformats)
        if dataformats is None:
            add_image(tag, image, global_step=resolved_step)
        else:
            add_image(tag, image, resolved_step, dataformats=dataformats)
        return True

    def emit_history_heatmap(  # noqa: PLR0913
        self,
        experiment: object,
        tag: str,
        history: MonitorTensorHistory,
        step: int | None = None,
        *,
        global_step: int | None = None,
        module_key: str | None = None,
    ) -> bool:
        heatmap = history.render_heatmap()
        if heatmap is None:
            return False
        return self.emit_image(
            experiment,
            tag,
            heatmap,
            step,
            global_step=global_step,
            dataformats="CHW",
            module_key=module_key,
        )

    def _claim_emission(
        self,
        kind: str,
        tag: str,
        step: int,
        module_key: str | None,
    ) -> bool:
        key = (kind, module_key or _default_module_key(tag), tag, int(step))
        if key in self._emitted:
            return False
        self._emitted[key] = None
        while len(self._emitted) > self.emitted_max_entries:
            self._emitted.pop(next(iter(self._emitted)))
        return True

    def _bounded_histogram(self, values: torch.Tensor) -> torch.Tensor:
        tensor = values.detach().float().reshape(-1).cpu()
        max_elements = max(1, int(self.histogram_max_elements))
        if tensor.numel() <= max_elements:
            return tensor
        indices = torch.linspace(
            0,
            tensor.numel() - 1,
            max_elements,
            dtype=torch.long,
            device=tensor.device,
        )
        return tensor.index_select(0, indices)

    def _bounded_image(
        self,
        image: torch.Tensor,
        *,
        dataformats: str | None,
    ) -> torch.Tensor:
        tensor = image.detach().float().cpu()
        formats = dataformats or self._infer_dataformats(tensor)
        tensor = self._cap_image_side(tensor, formats)
        while tensor.numel() * tensor.element_size() > self.image_max_raw_bytes:
            next_tensor = self._scale_image(tensor, formats, 0.75)
            if next_tensor.shape == tensor.shape:
                break
            tensor = next_tensor
        return tensor

    def _infer_dataformats(self, image: torch.Tensor) -> str:
        if image.dim() == 2:
            return "HW"
        if image.dim() == 3:
            return "CHW"
        if image.dim() == 4:
            return "NCHW"
        return ""

    def _cap_image_side(self, image: torch.Tensor, dataformats: str) -> torch.Tensor:
        height, width = self._image_hw(image, dataformats)
        max_side = max(1, int(self.image_max_side))
        if height <= max_side and width <= max_side:
            return image
        scale = min(max_side / max(height, 1), max_side / max(width, 1))
        return self._scale_image(image, dataformats, scale)

    def _image_hw(self, image: torch.Tensor, dataformats: str) -> tuple[int, int]:
        if dataformats.endswith("CHW") and image.dim() >= 3:
            return int(image.shape[-2]), int(image.shape[-1])
        if dataformats.endswith("HWC") and image.dim() >= 3:
            return int(image.shape[-3]), int(image.shape[-2])
        if dataformats == "HW" and image.dim() == 2:
            return int(image.shape[0]), int(image.shape[1])
        if image.dim() >= 2:
            return int(image.shape[-2]), int(image.shape[-1])
        return 1, int(image.numel())

    def _scale_image(
        self,
        image: torch.Tensor,
        dataformats: str,
        scale: float,
    ) -> torch.Tensor:
        height, width = self._image_hw(image, dataformats)
        next_height = max(1, int(height * scale))
        next_width = max(1, int(width * scale))
        if next_height == height and next_width == width:
            return image
        nchw_image, restore = self._image_as_nchw(image, dataformats)
        if nchw_image is None:
            return image
        scaled = F.interpolate(
            nchw_image,
            size=(next_height, next_width),
            mode="bilinear",
            align_corners=False,
        )
        return restore(scaled)

    def _image_as_nchw(
        self,
        image: torch.Tensor,
        dataformats: str,
    ) -> tuple[torch.Tensor | None, Callable[[torch.Tensor], torch.Tensor]]:
        if dataformats.endswith("CHW") and image.dim() == 3:
            return image.unsqueeze(0), lambda scaled: scaled.squeeze(0)
        if dataformats.endswith("NCHW") and image.dim() == 4:
            return image, lambda scaled: scaled
        if dataformats.endswith("HWC") and image.dim() == 3:
            return (
                image.permute(2, 0, 1).unsqueeze(0),
                lambda scaled: scaled.squeeze(0).permute(1, 2, 0),
            )
        if dataformats.endswith("NHWC") and image.dim() == 4:
            return (
                image.permute(0, 3, 1, 2),
                lambda scaled: scaled.permute(0, 2, 3, 1),
            )
        if image.dim() == 2:
            return (
                image.unsqueeze(0).unsqueeze(0),
                lambda scaled: scaled.squeeze(0).squeeze(0),
            )
        return None, lambda scaled: scaled

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
        media_cadence_steps = max(1, int(self.media_every_n_steps))
        resolved_step = _step_value(step, global_step)
        return resolved_step % media_cadence_steps == 0

    def emit_histogram(
        self,
        experiment: object,
        tag: str,
        values: torch.Tensor,
        step: int,
        *,
        module_key: str | None = None,
    ) -> bool:
        histogram_writer = getattr(experiment, "add_histogram", None)
        if not callable(histogram_writer) or not self._claim_emission(
            "histogram",
            tag,
            step,
            module_key,
        ):
            return False
        bounded_histogram_values = self._bounded_histogram(values)
        histogram_step = int(step)
        histogram_writer(tag, bounded_histogram_values, histogram_step)
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
        image_writer = getattr(experiment, "add_image", None)
        if (
            not callable(image_writer)
            or not self.should_emit_media(resolved_step)
            or not self._claim_emission("image", tag, resolved_step, module_key)
        ):
            return False
        bounded_image = self._bounded_image(image, dataformats=dataformats)
        if dataformats is None:
            image_writer(tag, bounded_image, global_step=resolved_step)
        else:
            image_writer(tag, bounded_image, resolved_step, dataformats=dataformats)
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
        history_heatmap = history.render_heatmap()
        if history_heatmap is None:
            return False
        return self.emit_image(
            experiment,
            tag,
            history_heatmap,
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
        resolved_module_key = module_key or _default_module_key(tag)
        emission_key = (kind, resolved_module_key, tag, int(step))
        if emission_key in self._emitted:
            return False
        self._emitted[emission_key] = None
        while len(self._emitted) > self.emitted_max_entries:
            oldest_emission_key = next(iter(self._emitted))
            self._emitted.pop(oldest_emission_key)
        return True

    def _bounded_histogram(self, values: torch.Tensor) -> torch.Tensor:
        flat_histogram_values = values.detach().float().reshape(-1).cpu()
        histogram_element_limit = max(1, int(self.histogram_max_elements))
        if flat_histogram_values.numel() <= histogram_element_limit:
            return flat_histogram_values
        last_histogram_index = flat_histogram_values.numel() - 1
        histogram_sample_indices = torch.linspace(
            0,
            last_histogram_index,
            histogram_element_limit,
            dtype=torch.long,
            device=flat_histogram_values.device,
        )
        bounded_histogram_values = flat_histogram_values.index_select(
            0, histogram_sample_indices
        )
        return bounded_histogram_values

    def _bounded_image(
        self,
        image: torch.Tensor,
        *,
        dataformats: str | None,
    ) -> torch.Tensor:
        image_snapshot = image.detach().float().cpu()
        resolved_dataformats = dataformats or self._infer_dataformats(image_snapshot)
        bounded_image = self._cap_image_side(image_snapshot, resolved_dataformats)
        bounded_image_raw_bytes = bounded_image.numel() * bounded_image.element_size()
        byte_limit_downscale_factor = 0.75
        while bounded_image_raw_bytes > self.image_max_raw_bytes:
            scaled_image = self._scale_image(
                bounded_image,
                resolved_dataformats,
                byte_limit_downscale_factor,
            )
            if scaled_image.shape == bounded_image.shape:
                break
            bounded_image = scaled_image
            bounded_image_raw_bytes = (
                bounded_image.numel() * bounded_image.element_size()
            )
        return bounded_image

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
        image_side_limit = max(1, int(self.image_max_side))
        if height <= image_side_limit and width <= image_side_limit:
            return image
        height_scale = image_side_limit / max(height, 1)
        width_scale = image_side_limit / max(width, 1)
        side_limit_scale = min(height_scale, width_scale)
        side_bounded_image = self._scale_image(image, dataformats, side_limit_scale)
        return side_bounded_image

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
        scaled_height = max(1, int(height * scale))
        scaled_width = max(1, int(width * scale))
        if scaled_height == height and scaled_width == width:
            return image
        nchw_image, restore_image_layout = self._image_as_nchw(image, dataformats)
        if nchw_image is None:
            return image
        scaled_nchw_image = F.interpolate(
            nchw_image,
            size=(scaled_height, scaled_width),
            mode="bilinear",
            align_corners=False,
        )
        scaled_image = restore_image_layout(scaled_nchw_image)
        return scaled_image

    def _image_as_nchw(
        self,
        image: torch.Tensor,
        dataformats: str,
    ) -> tuple[torch.Tensor | None, Callable[[torch.Tensor], torch.Tensor]]:
        if dataformats.endswith("CHW") and image.dim() == 3:
            return (
                image.unsqueeze(0),
                lambda scaled_nchw_image: scaled_nchw_image.squeeze(0),
            )
        if dataformats.endswith("NCHW") and image.dim() == 4:
            return image, lambda scaled_nchw_image: scaled_nchw_image
        if dataformats.endswith("HWC") and image.dim() == 3:
            return (
                image.permute(2, 0, 1).unsqueeze(0),
                lambda scaled_nchw_image: scaled_nchw_image.squeeze(0).permute(1, 2, 0),
            )
        if dataformats.endswith("NHWC") and image.dim() == 4:
            return (
                image.permute(0, 3, 1, 2),
                lambda scaled_nchw_image: scaled_nchw_image.permute(0, 2, 3, 1),
            )
        if image.dim() == 2:
            return (
                image.unsqueeze(0).unsqueeze(0),
                lambda scaled_nchw_image: scaled_nchw_image.squeeze(0).squeeze(0),
            )
        return None, lambda scaled_nchw_image: scaled_nchw_image

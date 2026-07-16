from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

HistoryNormalization = Literal["maximum", "unit_interval"]


class MonitorTensorHistory:
    """Store bounded diagnostic vectors and render them as a CHW heatmap."""

    def __init__(
        self,
        max_entries: int,
        *,
        normalization: HistoryNormalization = "maximum",
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than 0.")
        if normalization not in ("maximum", "unit_interval"):
            raise ValueError(
                "normalization must be either 'maximum' or 'unit_interval'."
            )
        self._max_entries = max_entries
        self._normalization = normalization
        self._tensors: list[torch.Tensor] = []

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def __bool__(self) -> bool:
        return bool(self._tensors)

    def append(self, values: torch.Tensor) -> None:
        self._tensors.append(values.detach().float().reshape(-1).cpu().clone())
        del self._tensors[: -self._max_entries]

    def clear(self) -> None:
        self._tensors.clear()

    def render_heatmap(self) -> torch.Tensor | None:
        if not self._tensors:
            return None
        vector_length = max(tensor.numel() for tensor in self._tensors)
        if vector_length == 0:
            return None
        padded_tensors = [
            F.pad(tensor, (0, vector_length - tensor.numel()))
            for tensor in self._tensors
        ]
        heatmap = torch.stack(padded_tensors).T
        if self._normalization == "maximum":
            heatmap = heatmap / heatmap.max().clamp_min(1e-6)
        else:
            heatmap = heatmap.clamp(0.0, 1.0)
        return heatmap.unsqueeze(0)

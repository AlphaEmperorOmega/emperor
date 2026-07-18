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
        history_entry_snapshot = values.detach().float().reshape(-1).cpu().clone()
        self._tensors.append(history_entry_snapshot)
        overflow_entries_slice = slice(None, -self._max_entries)
        del self._tensors[overflow_entries_slice]

    def clear(self) -> None:
        self._tensors.clear()

    def render_heatmap(self) -> torch.Tensor | None:
        if not self._tensors:
            return None
        maximum_entry_length = max(
            history_entry.numel() for history_entry in self._tensors
        )
        if maximum_entry_length == 0:
            return None
        padded_history_entries = [
            F.pad(
                history_entry,
                (0, maximum_entry_length - history_entry.numel()),
            )
            for history_entry in self._tensors
        ]
        unnormalized_heatmap = torch.stack(padded_history_entries).T
        if self._normalization == "maximum":
            normalization_epsilon = 1e-6
            maximum_normalization_scale = unnormalized_heatmap.max().clamp_min(
                normalization_epsilon
            )
            normalized_heatmap = unnormalized_heatmap / maximum_normalization_scale
        else:
            normalized_heatmap = unnormalized_heatmap.clamp(0.0, 1.0)
        chw_heatmap = normalized_heatmap.unsqueeze(0)
        return chw_heatmap

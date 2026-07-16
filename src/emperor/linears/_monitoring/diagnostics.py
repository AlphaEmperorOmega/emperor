from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lightning import LightningModule
    from torch import Tensor


@dataclass(frozen=True)
class _TensorSummary:
    mean: Tensor
    variance: Tensor
    norm: Tensor


@dataclass(frozen=True)
class _WeightConditioningMetrics:
    spectral_norm: Tensor
    condition_number: Tensor
    effective_rank: Tensor


@dataclass(frozen=True)
class _ParameterChangeMetrics:
    delta_norm: Tensor
    relative_delta_norm: Tensor


@dataclass(frozen=True)
class _LinearParameterChannelMetrics:
    values: Tensor
    summary: _TensorSummary
    change: _ParameterChangeMetrics | None
    gradient_summary: _TensorSummary | None
    update_ratio: Tensor | None


@dataclass(frozen=True)
class _LinearTrackingContext:
    pl_module: LightningModule
    module_name: str
    weights: _LinearParameterChannelMetrics
    bias: _LinearParameterChannelMetrics | None
    input_feature_norms: Tensor
    output_feature_norms: Tensor
    weight_conditioning: _WeightConditioningMetrics | None


class _LinearDiagnostics:
    @staticmethod
    def summarize(values: Tensor) -> _TensorSummary:
        detached_values = values.detach().float()
        return _TensorSummary(
            mean=detached_values.mean(),
            variance=detached_values.var(unbiased=False),
            norm=detached_values.norm(),
        )

    @staticmethod
    def weight_conditioning(weight: Tensor) -> _WeightConditioningMetrics:
        singular_values = torch.linalg.svdvals(weight.detach().float())
        spectral_norm = singular_values.max()
        condition_number = spectral_norm / singular_values.min().clamp_min(1e-12)
        normalized_spectrum = singular_values / singular_values.sum().clamp_min(1e-12)
        spectral_entropy = -(
            normalized_spectrum.clamp_min(1e-12).log() * normalized_spectrum
        ).sum()
        return _WeightConditioningMetrics(
            spectral_norm=spectral_norm,
            condition_number=condition_number,
            effective_rank=spectral_entropy.exp(),
        )

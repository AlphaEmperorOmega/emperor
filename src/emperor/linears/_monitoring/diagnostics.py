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
    def diagnostic_values(values: Tensor) -> Tensor:
        detached_values = values.detach()
        if detached_values.is_complex():
            detached_values = detached_values.abs()
        if detached_values.dtype in {torch.float32, torch.float64}:
            return detached_values
        return detached_values.float()

    @classmethod
    def summarize(cls, values: Tensor) -> _TensorSummary:
        detached_values = cls.diagnostic_values(values)
        scale = detached_values.abs().amax()
        safe_scale = torch.where(
            torch.isfinite(scale) & (scale > 0),
            scale,
            torch.ones_like(scale),
        )
        normalized_values = detached_values / safe_scale
        normalized_variance, normalized_mean = torch.var_mean(
            normalized_values,
            correction=0,
        )
        standard_deviation = normalized_variance.clamp_min(0).sqrt() * scale
        return _TensorSummary(
            mean=normalized_mean * scale,
            variance=standard_deviation.square(),
            norm=torch.linalg.vector_norm(normalized_values) * scale,
        )

    @classmethod
    def stable_norm(cls, values: Tensor, dim: int | None = None) -> Tensor:
        diagnostic_values = cls.diagnostic_values(values)
        absolute_values = diagnostic_values.abs()
        if dim is None:
            scale = absolute_values.amax()
        else:
            scale = absolute_values.amax(dim=dim, keepdim=True)
        safe_scale = torch.where(
            torch.isfinite(scale) & (scale > 0),
            scale,
            torch.ones_like(scale),
        )
        norm = torch.linalg.vector_norm(diagnostic_values / safe_scale, dim=dim)
        if dim is not None:
            scale = scale.squeeze(dim)
        return norm * scale

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

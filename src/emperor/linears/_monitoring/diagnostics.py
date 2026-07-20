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


@dataclass
class _TensorMoments:
    count: int = 0
    mean: Tensor | None = None
    second_moment: Tensor | None = None

    def add(self, values: Tensor) -> None:
        diagnostic_values = _LinearDiagnostics.diagnostic_values(values)
        batch_count = diagnostic_values.numel()
        if batch_count == 0:
            return

        absolute_values = diagnostic_values.abs()
        scale = absolute_values.amax()
        safe_scale = torch.where(
            torch.isfinite(scale) & (scale > 0),
            scale,
            torch.ones_like(scale),
        )
        normalized_values = diagnostic_values / safe_scale
        normalized_variance, normalized_mean = torch.var_mean(
            normalized_values,
            correction=0,
        )
        accumulator_dtype = {"cpu": torch.float64}.get(
            diagnostic_values.device.type,
            diagnostic_values.dtype,
        )
        accumulator_scale = scale.to(dtype=accumulator_dtype)
        batch_mean = normalized_mean.to(dtype=accumulator_dtype) * accumulator_scale
        batch_standard_deviation = (
            normalized_variance.clamp_min(0).sqrt().to(dtype=accumulator_dtype)
            * accumulator_scale
        )
        batch_second_moment = batch_standard_deviation.square() * batch_count
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.second_moment = batch_second_moment
            return

        assert self.mean is not None
        assert self.second_moment is not None
        promoted_dtype = torch.promote_types(self.mean.dtype, batch_mean.dtype)
        current_mean = self.mean.to(dtype=promoted_dtype)
        current_second_moment = self.second_moment.to(dtype=promoted_dtype)
        batch_mean = batch_mean.to(dtype=promoted_dtype)
        batch_second_moment = batch_second_moment.to(dtype=promoted_dtype)

        combined_count = self.count + batch_count
        mean_delta = batch_mean - current_mean
        self.mean = current_mean * (self.count / combined_count) + batch_mean * (
            batch_count / combined_count
        )
        self.second_moment = (
            current_second_moment
            + batch_second_moment
            + mean_delta.abs().square() * (self.count * batch_count / combined_count)
        )
        self.count = combined_count

    def summarize(self) -> _TensorSummary | None:
        if self.count == 0:
            return None
        assert self.mean is not None
        assert self.second_moment is not None
        variance = (self.second_moment / self.count).clamp_min(0)
        norm = torch.hypot(
            self.second_moment.clamp_min(0).sqrt(),
            self.mean.abs() * self.count**0.5,
        )
        return _TensorSummary(
            mean=self.mean,
            variance=variance,
            norm=norm,
        )


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
    def safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
        ratio = numerator / denominator
        both_zero = (numerator == 0) & (denominator == 0)
        return torch.where(both_zero, torch.zeros_like(ratio), ratio)

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

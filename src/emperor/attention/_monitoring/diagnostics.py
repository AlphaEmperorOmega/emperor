"""Private attention diagnostic capture and calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from emperor.attention._runtime import QKV

if TYPE_CHECKING:
    from typing import Literal

    from torch import Tensor
    from torch.nn import Module


@dataclass
class _AttentionObservation:
    projected_qkv: QKV | None = None
    processor_qkv: QKV | None = None
    merged_attention_mask: Tensor | None = None
    exact_attention_weights: Tensor | None = None
    restored_output: Tensor | None = None
    auxiliary_loss: Tensor | None = None


@dataclass(frozen=True)
class _AttentionDiagnosticMetrics:
    query_norm_mean: Tensor | None
    key_norm_mean: Tensor | None
    value_norm_mean: Tensor | None
    output_norm: Tensor | None
    auxiliary_loss: Tensor | None
    configured_dropout_probability: Tensor
    mask_coverage: Tensor
    per_head_entropy: Tensor | None
    per_head_max_probability: Tensor | None
    weight_source: Literal["exact", "approximate"] | None
    dropout_zero_fraction: Tensor | None


class _AttentionMonitorAdapter:
    """Capture and canonicalize standard attention weights."""

    @property
    def exact_weight_method_names(self) -> tuple[str, ...]:
        return ("_SelfAttentionProcessor__compute_masked_attention_weights",)

    @staticmethod
    def canonicalize(attention_weights: Tensor, num_heads: int) -> Tensor | None:
        if num_heads <= 0:
            return None
        detached_weights = attention_weights.detach().float()
        if detached_weights.dim() == 4:
            if detached_weights.size(1) == num_heads:
                return detached_weights
            if detached_weights.size(0) == num_heads:
                return detached_weights.permute(1, 0, 2, 3)
        if detached_weights.dim() == 3 and detached_weights.size(0) % num_heads == 0:
            return detached_weights.reshape(
                -1,
                num_heads,
                detached_weights.size(-2),
                detached_weights.size(-1),
            )
        return None


_DEFAULT_ATTENTION_MONITOR_ADAPTER = _AttentionMonitorAdapter()


def _resolve_attention_monitor_adapter(
    attention_module: Module,
) -> _AttentionMonitorAdapter:
    monitor_adapter = getattr(
        attention_module,
        "_MONITOR_ADAPTER",
        _DEFAULT_ATTENTION_MONITOR_ADAPTER,
    )
    if isinstance(monitor_adapter, _AttentionMonitorAdapter):
        return monitor_adapter
    return _DEFAULT_ATTENTION_MONITOR_ADAPTER


class _AttentionDiagnostics:
    """Calculate attention diagnostics without Lightning or emission concerns."""

    DEAD_HEAD_ENTROPY_FLOOR = 1e-6

    def __init__(
        self,
        monitor_adapter: _AttentionMonitorAdapter | None = None,
    ) -> None:
        self._monitor_adapter = monitor_adapter or _DEFAULT_ATTENTION_MONITOR_ADAPTER

    def calculate(
        self,
        observation: _AttentionObservation,
        *,
        num_heads: int,
        configured_dropout_probability: float,
        monitor_adapter: _AttentionMonitorAdapter | None = None,
    ) -> _AttentionDiagnosticMetrics:
        projected_qkv = observation.projected_qkv
        exact_weights = observation.exact_attention_weights
        selected_weights = exact_weights
        weight_source: Literal["exact", "approximate"] | None = None
        if exact_weights is not None:
            weight_source = "exact"
        else:
            selected_weights = self.approximate_attention_weights(
                observation.processor_qkv,
                observation.merged_attention_mask,
            )
            if selected_weights is not None:
                weight_source = "approximate"
        per_head_entropy, per_head_max_probability = self.per_head_statistics(
            selected_weights,
            num_heads,
            monitor_adapter,
        )
        return _AttentionDiagnosticMetrics(
            query_norm_mean=self.__projection_norm(projected_qkv, "query"),
            key_norm_mean=self.__projection_norm(projected_qkv, "key"),
            value_norm_mean=self.__projection_norm(projected_qkv, "value"),
            output_norm=self.__output_norm(observation.restored_output),
            auxiliary_loss=self.__mean(observation.auxiliary_loss),
            configured_dropout_probability=torch.tensor(
                float(configured_dropout_probability)
            ),
            mask_coverage=self.mask_coverage(observation.merged_attention_mask),
            per_head_entropy=per_head_entropy,
            per_head_max_probability=per_head_max_probability,
            weight_source=weight_source,
            dropout_zero_fraction=(
                (exact_weights.float() == 0.0).float().mean()
                if exact_weights is not None
                else None
            ),
        )

    @staticmethod
    def approximate_attention_weights(
        processor_qkv: QKV | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        if processor_qkv is None:
            return None
        query = processor_qkv.query
        key = processor_qkv.key
        if query.dim() not in (3, 4) or key.dim() not in (3, 4):
            return None
        query_values = query.detach().float()
        key_values = key.detach().float()
        attention_scores = torch.matmul(
            query_values * query_values.size(-1) ** -0.5,
            key_values.transpose(-2, -1),
        )
        if attention_mask is not None:
            detached_mask = attention_mask.detach()
            try:
                if detached_mask.dtype == torch.bool:
                    attention_scores = attention_scores.masked_fill(
                        detached_mask,
                        -torch.inf,
                    )
                else:
                    attention_scores = attention_scores + detached_mask.float()
            except RuntimeError:
                return None
        fully_masked_rows = torch.isneginf(attention_scores).all(
            dim=-1,
            keepdim=True,
        )
        safe_attention_scores = attention_scores.masked_fill(
            fully_masked_rows,
            0.0,
        )
        weights = F.softmax(safe_attention_scores, dim=-1)
        return weights.masked_fill(fully_masked_rows, 0.0)

    def per_head_statistics(
        self,
        attention_weights: Tensor | None,
        num_heads: int,
        monitor_adapter: _AttentionMonitorAdapter | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if attention_weights is None:
            return None, None
        adapter = monitor_adapter or self._monitor_adapter
        weights_by_head = adapter.canonicalize(
            attention_weights,
            num_heads,
        )
        if weights_by_head is None or weights_by_head.numel() == 0:
            return None, None
        normalized_weights = weights_by_head / weights_by_head.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        target_entropy = -(
            normalized_weights.clamp_min(1e-12).log() * normalized_weights
        ).sum(dim=-1)
        target_max_probability = normalized_weights.max(dim=-1).values
        aggregate_dimensions = tuple(
            dimension for dimension in range(target_entropy.dim()) if dimension != 1
        )
        return (
            target_entropy.mean(dim=aggregate_dimensions),
            target_max_probability.mean(dim=aggregate_dimensions),
        )

    @staticmethod
    def mask_coverage(attention_mask: Tensor | None) -> Tensor:
        if attention_mask is None or attention_mask.numel() == 0:
            return torch.zeros(())
        detached_mask = attention_mask.detach().float()
        if attention_mask.dtype == torch.bool:
            return detached_mask.mean()
        return (detached_mask != 0.0).float().mean()

    @staticmethod
    def __projection_norm(projected_qkv: QKV | None, name: str) -> Tensor | None:
        if projected_qkv is None:
            return None
        projection = getattr(projected_qkv, name).detach().float()
        return projection.norm(dim=-1).mean()

    @staticmethod
    def __output_norm(restored_output: Tensor | None) -> Tensor | None:
        return (
            restored_output.detach().float().norm()
            if restored_output is not None
            else None
        )

    @staticmethod
    def __mean(value: Tensor | None) -> Tensor | None:
        return value.detach().float().mean() if value is not None else None

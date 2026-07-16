from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lightning import LightningModule
    from torch import Tensor
    from torch.nn import Module


@dataclass
class _RecurrentObservation:
    step_deltas: list[Tensor] = field(default_factory=list)
    gate_values: list[Tensor] = field(default_factory=list)


@dataclass(frozen=True)
class _RecurrentDiagnosticMetrics:
    actual_steps: int
    step_delta_means: Tensor
    maximum_step_delta: Tensor
    flattened_step_deltas: Tensor
    gate_values: Tensor | None


@dataclass(frozen=True)
class _RecurrentTrackingContext:
    pl_module: LightningModule
    module_name: str
    metric_prefix: str
    recurrent_layer: Module
    metrics: _RecurrentDiagnosticMetrics | None
    device: torch.device | str
    experiment: object | None
    global_step: int


@dataclass(frozen=True)
class _LayerGateTrackingContext:
    pl_module: LightningModule
    module_name: str
    raw_values: Tensor
    effective_values: Tensor | None


@dataclass(frozen=True)
class _LayerDropoutTrackingContext:
    pl_module: LightningModule
    module_name: str
    input_values: Tensor
    output_values: Tensor


@dataclass(frozen=True)
class _LayerNormTrackingContext:
    pl_module: LightningModule
    module_name: str
    input_values: Tensor
    output_values: Tensor


@dataclass(frozen=True)
class _LayerActivationTrackingContext:
    pl_module: LightningModule
    module_name: str
    activation_values: Tensor


@dataclass(frozen=True)
class _LayerResidualTrackingContext:
    pl_module: LightningModule
    module_name: str
    output_values: Tensor
    input_values: Tensor
    previous_values: Tensor


class _RecurrentDiagnostics:
    @staticmethod
    def calculate(
        observation: _RecurrentObservation,
    ) -> _RecurrentDiagnosticMetrics | None:
        step_deltas = [
            step_delta.detach().float().reshape(-1)
            for step_delta in observation.step_deltas
        ]
        if not step_deltas:
            return None
        step_delta_means = torch.stack(
            [step_delta.mean() for step_delta in step_deltas]
        )
        maximum_step_delta = torch.stack(
            [step_delta.max() for step_delta in step_deltas]
        ).max()
        gate_values = (
            torch.cat(
                [
                    gate_value.detach().float().reshape(-1)
                    for gate_value in observation.gate_values
                ]
            )
            if observation.gate_values
            else None
        )
        return _RecurrentDiagnosticMetrics(
            actual_steps=len(step_deltas),
            step_delta_means=step_delta_means,
            maximum_step_delta=maximum_step_delta,
            flattened_step_deltas=torch.cat(step_deltas),
            gate_values=gate_values,
        )

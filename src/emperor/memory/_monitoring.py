from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class _MemoryDiagnosticMetrics:
    output_mean: Tensor
    output_variance: Tensor
    output_norm: Tensor
    delta_mean: Tensor
    delta_variance: Tensor
    delta_norm: Tensor
    relative_delta_norm: Tensor


@dataclass(frozen=True)
class _MemoryGateMetrics:
    open_mean: Tensor
    open_fraction: Tensor
    saturation_fraction: Tensor | None


@dataclass(frozen=True)
class _MemoryObservation:
    memory_module: Module
    input_values: Tensor
    output_values: Tensor
    gate_logits: Tensor | None


@dataclass(frozen=True)
class _MemoryTrackingContext:
    pl_module: LightningModule
    module_name: str
    memory_metrics: _MemoryDiagnosticMetrics
    gate_metrics: _MemoryGateMetrics | None


class _MemoryDiagnostics:
    _LOW_SATURATION_LOGIT = math.log(0.01 / 0.99)
    _HIGH_SATURATION_LOGIT = math.log(0.99 / 0.01)

    @staticmethod
    def calculate(
        input_values: Tensor, output_values: Tensor
    ) -> _MemoryDiagnosticMetrics:
        memory_delta = output_values - input_values
        delta_norm = memory_delta.norm()
        return _MemoryDiagnosticMetrics(
            output_mean=output_values.mean(),
            output_variance=output_values.var(unbiased=False),
            output_norm=output_values.norm(),
            delta_mean=memory_delta.mean(),
            delta_variance=memory_delta.var(unbiased=False),
            delta_norm=delta_norm,
            relative_delta_norm=delta_norm / input_values.norm().clamp_min(1e-6),
        )

    @staticmethod
    def calculate_gate(
        memory_module: Module,
        gate_logits: Tensor,
    ) -> _MemoryGateMetrics:
        from emperor.memory._variants.weighted import WeightedDynamicMemory

        detached_logits = gate_logits.detach().float()
        if isinstance(memory_module, WeightedDynamicMemory):
            memory_share = torch.softmax(detached_logits, dim=-1).select(
                dim=-1,
                index=1,
            )
            return _MemoryGateMetrics(
                open_mean=memory_share.mean(),
                open_fraction=(memory_share > 0.5).float().mean(),
                saturation_fraction=None,
            )
        gate_values = torch.sigmoid(detached_logits)
        return _MemoryGateMetrics(
            open_mean=gate_values.mean(),
            open_fraction=(gate_values > 0.5).float().mean(),
            saturation_fraction=(
                (
                    (detached_logits < _MemoryDiagnostics._LOW_SATURATION_LOGIT)
                    | (detached_logits > _MemoryDiagnostics._HIGH_SATURATION_LOGIT)
                )
                .float()
                .mean()
            ),
        )


class MemoryMonitorCallback(Callback):
    """Log output, contribution, and gate diagnostics for dynamic memory."""

    _GATE_SUBMODULE_NAMES = ("memory_gate_model", "memory_weight_model")

    def __init__(self, log_every_n_steps: int = 100) -> None:
        super().__init__()
        if not isinstance(log_every_n_steps, int) or isinstance(
            log_every_n_steps,
            bool,
        ):
            raise TypeError(
                "log_every_n_steps must be a positive integer, "
                f"received {type(log_every_n_steps).__name__}."
            )
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self._hooks: list[RemovableHandle] = []
        self._latest_gate_logits: dict[str, Tensor] = {}

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from emperor.memory._base import DynamicMemoryAbstract

        self.__cleanup()
        for module_name, memory_module in pl_module.named_modules():
            if not isinstance(memory_module, DynamicMemoryAbstract):
                continue
            self._hooks.extend(
                (
                    memory_module.register_forward_pre_hook(
                        self.__make_memory_pre_hook(module_name)
                    ),
                    memory_module.register_forward_hook(
                        self.__make_memory_forward_hook(
                            module_name,
                            trainer,
                            pl_module,
                        )
                    ),
                )
            )
            gate_submodule = self.__find_gate_submodule(memory_module)
            if gate_submodule is not None:
                self._hooks.append(
                    gate_submodule.register_forward_hook(
                        self.__make_gate_capture_hook(module_name)
                    )
                )

    @classmethod
    def __find_gate_submodule(cls, memory_module: Module) -> Module | None:
        for submodule_name in cls._GATE_SUBMODULE_NAMES:
            gate_submodule = getattr(memory_module, submodule_name, None)
            if gate_submodule is not None:
                return gate_submodule
        return None

    def __make_memory_pre_hook(
        self,
        module_name: str,
    ) -> Callable[[Module, tuple[object, ...]], None]:
        def clear_previous_gate_logits(
            _memory_module: Module,
            _inputs: tuple[object, ...],
        ) -> None:
            self._latest_gate_logits.pop(module_name, None)

        return clear_previous_gate_logits

    def __make_gate_capture_hook(
        self,
        module_name: str,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def capture_gate_logits(
            _gate_submodule: Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            gate_logits = self.__extract_hidden_tensor(output)
            if gate_logits is not None:
                self._latest_gate_logits[module_name] = gate_logits.detach()

        return capture_gate_logits

    @staticmethod
    def __extract_hidden_tensor(output: object) -> Tensor | None:
        if torch.is_tensor(output):
            return output
        hidden = getattr(output, "hidden", None)
        return hidden if torch.is_tensor(hidden) else None

    def __make_memory_forward_hook(
        self,
        module_name: str,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def log_memory_output(
            memory_module: Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            gate_logits = self._latest_gate_logits.pop(module_name, None)
            global_step = trainer.global_step
            if global_step % self.log_every_n_steps != 0:
                return
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            if not torch.is_tensor(output):
                return
            observation = _MemoryObservation(
                memory_module=memory_module,
                input_values=inputs[0],
                output_values=output,
                gate_logits=gate_logits,
            )
            context = self.__build_tracking_context(
                pl_module,
                module_name,
                observation,
            )
            self.__track_memory_diagnostics(context)

        return log_memory_output

    @staticmethod
    def __build_tracking_context(
        pl_module: LightningModule,
        module_name: str,
        observation: _MemoryObservation,
    ) -> _MemoryTrackingContext:
        gate_metrics = (
            _MemoryDiagnostics.calculate_gate(
                observation.memory_module,
                observation.gate_logits,
            )
            if observation.gate_logits is not None
            else None
        )
        return _MemoryTrackingContext(
            pl_module=pl_module,
            module_name=module_name,
            memory_metrics=_MemoryDiagnostics.calculate(
                observation.input_values.detach().float(),
                observation.output_values.detach().float(),
            ),
            gate_metrics=gate_metrics,
        )

    def __track_memory_diagnostics(
        self,
        context: _MemoryTrackingContext,
    ) -> None:
        self.__track_output_mean(context)
        self.__track_output_variance(context)
        self.__track_output_l2_norm(context)
        self.__track_contribution_delta_mean(context)
        self.__track_contribution_delta_variance(context)
        self.__track_contribution_delta_norm(context)
        self.__track_contribution_relative_delta_norm(context)
        self.__track_gate_open_mean(context)
        self.__track_gate_open_fraction(context)
        self.__track_gate_saturation_fraction(context)

    @staticmethod
    def __track_output_mean(context: _MemoryTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/memory/output_mean",
            context.memory_metrics.output_mean,
        )

    @staticmethod
    def __track_output_variance(context: _MemoryTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/memory/output_var",
            context.memory_metrics.output_variance,
        )

    @staticmethod
    def __track_output_l2_norm(context: _MemoryTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/memory/output_l2_norm",
            context.memory_metrics.output_norm,
        )

    @staticmethod
    def __track_contribution_delta_mean(context: _MemoryTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/memory/contribution/delta_mean",
            context.memory_metrics.delta_mean,
        )

    @staticmethod
    def __track_contribution_delta_variance(
        context: _MemoryTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/memory/contribution/delta_var",
            context.memory_metrics.delta_variance,
        )

    @staticmethod
    def __track_contribution_delta_norm(context: _MemoryTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/memory/contribution/delta_norm",
            context.memory_metrics.delta_norm,
        )

    @staticmethod
    def __track_contribution_relative_delta_norm(
        context: _MemoryTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/memory/contribution/relative_delta_norm",
            context.memory_metrics.relative_delta_norm,
        )

    @staticmethod
    def __track_gate_open_mean(context: _MemoryTrackingContext) -> None:
        if context.gate_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/memory/gate/open_mean",
            context.gate_metrics.open_mean,
        )

    @staticmethod
    def __track_gate_open_fraction(context: _MemoryTrackingContext) -> None:
        if context.gate_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/memory/gate/open_fraction",
            context.gate_metrics.open_fraction,
        )

    @staticmethod
    def __track_gate_saturation_fraction(context: _MemoryTrackingContext) -> None:
        if (
            context.gate_metrics is None
            or context.gate_metrics.saturation_fraction is None
        ):
            return
        context.pl_module.log(
            f"{context.module_name}/memory/gate/saturation_fraction",
            context.gate_metrics.saturation_fraction,
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__cleanup()

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        self.__cleanup()

    def __cleanup(self) -> None:
        for hook_handle in self._hooks:
            hook_handle.remove()
        self._hooks.clear()
        self._latest_gate_logits.clear()

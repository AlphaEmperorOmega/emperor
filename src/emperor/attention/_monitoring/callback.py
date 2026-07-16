"""Private attention monitoring callback implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import Callback

from emperor.attention._monitoring.diagnostics import (
    _AttentionDiagnosticMetrics,
    _AttentionDiagnostics,
    _AttentionMonitorAdapter,
    _AttentionObservation,
    _resolve_attention_monitor_adapter,
)
from emperor.attention._runtime import QKV
from emperor.monitoring import (
    MonitorEmissionPolicy,
    MonitorTensorHistory,
)

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class _AttentionMethodReplacement:
    owner: object
    method_name: str
    original_method: Callable[..., object]


class _AttentionDiagnosticsTracker:
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self._latest_observation = _AttentionObservation()

    @property
    def latest_observation(self) -> _AttentionObservation:
        return self._latest_observation

    def begin_observation(self) -> None:
        self._latest_observation = _AttentionObservation()

    def record_projected_qkv(self, projected_qkv: object) -> None:
        detached_qkv = self.__detach_qkv(projected_qkv)
        if detached_qkv is not None:
            self._latest_observation.projected_qkv = detached_qkv

    def record_processor_inputs(
        self,
        processor_qkv: object,
        merged_attention_mask: object,
    ) -> None:
        self._latest_observation.processor_qkv = self.__detach_qkv(processor_qkv)
        self._latest_observation.merged_attention_mask = self.__detach_tensor(
            merged_attention_mask
        )

    def record_exact_attention_weights(self, attention_weights: object) -> None:
        detached_weights = self.__detach_tensor(attention_weights)
        if detached_weights is not None:
            self._latest_observation.exact_attention_weights = detached_weights

    def record_forward_output(self, forward_output: object) -> None:
        restored_output, returned_weights, auxiliary_loss = self.__parse_forward_output(
            forward_output
        )
        self._latest_observation.restored_output = restored_output
        if (
            returned_weights is not None
            and self._latest_observation.exact_attention_weights is None
        ):
            self._latest_observation.exact_attention_weights = returned_weights
        self._latest_observation.auxiliary_loss = auxiliary_loss

    @classmethod
    def __detach_qkv(cls, value: object) -> QKV | None:
        if not isinstance(value, QKV):
            return None
        return QKV(
            query=value.query.detach(),
            key=value.key.detach(),
            value=value.value.detach(),
        )

    @staticmethod
    def __detach_tensor(value: object) -> Tensor | None:
        return value.detach() if torch.is_tensor(value) else None

    @classmethod
    def __parse_forward_output(
        cls,
        forward_output: object,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if not isinstance(forward_output, tuple):
            return cls.__detach_tensor(forward_output), None, None
        output = forward_output[0] if len(forward_output) > 0 else None
        attention_weights = forward_output[1] if len(forward_output) > 1 else None
        auxiliary_loss = forward_output[2] if len(forward_output) > 2 else None
        return (
            cls.__detach_tensor(output),
            cls.__detach_tensor(attention_weights),
            cls.__detach_tensor(auxiliary_loss),
        )


class _AttentionDiagnosticsTrackerManager:
    """Attach, restore, and own attention diagnostic instrumentation."""

    def __init__(self) -> None:
        self._trackers: dict[int, _AttentionDiagnosticsTracker] = {}
        self._hook_handles: list[RemovableHandle] = []
        self._method_replacements: list[_AttentionMethodReplacement] = []

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(tracker.module_name for tracker in self._trackers.values())

    @property
    def hook_count(self) -> int:
        return len(self._hook_handles)

    @property
    def replacement_count(self) -> int:
        return len(self._method_replacements)

    def tracker_for(self, attention_module: Module) -> _AttentionDiagnosticsTracker:
        return self._trackers[id(attention_module)]

    def attach(
        self,
        module_name: str,
        attention_module: Module,
        should_capture: Callable[[], bool],
        observation_recorder: Callable[
            [str, Module, _AttentionObservation],
            None,
        ],
    ) -> None:
        tracker = _AttentionDiagnosticsTracker(module_name)
        self._trackers[id(attention_module)] = tracker
        monitor_adapter = _resolve_attention_monitor_adapter(attention_module)
        projector_attached = self.__attach_projector(
            attention_module,
            tracker,
            should_capture,
        )
        processor_attached = self.__attach_processor(
            attention_module,
            tracker,
            should_capture,
            monitor_adapter,
            begin_observation=not projector_attached,
        )
        self._hook_handles.append(
            attention_module.register_forward_hook(
                self.__make_forward_hook(
                    attention_module,
                    tracker,
                    should_capture,
                    observation_recorder,
                    begin_observation=not projector_attached and not processor_attached,
                )
            )
        )

    def detach(self) -> None:
        for hook_handle in self._hook_handles:
            hook_handle.remove()
        self._hook_handles.clear()
        for replacement in reversed(self._method_replacements):
            setattr(
                replacement.owner,
                replacement.method_name,
                replacement.original_method,
            )
        self._method_replacements.clear()
        self._trackers.clear()

    def __attach_projector(
        self,
        attention_module: Module,
        tracker: _AttentionDiagnosticsTracker,
        should_capture: Callable[[], bool],
    ) -> bool:
        projector = getattr(attention_module, "projector", None)
        method_name = "compute_qkv_projections"
        original_projection = getattr(projector, method_name, None)
        if not callable(original_projection):
            return False

        def capture_projected_qkv(*args: object, **kwargs: object) -> object:
            capture_this_forward = should_capture()
            if capture_this_forward:
                tracker.begin_observation()
            projected_qkv = original_projection(*args, **kwargs)
            if capture_this_forward:
                tracker.record_projected_qkv(projected_qkv)
            return projected_qkv

        self.__replace_method(
            projector,
            method_name,
            original_projection,
            capture_projected_qkv,
        )
        return True

    def __attach_processor(
        self,
        attention_module: Module,
        tracker: _AttentionDiagnosticsTracker,
        should_capture: Callable[[], bool],
        monitor_adapter: _AttentionMonitorAdapter,
        *,
        begin_observation: bool,
    ) -> bool:
        processor = getattr(attention_module, "processor", None)
        method_name = "compute_attention"
        original_attention = getattr(processor, method_name, None)
        if not callable(original_attention):
            return False

        def capture_processor_inputs(*args: object, **kwargs: object) -> object:
            if should_capture():
                if begin_observation:
                    tracker.begin_observation()
                tracker.record_processor_inputs(
                    args[0] if args else kwargs.get("qkv"),
                    args[1] if len(args) > 1 else kwargs.get("merged_attention_mask"),
                )
            return original_attention(*args, **kwargs)

        self.__replace_method(
            processor,
            method_name,
            original_attention,
            capture_processor_inputs,
        )
        self.__attach_exact_weight_methods(
            processor,
            tracker,
            should_capture,
            monitor_adapter,
        )
        return True

    def __attach_exact_weight_methods(
        self,
        processor: object,
        tracker: _AttentionDiagnosticsTracker,
        should_capture: Callable[[], bool],
        monitor_adapter: _AttentionMonitorAdapter,
    ) -> None:
        for method_name in monitor_adapter.exact_weight_method_names:
            original_weight_method = getattr(processor, method_name, None)
            if not callable(original_weight_method):
                continue
            self.__replace_method(
                processor,
                method_name,
                original_weight_method,
                self.__make_exact_weight_wrapper(
                    tracker,
                    should_capture,
                    original_weight_method,
                ),
            )

    @staticmethod
    def __make_exact_weight_wrapper(
        tracker: _AttentionDiagnosticsTracker,
        should_capture: Callable[[], bool],
        original_weight_method: Callable[..., object],
    ) -> Callable[..., object]:
        def capture_exact_weights(*args: object, **kwargs: object) -> object:
            attention_weights = original_weight_method(*args, **kwargs)
            if should_capture():
                tracker.record_exact_attention_weights(attention_weights)
            return attention_weights

        return capture_exact_weights

    @staticmethod
    def __make_forward_hook(
        attention_module: Module,
        tracker: _AttentionDiagnosticsTracker,
        should_capture: Callable[[], bool],
        observation_recorder: Callable[
            [str, Module, _AttentionObservation],
            None,
        ],
        *,
        begin_observation: bool,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def record_forward_diagnostics(
            _layer: Module,
            _inputs: tuple[object, ...],
            forward_output: object,
        ) -> None:
            if not should_capture():
                return
            if begin_observation:
                tracker.begin_observation()
            tracker.record_forward_output(forward_output)
            observation_recorder(
                tracker.module_name,
                attention_module,
                tracker.latest_observation,
            )

        return record_forward_diagnostics

    def __replace_method(
        self,
        owner: object,
        method_name: str,
        original_method: Callable[..., object],
        replacement_method: Callable[..., object],
    ) -> None:
        setattr(owner, method_name, replacement_method)
        self._method_replacements.append(
            _AttentionMethodReplacement(owner, method_name, original_method)
        )


@dataclass(frozen=True)
class _AttentionTrackingContext:
    pl_module: LightningModule
    module_name: str
    metric_prefix: str
    metrics: _AttentionDiagnosticMetrics
    experiment: object | None
    global_step: int


class AttentionMonitorCallback(Callback):
    """Own attention-monitor lifecycle, cadence, history, and metric emission."""

    DEAD_HEAD_ENTROPY_FLOOR = _AttentionDiagnostics.DEAD_HEAD_ENTROPY_FLOOR

    def __init__(
        self,
        log_every_n_steps: int = 100,
        history_size: int = 128,
        log_per_head_scalars: bool = False,
    ) -> None:
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        if history_size <= 0:
            raise ValueError("history_size must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self.log_per_head_scalars = log_per_head_scalars
        self._tracker_manager = _AttentionDiagnosticsTrackerManager()
        self._diagnostics = _AttentionDiagnostics()
        self._entropy_history: dict[str, MonitorTensorHistory] = {}
        self._max_probability_history: dict[str, MonitorTensorHistory] = {}
        self._emission_policy = MonitorEmissionPolicy()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from emperor.attention._base import MultiHeadAttentionAbstract

        self.__cleanup()
        for module_name, attention_module in pl_module.named_modules():
            if isinstance(attention_module, MultiHeadAttentionAbstract):
                self.__attach_attention_module(
                    module_name,
                    attention_module,
                    pl_module,
                )

    def __attach_attention_module(
        self,
        module_name: str,
        attention_module: Module,
        pl_module: LightningModule,
    ) -> None:
        self._entropy_history[module_name] = MonitorTensorHistory(self.history_size)
        self._max_probability_history[module_name] = MonitorTensorHistory(
            self.history_size
        )

        def should_capture() -> bool:
            return self.__should_sample(pl_module)

        def emit_observation(
            observed_module_name: str,
            observed_module: Module,
            observation: _AttentionObservation,
        ) -> None:
            self.__emit_observation(
                pl_module,
                observed_module_name,
                observed_module,
                observation,
            )

        self._tracker_manager.attach(
            module_name,
            attention_module,
            should_capture,
            emit_observation,
        )

    def __should_sample(self, pl_module: LightningModule) -> bool:
        return getattr(pl_module, "global_step", 0) % self.log_every_n_steps == 0

    def __emit_observation(
        self,
        pl_module: LightningModule,
        module_name: str,
        attention_module: Module,
        observation: _AttentionObservation,
    ) -> None:
        context = self.__build_tracking_context(
            pl_module,
            module_name,
            attention_module,
            observation,
        )
        self.__track_attention_observation(context)

    def __build_tracking_context(
        self,
        pl_module: LightningModule,
        module_name: str,
        attention_module: Module,
        observation: _AttentionObservation,
    ) -> _AttentionTrackingContext:
        metrics = self._diagnostics.calculate(
            observation,
            num_heads=int(getattr(attention_module, "num_heads", 0) or 0),
            configured_dropout_probability=float(
                getattr(attention_module, "dropout_probability", 0.0)
            ),
            monitor_adapter=_resolve_attention_monitor_adapter(attention_module),
        )
        experiment = getattr(getattr(pl_module, "logger", None), "experiment", None)
        return _AttentionTrackingContext(
            pl_module=pl_module,
            module_name=module_name,
            metric_prefix=f"{module_name}/attention",
            metrics=metrics,
            experiment=experiment,
            global_step=getattr(pl_module, "global_step", 0),
        )

    def __track_attention_observation(
        self,
        context: _AttentionTrackingContext,
    ) -> None:
        self.__track_query_norm_mean(context)
        self.__track_key_norm_mean(context)
        self.__track_value_norm_mean(context)
        self.__track_output_norm(context)
        self.__track_auxiliary_loss(context)
        self.__track_configured_dropout_probability(context)
        self.__track_mask_coverage(context)
        self.__track_entropy_mean(context)
        self.__track_max_probability_mean(context)
        self.__track_dead_head_fraction(context)
        self.__track_per_head_entropy(context)
        self.__track_per_head_max_probability(context)
        self.__track_entropy_history(context)
        self.__track_max_probability_history(context)
        self.__track_entropy_histogram(context)
        self.__track_entropy_heatmap(context)
        self.__track_max_probability_histogram(context)
        self.__track_max_probability_heatmap(context)
        self.__track_dropout_zero_fraction(context)

    @staticmethod
    def __track_query_norm_mean(context: _AttentionTrackingContext) -> None:
        query_norm_mean = context.metrics.query_norm_mean
        if query_norm_mean is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/q_norm_mean",
                query_norm_mean,
            )

    @staticmethod
    def __track_key_norm_mean(context: _AttentionTrackingContext) -> None:
        key_norm_mean = context.metrics.key_norm_mean
        if key_norm_mean is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/k_norm_mean",
                key_norm_mean,
            )

    @staticmethod
    def __track_value_norm_mean(context: _AttentionTrackingContext) -> None:
        value_norm_mean = context.metrics.value_norm_mean
        if value_norm_mean is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/v_norm_mean",
                value_norm_mean,
            )

    @staticmethod
    def __track_output_norm(context: _AttentionTrackingContext) -> None:
        output_norm = context.metrics.output_norm
        if output_norm is not None:
            context.pl_module.log(f"{context.metric_prefix}/output_norm", output_norm)

    @staticmethod
    def __track_auxiliary_loss(context: _AttentionTrackingContext) -> None:
        auxiliary_loss = context.metrics.auxiliary_loss
        if auxiliary_loss is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/auxiliary_loss",
                auxiliary_loss,
            )

    @staticmethod
    def __track_configured_dropout_probability(
        context: _AttentionTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.metric_prefix}/configured_dropout_probability",
            context.metrics.configured_dropout_probability,
        )

    @staticmethod
    def __track_mask_coverage(context: _AttentionTrackingContext) -> None:
        context.pl_module.log(
            f"{context.metric_prefix}/mask_coverage",
            context.metrics.mask_coverage,
        )

    @staticmethod
    def __track_entropy_mean(context: _AttentionTrackingContext) -> None:
        per_head_entropy = context.metrics.per_head_entropy
        if per_head_entropy is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/"
                f"{AttentionMonitorCallback.__weight_metric_prefix(context)}"
                "entropy_mean",
                per_head_entropy.mean(),
            )

    @staticmethod
    def __track_max_probability_mean(context: _AttentionTrackingContext) -> None:
        per_head_max_probability = context.metrics.per_head_max_probability
        if per_head_max_probability is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/"
                f"{AttentionMonitorCallback.__weight_metric_prefix(context)}"
                "max_probability_mean",
                per_head_max_probability.mean(),
            )

    def __track_dead_head_fraction(self, context: _AttentionTrackingContext) -> None:
        per_head_entropy = context.metrics.per_head_entropy
        if per_head_entropy is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/{self.__weight_metric_prefix(context)}"
                "dead_head_fraction",
                (per_head_entropy <= self.DEAD_HEAD_ENTROPY_FLOOR).float().mean(),
            )

    def __track_per_head_entropy(self, context: _AttentionTrackingContext) -> None:
        per_head_entropy = context.metrics.per_head_entropy
        if not self.log_per_head_scalars or per_head_entropy is None:
            return
        metric_name = f"{self.__weight_metric_prefix(context)}entropy"
        for head_index, entropy in enumerate(per_head_entropy):
            context.pl_module.log(
                f"{context.metric_prefix}/head_{head_index}/{metric_name}",
                entropy,
            )

    def __track_per_head_max_probability(
        self,
        context: _AttentionTrackingContext,
    ) -> None:
        per_head_max_probability = context.metrics.per_head_max_probability
        if not self.log_per_head_scalars or per_head_max_probability is None:
            return
        metric_name = f"{self.__weight_metric_prefix(context)}max_probability"
        for head_index, max_probability in enumerate(per_head_max_probability):
            context.pl_module.log(
                f"{context.metric_prefix}/head_{head_index}/{metric_name}",
                max_probability,
            )

    def __track_entropy_history(self, context: _AttentionTrackingContext) -> None:
        if not self.__can_track_exact_weight_history(context):
            return
        self._entropy_history[context.module_name].append(
            context.metrics.per_head_entropy
        )

    def __track_max_probability_history(
        self,
        context: _AttentionTrackingContext,
    ) -> None:
        if not self.__can_track_exact_weight_history(context):
            return
        self._max_probability_history[context.module_name].append(
            context.metrics.per_head_max_probability
        )

    def __track_entropy_histogram(self, context: _AttentionTrackingContext) -> None:
        if not self.__can_emit_exact_weight_visual(context):
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.metric_prefix}/histogram/entropy_by_head",
            context.metrics.per_head_entropy,
            context.global_step,
        )

    def __track_entropy_heatmap(self, context: _AttentionTrackingContext) -> None:
        if not self.__can_emit_exact_weight_visual(context):
            return
        self._emission_policy.emit_history_heatmap(
            context.experiment,
            f"{context.metric_prefix}/heatmap/entropy_by_head",
            self._entropy_history[context.module_name],
            context.global_step,
        )

    def __track_max_probability_histogram(
        self,
        context: _AttentionTrackingContext,
    ) -> None:
        if not self.__can_emit_exact_weight_visual(context):
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.metric_prefix}/histogram/max_probability_by_head",
            context.metrics.per_head_max_probability,
            context.global_step,
        )

    def __track_max_probability_heatmap(
        self,
        context: _AttentionTrackingContext,
    ) -> None:
        if not self.__can_emit_exact_weight_visual(context):
            return
        self._emission_policy.emit_history_heatmap(
            context.experiment,
            f"{context.metric_prefix}/heatmap/max_probability_by_head",
            self._max_probability_history[context.module_name],
            context.global_step,
        )

    @staticmethod
    def __track_dropout_zero_fraction(context: _AttentionTrackingContext) -> None:
        dropout_zero_fraction = context.metrics.dropout_zero_fraction
        if dropout_zero_fraction is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/dropout_zero_fraction",
                dropout_zero_fraction,
            )

    @staticmethod
    def __weight_metric_prefix(context: _AttentionTrackingContext) -> str:
        return "approximate_" if context.metrics.weight_source == "approximate" else ""

    def __can_track_exact_weight_history(
        self,
        context: _AttentionTrackingContext,
    ) -> bool:
        return (
            context.metrics.weight_source == "exact"
            and context.metrics.per_head_entropy is not None
            and context.metrics.per_head_max_probability is not None
            and context.module_name in self._entropy_history
            and context.module_name in self._max_probability_history
        )

    def __can_emit_exact_weight_visual(
        self,
        context: _AttentionTrackingContext,
    ) -> bool:
        return context.experiment is not None and self.__can_track_exact_weight_history(
            context
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
        self._tracker_manager.detach()
        self._entropy_history.clear()
        self._max_probability_history.clear()
        self._emission_policy.clear()

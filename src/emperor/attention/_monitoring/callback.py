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
from emperor.attention._runtime import MultiHeadAttentionInputs
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
class _AttentionMethodObserver:
    """Observe one method call without changing its arguments or result."""

    before_call: Callable[[tuple[object, ...], dict[str, object]], object]
    after_call: Callable[[object, object], None]


class _AttentionMethodProbe:
    """Own one installed method wrapper and all of its observers."""

    def __init__(self, owner: object, method_name: str) -> None:
        original_method = getattr(owner, method_name, None)
        if not callable(original_method):
            raise AttributeError(
                f"{type(owner).__name__}.{method_name} must be callable for "
                "attention monitoring."
            )

        local_attributes = vars(owner)
        self.owner = owner
        self.method_name = method_name
        self._original_method = original_method
        self._had_local_attribute = method_name in local_attributes
        self._original_local_attribute = local_attributes.get(method_name)
        self._observers: dict[int, _AttentionMethodObserver] = {}
        self._next_observer_id = 0
        setattr(owner, method_name, self.__call_original_and_notify_observers)

    @property
    def observer_count(self) -> int:
        return len(self._observers)

    def add_observer(self, observer: _AttentionMethodObserver) -> int:
        observer_id = self._next_observer_id
        self._next_observer_id += 1
        self._observers[observer_id] = observer
        return observer_id

    def remove_observer(self, observer_id: int) -> None:
        self._observers.pop(observer_id, None)

    def restore_original_method(self) -> None:
        if self._had_local_attribute:
            setattr(
                self.owner,
                self.method_name,
                self._original_local_attribute,
            )
            return
        delattr(self.owner, self.method_name)

    def __call_original_and_notify_observers(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        observer_calls = tuple(
            (observer, observer.before_call(args, kwargs))
            for observer in tuple(self._observers.values())
        )
        result = self._original_method(*args, **kwargs)
        for observer, call in observer_calls:
            observer.after_call(call, result)
        return result


class _AttentionMethodInstrumentation:
    """Share method probes and restore them after their final subscription."""

    def __init__(self) -> None:
        self._probes: dict[tuple[int, str], _AttentionMethodProbe] = {}

    @property
    def probe_count(self) -> int:
        return len(self._probes)

    def subscribe(
        self,
        owner: object,
        method_name: str,
        observer: _AttentionMethodObserver,
    ) -> Callable[[], None]:
        probe_key = (id(owner), method_name)
        probe = self._probes.get(probe_key)
        if probe is None:
            probe = _AttentionMethodProbe(owner, method_name)
            self._probes[probe_key] = probe
        # A live probe strongly references its owner, so Python cannot reuse
        # that owner's identity for a different object until the probe is gone.

        observer_id = probe.add_observer(observer)
        removed = False

        def remove_subscription() -> None:
            nonlocal removed
            if removed:
                return
            removed = True
            probe.remove_observer(observer_id)
            if probe.observer_count > 0:
                return
            probe.restore_original_method()
            self._probes.pop(probe_key, None)

        return remove_subscription


_ATTENTION_METHOD_INSTRUMENTATION = _AttentionMethodInstrumentation()


class _AttentionDiagnosticsTracker:
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self._latest_observation = _AttentionObservation()

    @property
    def latest_observation(self) -> _AttentionObservation:
        return self._latest_observation

    def begin_observation(self) -> None:
        self._latest_observation = _AttentionObservation()

    def record_projected_inputs(self, projected_inputs: object) -> None:
        detached_inputs = self.__detach_attention_inputs(projected_inputs)
        if detached_inputs is not None:
            self._latest_observation.projected_inputs = detached_inputs

    def record_processor_inputs(self, processor_inputs: object) -> None:
        self._latest_observation.processor_inputs = self.__detach_attention_inputs(
            processor_inputs
        )

    def record_raw_attention_logits(self, raw_attention_logits: object) -> None:
        detached_logits = self.__detach_tensor(raw_attention_logits)
        if detached_logits is not None:
            self._latest_observation.raw_attention_logits = detached_logits

    def record_normalized_attention_weights(
        self,
        normalized_attention_weights: object,
    ) -> None:
        detached_weights = self.__detach_tensor(normalized_attention_weights)
        if detached_weights is not None:
            self._latest_observation.normalized_attention_weights = detached_weights

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
    def __detach_attention_inputs(
        cls,
        value: object,
    ) -> MultiHeadAttentionInputs | None:
        if not isinstance(value, MultiHeadAttentionInputs):
            return None
        return MultiHeadAttentionInputs(
            query=value.query.detach(),
            key=value.key.detach(),
            value=value.value.detach(),
            merged_attention_mask=cls.__detach_tensor(value.merged_attention_mask),
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

    def __init__(
        self,
        method_instrumentation: _AttentionMethodInstrumentation = (
            _ATTENTION_METHOD_INSTRUMENTATION
        ),
    ) -> None:
        self._method_instrumentation = method_instrumentation
        self._trackers: dict[int, _AttentionDiagnosticsTracker] = {}
        self._hook_handles: list[RemovableHandle] = []
        self._method_subscriptions: list[Callable[[], None]] = []

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(tracker.module_name for tracker in self._trackers.values())

    @property
    def hook_count(self) -> int:
        return len(self._hook_handles)

    @property
    def subscription_count(self) -> int:
        return len(self._method_subscriptions)

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
        monitor_adapter: _AttentionMonitorAdapter | None = None,
    ) -> None:
        tracker_key = id(attention_module)
        initial_subscription_count = len(self._method_subscriptions)
        tracker = _AttentionDiagnosticsTracker(module_name)
        self._trackers[tracker_key] = tracker
        try:
            resolved_monitor_adapter = (
                monitor_adapter or _resolve_attention_monitor_adapter(attention_module)
            )
            projector_attached = self.__attach_projector(
                attention_module,
                tracker,
                should_capture,
            )
            processor_attached = self.__attach_processor(
                attention_module,
                tracker,
                should_capture,
                resolved_monitor_adapter,
                begin_observation=not projector_attached,
            )
            self._hook_handles.append(
                attention_module.register_forward_hook(
                    self.__make_forward_hook(
                        attention_module,
                        tracker,
                        should_capture,
                        observation_recorder,
                        begin_observation=(
                            not projector_attached and not processor_attached
                        ),
                    )
                )
            )
        except BaseException:
            self.__rollback_attachment(
                tracker_key,
                initial_subscription_count,
            )
            raise

    def detach(self) -> None:
        for hook_handle in self._hook_handles:
            hook_handle.remove()
        self._hook_handles.clear()
        for remove_subscription in reversed(self._method_subscriptions):
            remove_subscription()
        self._method_subscriptions.clear()
        self._trackers.clear()

    def __attach_projector(
        self,
        attention_module: Module,
        tracker: _AttentionDiagnosticsTracker,
        should_capture: Callable[[], bool],
    ) -> bool:
        projector = getattr(attention_module, "projector", None)
        method_name = "compute_qkv_projections"
        if not callable(getattr(projector, method_name, None)):
            return False

        def begin_projected_input_capture(
            _args: tuple[object, ...],
            _kwargs: dict[str, object],
        ) -> bool:
            capture_this_forward = should_capture()
            if capture_this_forward:
                tracker.begin_observation()
            return capture_this_forward

        def capture_projected_inputs(
            capture_this_forward: object,
            projected_inputs: object,
        ) -> None:
            if capture_this_forward:
                tracker.record_projected_inputs(projected_inputs)

        self.__observe_method(
            projector,
            method_name,
            _AttentionMethodObserver(
                before_call=begin_projected_input_capture,
                after_call=capture_projected_inputs,
            ),
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
        if not callable(getattr(processor, method_name, None)):
            return False

        def capture_processor_inputs(
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> None:
            if should_capture():
                if begin_observation:
                    tracker.begin_observation()
                processor_inputs = args[0] if args else kwargs.get("attention_inputs")
                tracker.record_processor_inputs(processor_inputs)

        self.__observe_method(
            processor,
            method_name,
            _AttentionMethodObserver(
                before_call=capture_processor_inputs,
                after_call=self.__ignore_method_result,
            ),
        )
        self.__attach_attention_weight_methods(
            processor,
            tracker,
            should_capture,
            monitor_adapter,
        )
        return True

    def __attach_attention_weight_methods(
        self,
        processor: object,
        tracker: _AttentionDiagnosticsTracker,
        should_capture: Callable[[], bool],
        monitor_adapter: _AttentionMonitorAdapter,
    ) -> None:
        observed_results = (
            (
                monitor_adapter.raw_attention_logit_method_name,
                tracker.record_raw_attention_logits,
            ),
            (
                monitor_adapter.normalized_attention_weight_method_name,
                tracker.record_normalized_attention_weights,
            ),
            (
                monitor_adapter.exact_weight_method_name,
                tracker.record_exact_attention_weights,
            ),
        )
        for method_name, record_result in observed_results:
            self.__attach_tensor_result_method(
                processor,
                method_name,
                should_capture,
                record_result,
            )

    def __attach_tensor_result_method(
        self,
        processor: object,
        method_name: str | None,
        should_capture: Callable[[], bool],
        record_result: Callable[[object], None],
    ) -> None:
        if method_name is None:
            return

        def capture_tensor_result(
            capture_this_forward: object,
            result: object,
        ) -> None:
            if capture_this_forward:
                record_result(result)

        self.__observe_method(
            processor,
            method_name,
            _AttentionMethodObserver(
                before_call=lambda _args, _kwargs: should_capture(),
                after_call=capture_tensor_result,
            ),
        )

    @staticmethod
    def __ignore_method_result(_call: object, _result: object) -> None:
        return None

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

    def __observe_method(
        self,
        owner: object,
        method_name: str,
        observer: _AttentionMethodObserver,
    ) -> None:
        self._method_subscriptions.append(
            self._method_instrumentation.subscribe(owner, method_name, observer)
        )

    def __rollback_attachment(
        self,
        tracker_key: int,
        initial_subscription_count: int,
    ) -> None:
        # The forward hook is the final successful action in attach(), so no
        # manager-owned hook can exist when setup enters this rollback path.
        for remove_subscription in reversed(
            self._method_subscriptions[initial_subscription_count:]
        ):
            remove_subscription()
        del self._method_subscriptions[initial_subscription_count:]
        self._trackers.pop(tracker_key, None)


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
        from emperor.attention._variants.mixer.layer import MixerAttention

        attention_runtime_types = (MultiHeadAttentionAbstract, MixerAttention)

        self.__cleanup()
        try:
            for module_name, attention_module in pl_module.named_modules():
                if isinstance(attention_module, attention_runtime_types):
                    self.__attach_attention_module(
                        module_name,
                        attention_module,
                        pl_module,
                    )
        except BaseException:
            self.__cleanup()
            raise

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
        self.__track_finite_raw_attention_logit_mean(context)
        self.__track_finite_raw_attention_logit_std(context)
        self.__track_pre_dropout_entropy_mean(context)
        self.__track_pre_dropout_max_probability_mean(context)
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
    def __track_finite_raw_attention_logit_mean(
        context: _AttentionTrackingContext,
    ) -> None:
        mean = context.metrics.finite_raw_attention_logit_mean
        if mean is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/finite_raw_logit_mean",
                mean,
            )

    @staticmethod
    def __track_finite_raw_attention_logit_std(
        context: _AttentionTrackingContext,
    ) -> None:
        standard_deviation = context.metrics.finite_raw_attention_logit_std
        if standard_deviation is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/finite_raw_logit_std",
                standard_deviation,
            )

    @staticmethod
    def __track_pre_dropout_entropy_mean(
        context: _AttentionTrackingContext,
    ) -> None:
        per_head_entropy = context.metrics.pre_dropout_per_head_entropy
        if per_head_entropy is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/pre_dropout_entropy_mean",
                per_head_entropy.mean(),
            )

    @staticmethod
    def __track_pre_dropout_max_probability_mean(
        context: _AttentionTrackingContext,
    ) -> None:
        per_head_maximum = context.metrics.pre_dropout_per_head_max_probability
        if per_head_maximum is not None:
            context.pl_module.log(
                f"{context.metric_prefix}/pre_dropout_max_probability_mean",
                per_head_maximum.mean(),
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

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from lightning.pytorch.callbacks import Callback

from model_runtime.runs._metrics import portable_metric_values
from model_runtime.runs._progress_events import (
    ClusterInitializedEvent,
    EpochStartedEvent,
    FitCompletedEvent,
    NeuronAddedEvent,
    NeuronsAddedEvent,
    StepEvent,
    TestCompletedEvent,
    ValidationEvent,
)
from model_runtime.runs.progress import ContextualRunProgress

CLUSTER_COORDINATE_SAMPLE_LIMIT = 100
NEURON_ADDED_BURST_LIMIT = 100


def _coordinate_from_neuron_name(name: str) -> list[int] | None:
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != "neuron":
        return None
    try:
        return [int(parts[1]), int(parts[2]), int(parts[3])]
    except ValueError:
        return None


class _LightningRunProgressAdapter(Callback):
    """Translate Lightning lifecycle hooks to portable Run events."""

    def __init__(
        self,
        progress: ContextualRunProgress,
        *,
        step_interval: int,
    ) -> None:
        super().__init__()
        self._progress = progress
        self._step_interval = max(1, int(step_interval))
        self._clusters: list[tuple[str, Any]] = []
        self._known_names: dict[str, set[str]] = {}

    @staticmethod
    def _metrics(trainer: Any) -> dict[str, Any]:
        metrics: object = getattr(trainer, "callback_metrics", {})
        if not isinstance(metrics, Mapping):
            return {}
        return portable_metric_values(cast(Mapping[Any, Any], metrics))

    @staticmethod
    def _capacity(cluster: Any) -> list[int]:
        return [
            cluster.x_axis_total_neurons,
            cluster.y_axis_total_neurons,
            cluster.z_axis_total_neurons,
        ]

    @staticmethod
    def _coordinates(names: set[str]) -> list[list[int]]:
        coordinates = (_coordinate_from_neuron_name(name) for name in names)
        return sorted(
            coordinate for coordinate in coordinates if coordinate is not None
        )

    def _coordinate_sample_payload(self, names: set[str]) -> dict[str, Any]:
        coordinates = self._coordinates(names)
        sampled = coordinates[:CLUSTER_COORDINATE_SAMPLE_LIMIT]
        return {
            "coordinates": sampled,
            "coordinateCount": len(coordinates),
            "coordinatesTruncated": len(coordinates) > len(sampled),
        }

    def _cluster_initialized_event(
        self,
        name: str,
        cluster: Any,
        names: set[str],
    ) -> ClusterInitializedEvent:
        count = len(names)
        capacity = self._capacity(cluster)
        coordinate_sample = self._coordinate_sample_payload(names)
        return ClusterInitializedEvent(
            node=name,
            count=count,
            capacity=capacity,
            coordinates=coordinate_sample["coordinates"],
            coordinate_count=coordinate_sample["coordinateCount"],
            coordinates_truncated=coordinate_sample["coordinatesTruncated"],
        )

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        self._clear_growth_state()
        from emperor.neuron import NeuronCluster

        try:
            self._clusters.extend(
                (name, module)
                for name, module in pl_module.named_modules()
                if isinstance(module, NeuronCluster)
            )
            for name, cluster in self._clusters:
                names = set(cluster.cluster.keys())
                self._known_names[name] = names
                self._progress.write_event(
                    self._cluster_initialized_event(name, cluster, names)
                )
        except BaseException:
            self._clear_growth_state()
            raise

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._progress.write_event(
            EpochStartedEvent(
                epoch=int(trainer.current_epoch),
                step=int(trainer.global_step),
            )
        )

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        global_step = int(trainer.global_step)
        if self._step_interval == 1 or global_step % self._step_interval == 0:
            self._progress.write_event(
                StepEvent(
                    epoch=int(trainer.current_epoch),
                    step=global_step,
                    batch=int(batch_idx),
                    metrics=self._metrics(trainer),
                )
            )
        self._emit_neuron_growth(trainer)

    def _emit_neuron_growth(self, trainer: Any) -> None:
        for name, cluster in self._clusters:
            current = set(cluster.cluster.keys())
            new_names = current - self._known_names.get(name, set())
            self._known_names[name] = current
            coordinates = [
                coordinate
                for coordinate in (
                    _coordinate_from_neuron_name(new_name)
                    for new_name in sorted(new_names)
                )
                if coordinate is not None
            ]
            if not coordinates:
                continue
            count = len(current)
            capacity = self._capacity(cluster)
            epoch = int(getattr(trainer, "current_epoch", 0))
            step = int(getattr(trainer, "global_step", 0))
            if len(coordinates) > NEURON_ADDED_BURST_LIMIT:
                self._progress.write_event(
                    NeuronsAddedEvent(
                        coordinates=coordinates[:CLUSTER_COORDINATE_SAMPLE_LIMIT],
                        coordinate_count=len(coordinates),
                        coordinates_truncated=(
                            len(coordinates) > CLUSTER_COORDINATE_SAMPLE_LIMIT
                        ),
                        node=name,
                        count=count,
                        capacity=capacity,
                        epoch=epoch,
                        step=step,
                    )
                )
                continue
            for coordinate in coordinates:
                self._progress.write_event(
                    NeuronAddedEvent(
                        coord=coordinate,
                        node=name,
                        count=count,
                        capacity=capacity,
                        epoch=epoch,
                        step=step,
                    )
                )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        self._progress.write_event(
            ValidationEvent(
                epoch=int(trainer.current_epoch),
                step=int(trainer.global_step),
                metrics=self._metrics(trainer),
            )
        )

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        try:
            self._progress.write_event(
                FitCompletedEvent(
                    epoch=int(trainer.current_epoch),
                    step=int(trainer.global_step),
                    metrics=self._metrics(trainer),
                )
            )
        finally:
            self._clear_growth_state()

    def on_exception(
        self,
        trainer: Any,
        pl_module: Any,
        exception: BaseException,
    ) -> None:
        self._clear_growth_state()

    def _clear_growth_state(self) -> None:
        self._clusters.clear()
        self._known_names.clear()

    def on_test_end(self, trainer: Any, pl_module: Any) -> None:
        self._progress.write_event(
            TestCompletedEvent(
                epoch=int(trainer.current_epoch),
                step=int(trainer.global_step),
                metrics=self._metrics(trainer),
            )
        )


def lightning_progress_adapter(
    progress: ContextualRunProgress,
    *,
    step_interval: int,
) -> Callback:
    return _LightningRunProgressAdapter(
        progress,
        step_interval=step_interval,
    )


__all__ = ["lightning_progress_adapter"]

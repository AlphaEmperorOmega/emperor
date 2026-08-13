from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from heapq import nsmallest
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
    def _coordinate_sample(
        names: Iterable[str],
        *,
        maximum: int,
        order_by_name: bool,
    ) -> tuple[list[list[int]], int]:
        coordinate_count = 0

        def coordinate_entries() -> Iterator[tuple[str, list[int]]]:
            nonlocal coordinate_count
            for name in names:
                coordinate = _coordinate_from_neuron_name(name)
                if coordinate is None:
                    continue
                coordinate_count += 1
                yield name, coordinate

        entries = coordinate_entries()
        sampled_entries = (
            nsmallest(maximum, entries, key=lambda entry: entry[0])
            if order_by_name
            else nsmallest(maximum, entries, key=lambda entry: tuple(entry[1]))
        )
        return [coordinate for _name, coordinate in sampled_entries], coordinate_count

    def _coordinate_sample_payload(self, names: set[str]) -> dict[str, Any]:
        sampled, coordinate_count = self._coordinate_sample(
            names,
            maximum=CLUSTER_COORDINATE_SAMPLE_LIMIT,
            order_by_name=False,
        )
        return {
            "coordinates": sampled,
            "coordinateCount": coordinate_count,
            "coordinatesTruncated": coordinate_count > len(sampled),
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
            previous = self._known_names.get(name)
            self._known_names[name] = current
            coordinates, coordinate_count = self._coordinate_sample(
                (
                    current_name
                    for current_name in current
                    if previous is None or current_name not in previous
                ),
                maximum=max(
                    CLUSTER_COORDINATE_SAMPLE_LIMIT,
                    NEURON_ADDED_BURST_LIMIT,
                ),
                order_by_name=True,
            )
            if coordinate_count == 0:
                continue
            count = len(current)
            capacity = self._capacity(cluster)
            epoch = int(getattr(trainer, "current_epoch", 0))
            step = int(getattr(trainer, "global_step", 0))
            if coordinate_count > NEURON_ADDED_BURST_LIMIT:
                sampled_coordinates = coordinates[:CLUSTER_COORDINATE_SAMPLE_LIMIT]
                self._progress.write_event(
                    NeuronsAddedEvent(
                        coordinates=sampled_coordinates,
                        coordinate_count=coordinate_count,
                        coordinates_truncated=(
                            coordinate_count > len(sampled_coordinates)
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

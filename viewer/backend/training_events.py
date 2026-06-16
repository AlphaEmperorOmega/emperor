from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lightning.pytorch.callbacks import Callback

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


class NeuronClusterGrowthCallback(Callback):
    """Writes a progress event whenever a neuron cluster instantiates a new
    neuron, so the viewer can report growth (with coordinates) over time.

    Emits ``cluster_initialized`` once per fit and ``neuron_added`` per new
    neuron. Models without a neuron cluster produce no events.
    """

    def __init__(self, write_event: Callable[[dict[str, Any]], None]) -> None:
        super().__init__()
        self._write_event = write_event
        self._clusters: list[tuple[str, Any]] = []
        self._known_names: dict[str, set[str]] = {}

    def on_fit_start(self, trainer, pl_module) -> None:
        from emperor.neuron.core.model import NeuronCluster

        self._clusters = [
            (name, module)
            for name, module in pl_module.named_modules()
            if isinstance(module, NeuronCluster)
        ]
        for name, cluster in self._clusters:
            names = set(cluster.cluster.keys())
            self._known_names[name] = names
            self._write_event(
                {
                    "type": "cluster_initialized",
                    "node": name,
                    "count": len(names),
                    "capacity": self.__capacity(cluster),
                    **self.__coordinate_sample_payload(names),
                }
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        for name, cluster in self._clusters:
            current = set(cluster.cluster.keys())
            new_names = current - self._known_names.get(name, set())
            if not new_names:
                continue
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
            epoch = int(getattr(trainer, "current_epoch", 0))
            step = int(getattr(trainer, "global_step", 0))
            if len(coordinates) > NEURON_ADDED_BURST_LIMIT:
                self._write_event(
                    {
                        "type": "neurons_added",
                        "node": name,
                        "coordinates": coordinates[:CLUSTER_COORDINATE_SAMPLE_LIMIT],
                        "coordinateCount": len(coordinates),
                        "coordinatesTruncated": (
                            len(coordinates) > CLUSTER_COORDINATE_SAMPLE_LIMIT
                        ),
                        "count": len(current),
                        "capacity": self.__capacity(cluster),
                        "epoch": epoch,
                        "step": step,
                    }
                )
                continue
            for coordinate in coordinates:
                if coordinate is None:
                    continue
                self._write_event(
                    {
                        "type": "neuron_added",
                        "node": name,
                        "coord": coordinate,
                        "count": len(current),
                        "capacity": self.__capacity(cluster),
                        "epoch": epoch,
                        "step": step,
                    }
                )

    def __capacity(self, cluster) -> list[int]:
        return [
            cluster.x_axis_total_neurons,
            cluster.y_axis_total_neurons,
            cluster.z_axis_total_neurons,
        ]

    def __coordinates(self, names: set[str]) -> list[list[int]]:
        coordinates = (_coordinate_from_neuron_name(name) for name in names)
        return sorted(
            coordinate for coordinate in coordinates if coordinate is not None
        )

    def __coordinate_sample_payload(self, names: set[str]) -> dict[str, Any]:
        coordinates = self.__coordinates(names)
        sampled = coordinates[:CLUSTER_COORDINATE_SAMPLE_LIMIT]
        return {
            "coordinates": sampled,
            "coordinateCount": len(coordinates),
            "coordinatesTruncated": len(coordinates) > len(sampled),
        }

    def on_fit_end(self, trainer, pl_module) -> None:
        self._clusters = []
        self._known_names.clear()

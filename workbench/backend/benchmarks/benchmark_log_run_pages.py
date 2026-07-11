"""Reproducible synthetic benchmark for paginated log-run projections."""

from __future__ import annotations

import json
from statistics import median
from time import perf_counter
from typing import Any
from unittest.mock import patch

from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.run_history import RunHistoryService

PAGE_SIZE = 100
SAMPLE_COUNT = 7


class SyntheticRun:
    def __init__(self, index: int) -> None:
        self.id = f"run-{index}"
        self.group = "benchmark"
        self._experiment = "benchmark"
        self.modelType = "linears"
        self.model = "linears/linear"
        self.preset = "BASELINE"
        self.dataset = f"Dataset-{index % 6}"
        self.runName = self.id
        self.timestamp = None
        self.version = "version_0"
        self.relativePath = f"benchmark/{self.id}/version_0"
        self.hasResult = True
        self.eventFileCount = 1
        self.checkpointCount = 0
        self.hasHparams = True
        self.metrics = {f"metric/{metric}": metric / 25 for metric in range(25)}
        self.response_count = 0
        self.experiment_read_count = 0

    @property
    def experiment(self) -> str:
        self.experiment_read_count += 1
        return self._experiment

    def to_response(self) -> dict[str, Any]:
        self.response_count += 1
        return {
            "id": self.id,
            "group": self.group,
            "experiment": self._experiment,
            "modelType": self.modelType,
            "model": "linear",
            "preset": self.preset,
            "dataset": self.dataset,
            "runName": self.runName,
            "timestamp": self.timestamp,
            "version": self.version,
            "relativePath": self.relativePath,
            "hasResult": self.hasResult,
            "eventFileCount": self.eventFileCount,
            "checkpointCount": self.checkpointCount,
            "hasHparams": self.hasHparams,
            "metrics": dict(self.metrics),
        }


def _service() -> RunHistoryService:
    return RunHistoryService(
        logs_root="logs",
        mutation_coordinator=LogExperimentMutationCoordinator(),
        active_log_writers=lambda: (),
    )


def baseline_full_response(runs: list[SyntheticRun]) -> dict[str, Any]:
    serialized = []
    for run in runs:
        response = run.to_response()
        response["hasLayerMonitorData"] = None
        serialized.append(response)
    return {
        "runs": serialized,
        "total": len(serialized),
        "limit": PAGE_SIZE,
        "offset": 0,
        "hasMore": False,
    }


def measure(run_count: int, *, current_all_pages: bool) -> dict[str, float]:
    durations: list[float] = []
    bytes_per_sample: list[int] = []
    serialized_per_sample: list[int] = []
    filter_reads_per_sample: list[int] = []

    for _sample in range(SAMPLE_COUNT):
        runs = [SyntheticRun(index) for index in range(run_count)]
        started_at = perf_counter()
        if current_all_pages:
            service = _service()
            with patch.object(service._scanner, "list_runs", return_value=runs):
                responses = []
                for offset in range(0, run_count, PAGE_SIZE):
                    responses.append(
                        service.list_runs(
                            limit=PAGE_SIZE,
                            offset=offset,
                            experiment=["benchmark"],
                            projection="summary",
                        )
                    )
        else:
            responses = [baseline_full_response(runs)]
        encoded = [
            json.dumps(response, separators=(",", ":")).encode()
            for response in responses
        ]
        durations.append((perf_counter() - started_at) * 1_000)
        bytes_per_sample.append(sum(map(len, encoded)))
        serialized_per_sample.append(sum(run.response_count for run in runs))
        filter_reads_per_sample.append(sum(run.experiment_read_count for run in runs))

    return {
        "ms": median(durations),
        "bytes": median(bytes_per_sample),
        "serialized": median(serialized_per_sample),
        "filter_reads": median(filter_reads_per_sample),
    }


def measure_current_first_page(run_count: int) -> dict[str, float]:
    durations: list[float] = []
    bytes_per_sample: list[int] = []
    serialized_per_sample: list[int] = []
    filter_reads_per_sample: list[int] = []
    for _sample in range(SAMPLE_COUNT):
        runs = [SyntheticRun(index) for index in range(run_count)]
        service = _service()
        started_at = perf_counter()
        with patch.object(service._scanner, "list_runs", return_value=runs):
            response = service.list_runs(
                limit=PAGE_SIZE,
                offset=0,
                experiment=["benchmark"],
                projection="summary",
            )
        encoded = json.dumps(response, separators=(",", ":")).encode()
        durations.append((perf_counter() - started_at) * 1_000)
        bytes_per_sample.append(len(encoded))
        serialized_per_sample.append(sum(run.response_count for run in runs))
        filter_reads_per_sample.append(sum(run.experiment_read_count for run in runs))
    return {
        "ms": median(durations),
        "bytes": median(bytes_per_sample),
        "serialized": median(serialized_per_sample),
        "filter_reads": median(filter_reads_per_sample),
    }


def main() -> None:
    print(
        "Synthetic log-run listing benchmark "
        f"(page={PAGE_SIZE}, median of {SAMPLE_COUNT})"
    )
    heading = (
        "runs | baseline full ms/bytes/serialized | "
        "current first-page ms/bytes/serialized | "
        "current all-pages ms/bytes/serialized/filter-reads"
    )
    print(heading)
    for run_count in (5, 100, 500):
        baseline = measure(run_count, current_all_pages=False)
        first_page = measure_current_first_page(run_count)
        all_pages = measure(run_count, current_all_pages=True)
        baseline_result = (
            f"{baseline['ms']:.2f}/{baseline['bytes']:.0f}/{baseline['serialized']:.0f}"
        )
        first_page_result = (
            f"{first_page['ms']:.2f}/{first_page['bytes']:.0f}/"
            f"{first_page['serialized']:.0f}"
        )
        print(
            f"{run_count:4d} | "
            f"{baseline_result} | "
            f"{first_page_result} | "
            f"{all_pages['ms']:.2f}/{all_pages['bytes']:.0f}/"
            f"{all_pages['serialized']:.0f}/{all_pages['filter_reads']:.0f}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter

from emperor_workbench.log_experiments import LogExperimentMutationCoordinator
from emperor_workbench.run_history import RunHistoryService

DEFAULT_RUN_COUNTS = (100, 500, 1_000)
SAMPLE_COUNT = 5


def _write_runs(logs_root: Path, run_count: int) -> None:
    for index in range(run_count):
        run_dir = logs_root.joinpath(
            "benchmark",
            "linears",
            "linear",
            "BASELINE",
            f"Dataset-{index % 8}",
            f"run_20260716_{index:06d}",
            "version_0",
        )
        run_dir.mkdir(parents=True)
        run_dir.joinpath("result.json").write_text(
            json.dumps({"metrics": {"validation/accuracy": index / run_count}}),
            encoding="utf-8",
        )


def _service(logs_root: Path, state_root: Path) -> RunHistoryService:
    return RunHistoryService(
        logs_root=logs_root,
        state_root=state_root,
        mutation_coordinator=LogExperimentMutationCoordinator(),
        active_log_writers=lambda: (),
        model_identity_resolver=(
            lambda value: "linears/linear"
            if value in {"linear", "linears/linear"}
            else None
        ),
    )


def measure(run_count: int) -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        logs_root = root / "logs"
        _write_runs(logs_root, run_count)
        cold_samples: list[float] = []
        warm_samples: list[float] = []
        for _sample in range(SAMPLE_COUNT):
            service = _service(logs_root, root / "state")
            started = perf_counter()
            page = service.list_runs(limit=100, offset=0, projection="summary")
            cold_samples.append((perf_counter() - started) * 1_000)
            if page.total != run_count:
                raise RuntimeError(f"Expected {run_count} Runs, observed {page.total}.")
            started = perf_counter()
            service.list_runs(limit=100, offset=0, projection="summary")
            warm_samples.append((perf_counter() - started) * 1_000)
        return median(cold_samples), median(warm_samples)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark Run History pages.")
    parser.add_argument(
        "run_counts",
        nargs="*",
        type=int,
        default=list(DEFAULT_RUN_COUNTS),
    )
    arguments = parser.parse_args(argv)
    print("runs | cold first-page ms | warm first-page ms")
    for run_count in arguments.run_counts:
        cold_ms, warm_ms = measure(run_count)
        print(f"{run_count:4d} | {cold_ms:18.2f} | {warm_ms:18.2f}")


if __name__ == "__main__":
    main()

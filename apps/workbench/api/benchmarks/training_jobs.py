from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter

from emperor_workbench.log_experiments import LogExperimentMutationCoordinator
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_plans import RunPlanService
from emperor_workbench.training_jobs import TrainingJobService

SAMPLE_COUNT = 7


def _service(root: Path, adapter: ProjectAdapterClient) -> TrainingJobService:
    return TrainingJobService(
        root=root / "jobs",
        logs_root=root / "logs",
        cancellation_mode="process-group",
        mutation_coordinator=LogExperimentMutationCoordinator(),
        run_plans=RunPlanService(
            model_packages=ModelPackageCatalog(adapter),
        ),
    )


def measure(iterations: int) -> float:
    samples: list[float] = []
    with (
        tempfile.TemporaryDirectory() as directory,
        ProjectAdapterClient(persistent=False) as adapter,
    ):
        service = _service(Path(directory), adapter)
        for _sample in range(SAMPLE_COUNT):
            started = perf_counter()
            for _iteration in range(iterations):
                if service.active_jobs():
                    raise RuntimeError("Fresh benchmark service has active jobs.")
            samples.append((perf_counter() - started) * 1_000)
    return median(samples)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark empty Training Job status reads."
    )
    parser.add_argument("--iterations", type=int, default=10_000)
    arguments = parser.parse_args(argv)
    duration_ms = measure(arguments.iterations)
    print(
        f"{arguments.iterations} active-job reads: {duration_ms:.2f} ms "
        f"({duration_ms / arguments.iterations:.4f} ms/read)"
    )


if __name__ == "__main__":
    main()

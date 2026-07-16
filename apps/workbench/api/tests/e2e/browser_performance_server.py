from __future__ import annotations

import argparse
import json
import math
import os
import zipfile
from pathlib import Path

import uvicorn

from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.settings import WorkbenchApiSettings
from tests.support.training_jobs import (
    FakeProcess,
    FakeRunner,
    TrainingJobServiceHarness,
    create_app_with_training_service,
    write_tensorboard_run,
)


def _seed_log_runs(logs_root: Path) -> None:
    for run_index in range(8):
        points = list(range(1, 81))
        scalars = {
            "train/loss_epoch": [
                (step, 1.4 * math.exp(-step / 32) + run_index * 0.012)
                for step in points
            ],
            "validation/loss_epoch": [
                (step, 1.5 * math.exp(-step / 34) + run_index * 0.014)
                for step in points
            ],
            "train/accuracy_epoch": [
                (step, min(0.99, 0.42 + step / 170 + run_index * 0.004))
                for step in points
            ],
            "validation/accuracy_epoch": [
                (step, min(0.97, 0.39 + step / 185 + run_index * 0.003))
                for step in points
            ],
        }
        run_directory = write_tensorboard_run(
            logs_root,
            [
                "browser_performance",
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                f"run_{run_index:02d}_20260710_{run_index:06d}",
                "version_0",
            ],
            scalars=scalars,
            metrics={
                "test/accuracy": 0.81 + run_index * 0.01,
                "test/loss": 0.42 - run_index * 0.018,
            },
        )
        (run_directory / "result.json").write_text(
            json.dumps(
                {
                    "params": {
                        "hidden_dim": 64 + run_index * 16,
                        "learning_rate": 0.001,
                    },
                    "metrics": {
                        "test/accuracy": 0.81 + run_index * 0.01,
                        "test/loss": 0.42 - run_index * 0.018,
                    },
                }
            ),
            encoding="utf-8",
        )


def _write_import_fixture(root: Path) -> None:
    archive_path = root / "browser-performance-import.zip"
    prefix = (
        "browser_import/linears/linear/BASELINE/Mnist/"
        "imported_20260710_120000/version_0"
    )
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(f"{prefix}/hparams.yaml", "batch_size: 4\n")
        archive.writestr(
            f"{prefix}/result.json",
            json.dumps({"metrics": {"test/accuracy": 0.93}}),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--frontend-origin", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    logs_root = root / "logs"
    state_root = root / "state"
    snapshots_root = root / "snapshots"
    os.environ["WORKBENCH_API_LOGS_ROOT"] = str(logs_root)
    os.environ["WORKBENCH_API_STATE_ROOT"] = str(state_root)
    os.environ["WORKBENCH_API_SNAPSHOTS_ROOT"] = str(snapshots_root)
    _seed_log_runs(logs_root)
    _write_import_fixture(root)

    settings = WorkbenchApiSettings(
        cors_origins=[args.frontend_origin],
        logs_root=str(logs_root),
        snapshots_root=str(snapshots_root),
        state_root=str(state_root),
        auth_mode="none",
        allow_unsafe_local_mutations=True,
        allow_log_imports=True,
        training_cancellation_mode="process-group",
    )
    project_adapter = ProjectAdapterClient()
    try:
        training_manager = TrainingJobServiceHarness(
            root=root / "jobs",
            logs_root=logs_root,
            runner=FakeRunner(FakeProcess(exit_code=0)),
            cancellation_mode="process-group",
            project_adapter=project_adapter,
        )
        app = create_app_with_training_service(
            settings,
            training_manager,
            project_adapter=project_adapter,
        )
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=args.port,
            access_log=False,
            log_level="warning",
            reset_contextvars=True,
        )
    finally:
        project_adapter.close()


if __name__ == "__main__":
    main()

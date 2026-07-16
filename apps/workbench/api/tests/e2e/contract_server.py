from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.settings import WorkbenchApiSettings
from tests.support.training_jobs import (
    FakeRunner,
    TrainingJobServiceHarness,
    create_app_with_training_service,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--token", required=True)
    parser.add_argument("--frontend-origin", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    logs_root = root / "logs"
    state_root = root / "state"
    snapshots_root = root / "snapshots"
    os.environ["WORKBENCH_API_LOGS_ROOT"] = str(logs_root)
    os.environ["WORKBENCH_API_STATE_ROOT"] = str(state_root)
    os.environ["WORKBENCH_API_SNAPSHOTS_ROOT"] = str(snapshots_root)
    settings = WorkbenchApiSettings(
        cors_origins=[args.frontend_origin],
        logs_root=str(logs_root),
        snapshots_root=str(snapshots_root),
        state_root=str(state_root),
        auth_mode="bearer",
        token=args.token,
        allow_unsafe_local_mutations=True,
        allow_log_imports=False,
        training_cancellation_mode="process-group",
    )
    project_adapter = ProjectAdapterClient()
    try:
        training_manager = TrainingJobServiceHarness(
            root=root / "jobs",
            logs_root=logs_root,
            runner=FakeRunner(),
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

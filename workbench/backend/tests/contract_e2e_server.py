"""Temporary live backend used by the frontend/backend contract E2E test."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from workbench.backend.api import WorkbenchApiSettings
from workbench.backend.tests.helpers import (
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
    settings = WorkbenchApiSettings(
        cors_origins=[args.frontend_origin],
        logs_root=str(logs_root),
        snapshots_root=str(root / "snapshots"),
        auth_mode="bearer",
        token=args.token,
        allow_unsafe_local_mutations=True,
        allow_log_imports=False,
        training_cancellation_mode="process-group",
    )
    training_manager = TrainingJobServiceHarness(
        root=root / "jobs",
        logs_root=logs_root,
        runner=FakeRunner(),
        cancellation_mode="process-group",
    )
    app = create_app_with_training_service(settings, training_manager)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

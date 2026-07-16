from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import Lock

from emperor_workbench.log_experiments import LogExperimentMutationCoordinator
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_history import (
    KnownModelPackageIdentityResolver,
    LogRun,
    RunHistoryService,
)

_PROJECT_ADAPTER: ProjectAdapterClient | None = None


def install_project_adapter(client: ProjectAdapterClient | None) -> None:
    global _PROJECT_ADAPTER
    _PROJECT_ADAPTER = client


def project_adapter_client() -> ProjectAdapterClient:
    if _PROJECT_ADAPTER is None:
        raise RuntimeError("The test project Adapter fixture is not active.")
    return _PROJECT_ADAPTER


def model_identity_resolver(
    project_adapter: ProjectAdapterClient | None = None,
) -> KnownModelPackageIdentityResolver:
    catalog = ModelPackageCatalog(project_adapter or project_adapter_client())
    model_ids: Mapping[str, str] | None = None
    lookup_lock = Lock()

    def resolve(model_token: str) -> str | None:
        nonlocal model_ids
        if model_ids is None:
            with lookup_lock:
                if model_ids is None:
                    model_ids = catalog.identity_lookup()
        return model_ids.get(model_token)

    return resolve


def list_log_runs(
    *,
    logs_root: Path | str = "logs",
    state_root: Path | None = None,
) -> list[LogRun]:
    service = RunHistoryService(
        logs_root=logs_root,
        state_root=state_root,
        mutation_coordinator=LogExperimentMutationCoordinator(),
        active_log_writers=lambda: (),
        model_identity_resolver=model_identity_resolver(),
    )
    return list(
        service.list_runs(
            limit=1_000_000,
            offset=0,
            projection="full",
        ).runs
    )


__all__ = [
    "install_project_adapter",
    "model_identity_resolver",
    "list_log_runs",
    "project_adapter_client",
]

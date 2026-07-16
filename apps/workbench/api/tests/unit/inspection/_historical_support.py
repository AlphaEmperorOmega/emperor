from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from model_runtime.inspection import InspectionResult

from emperor_workbench.inspection import (
    InProcessInspectionExecutor,
    InspectionService,
)
from emperor_workbench.log_experiments import (
    LogExperimentMutationCoordinator,
)
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.run_history import (
    HistoricalInspectionSource,
    RunHistoryService,
)
from tests.support.inspection import inspection_response
from tests.support.model_packages import (
    model_identity_resolver,
    project_adapter_client,
)


def inspection_service(
    historical_source: HistoricalInspectionSource | None = None,
) -> InspectionService:
    return InspectionService(
        InProcessInspectionExecutor(),
        historical_source=historical_source,
    )


def semantic_inspection(
    service: InspectionService,
    *,
    model_type: str,
    model: str,
    preset: str,
    overrides: dict[str, Any],
    dataset: str | None,
    experiment_task: str | None = None,
    log_run_id: str | None = None,
) -> InspectionResult:
    selected = ModelPackageCatalog(project_adapter_client()).select_parts(
        model_type,
        model,
    )
    return service.inspect(
        selected,
        preset=preset,
        overrides=overrides,
        dataset=dataset,
        experiment_task=experiment_task,
        log_run_id=log_run_id,
    )


def http_inspection(
    service: InspectionService,
    **request: Any,
) -> dict[str, Any]:
    return inspection_response(semantic_inspection(service, **request)).model_dump(
        mode="json"
    )


def run_history(logs_root: Path) -> RunHistoryService:
    return RunHistoryService(
        logs_root=logs_root,
        mutation_coordinator=LogExperimentMutationCoordinator(),
        active_log_writers=lambda: (),
        model_identity_resolver=model_identity_resolver(),
    )


def first_run_id(run_history_service: RunHistoryService) -> str:
    return run_history_service.list_runs(limit=1, offset=0).runs[0].id


def checkpoint_state_dict(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layer_count: int,
    stack_prefix: str = "main_model.layers",
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {
        "input_model.model.weight_params": torch.zeros(input_dim, hidden_dim),
        "input_model.model.bias_params": torch.zeros(hidden_dim),
        "output_model.model.weight_params": torch.zeros(hidden_dim, output_dim),
        "output_model.model.bias_params": torch.zeros(output_dim),
    }
    for index in range(layer_count):
        state_dict[f"{stack_prefix}.{index}.model.weight_params"] = torch.zeros(
            hidden_dim,
            hidden_dim,
        )
        state_dict[f"{stack_prefix}.{index}.model.bias_params"] = torch.zeros(
            hidden_dim,
        )
    return state_dict


__all__ = [
    "checkpoint_state_dict",
    "first_run_id",
    "http_inspection",
    "inspection_service",
    "run_history",
    "semantic_inspection",
]

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from emperor_workbench.api._blocking import run_blocking_io
from emperor_workbench.api._dependencies import (
    get_project_adapter_client,
    get_training_run_plan_service,
)
from emperor_workbench.api._mutations import (
    HttpOperationPolicy,
    declare_http_operation,
)
from emperor_workbench.api._security import require_bearer_auth
from emperor_workbench.api.v1.run_plans._commands import (
    create_run_plan_command,
)
from emperor_workbench.api.v1.run_plans._contracts import (
    TrainingRunPlanCreateRequest,
    TrainingRunPlanResponse,
)
from emperor_workbench.api.v1.run_plans._mapping import run_plan_to_payload
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_plans import RunPlanService

router = APIRouter(
    prefix="/training",
    tags=["training"],
    dependencies=[Depends(require_bearer_auth)],
)


@router.post(
    "/run-plan",
    response_model=TrainingRunPlanResponse,
    summary="Create a training run plan",
    response_description="Materialized training runs for the current request.",
)
@declare_http_operation(HttpOperationPolicy.READ_ONLY)
async def create_training_run_plan(
    request: TrainingRunPlanCreateRequest,
    service: Annotated[
        RunPlanService,
        Depends(get_training_run_plan_service),
    ],
    project_adapter: Annotated[
        ProjectAdapterClient,
        Depends(get_project_adapter_client),
    ],
) -> TrainingRunPlanResponse:
    command = await run_blocking_io(
        create_run_plan_command,
        request,
        project_adapter=project_adapter,
    )
    return TrainingRunPlanResponse.model_validate(
        run_plan_to_payload(
            await run_blocking_io(
                service.preview,
                command,
            )
        )
    )


__all__ = ["router"]

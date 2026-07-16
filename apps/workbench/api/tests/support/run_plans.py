from __future__ import annotations

from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.run_plans import (
    CreateTrainingRunPlanCommand,
    RunPlanPersistenceCodec,
    RunPlanService,
    TrainingSearch,
)
from tests.support.model_packages import project_adapter_client


def worker_payload(
    *,
    preset: str = "baseline",
    presets: list[str] | None = None,
    datasets: list[str] | None = None,
    overrides: dict | None = None,
    search: dict | None = None,
    monitors: list[str] | None = None,
) -> dict[str, object]:
    service = RunPlanService(
        model_packages=ModelPackageCatalog(project_adapter_client()),
    )
    plan = service.preview(
        CreateTrainingRunPlanCommand(
            model="linears/linear",
            preset=preset,
            presets=presets,
            datasets=datasets or ["Mnist"],
            overrides=overrides or {},
            search=(
                TrainingSearch(
                    mode=search["mode"],
                    values=dict(search["values"]),
                    random_samples=search.get("randomSamples"),
                )
                if search is not None
                else None
            ),
            log_folder="unit_logs",
            monitors=monitors or [],
        )
    )
    serialized_plan = RunPlanPersistenceCodec.encode(plan)
    return {
        "id": "job-123",
        "monitors": monitors or [],
        "plannedRunCount": len(plan.runs),
        "runPlan": serialized_plan,
    }


__all__ = ["worker_payload"]

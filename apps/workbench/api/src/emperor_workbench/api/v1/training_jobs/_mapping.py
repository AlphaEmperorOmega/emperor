from __future__ import annotations

from typing import Any

from emperor_workbench.api.v1.run_plans._mapping import (
    run_plan_to_payload,
    search_to_payload,
)
from emperor_workbench.model_packages import ModelPackageIdentity
from emperor_workbench.training_jobs import (
    ActiveTrainingJob,
    TrainingJobView,
    TrainingProgressEventsPage,
    TrainingResultLinkView,
)


def _model_identity(model_id: str) -> dict[str, str]:
    identity = ModelPackageIdentity.from_id(model_id)
    return {"modelType": identity.model_type, "model": identity.model}


def _result_link_to_payload(link: TrainingResultLinkView) -> dict[str, Any]:
    return {"preset": link.preset, "dataset": link.dataset, "logDir": link.log_dir}


def training_job_to_payload(job: TrainingJobView) -> dict[str, Any]:
    return {
        "id": job.id,
        "status": job.status,
        **_model_identity(job.model),
        "preset": job.preset,
        "presets": job.presets,
        "experimentTask": job.experiment_task,
        "datasets": job.datasets,
        "overrides": job.overrides,
        "search": search_to_payload(job.search) if job.search is not None else None,
        "plannedRunCount": job.planned_run_count,
        "runPlan": (
            run_plan_to_payload(job.run_plan) if job.run_plan is not None else None
        ),
        "monitors": job.monitors,
        "logFolder": job.log_folder,
        "createdAt": job.created_at,
        "updatedAt": job.updated_at,
        "exitCode": job.exit_code,
        "pid": job.pid,
        "cancellationMode": job.cancellation_mode,
        "currentPreset": job.current_preset,
        "currentDataset": job.current_dataset,
        "epoch": job.epoch,
        "step": job.step,
        "metrics": job.metrics,
        "logDir": job.log_dir,
        "events": job.events,
        "eventCount": job.event_count,
        "eventCounts": job.event_counts,
        "eventsTruncated": job.events_truncated,
        "clusterGrowth": job.cluster_growth,
        "logTail": job.log_tail,
        "logTailTruncated": job.log_tail_truncated,
        "resultLinks": [_result_link_to_payload(link) for link in job.result_links],
    }


def active_training_job_to_payload(job: ActiveTrainingJob) -> dict[str, str]:
    return {"id": job.id, "status": job.status, "logFolder": job.log_folder}


def training_events_page_to_payload(
    page: TrainingProgressEventsPage,
) -> dict[str, Any]:
    return {
        "jobId": page.job_id,
        "offset": page.offset,
        "limit": page.limit,
        "totalCount": page.total_count,
        "nextOffset": page.next_offset,
        "events": page.events,
    }


__all__ = [
    "active_training_job_to_payload",
    "training_events_page_to_payload",
    "training_job_to_payload",
]

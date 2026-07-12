"""Map typed Training Job values to the stable HTTP payload."""

from __future__ import annotations

from typing import Any

from emperor.model_packages import model_identity_payload_from_id

from workbench.backend.training_jobs.contracts import (
    ActiveTrainingJob,
    TrainingJobView,
    TrainingProgressEventsPage,
    TrainingResultLinkView,
)
from workbench.backend.training_jobs.run_plan_adapter import (
    training_run_plan_to_payload,
    training_search_to_payload,
)


def _result_link_to_payload(link: TrainingResultLinkView) -> dict[str, Any]:
    return {"preset": link.preset, "dataset": link.dataset, "logDir": link.log_dir}


def training_job_to_payload(job: TrainingJobView) -> dict[str, Any]:
    return {
        "id": job.id,
        "status": job.status,
        **model_identity_payload_from_id(job.model),
        "preset": job.preset,
        "presets": job.presets,
        "experimentTask": job.experiment_task,
        "datasets": job.datasets,
        "overrides": job.overrides,
        "search": training_search_to_payload(job.search) if job.search else None,
        "plannedRunCount": job.planned_run_count,
        "runPlan": (
            training_run_plan_to_payload(job.run_plan) if job.run_plan else None
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
    "training_run_plan_to_payload",
]

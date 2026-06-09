"""Project stored training jobs and progress events into API payloads."""

from __future__ import annotations

from typing import Any

from viewer.backend.job_store import TrainingJobRecord
from viewer.backend.training_monitor_locator import TrainingMonitorLocator
from viewer.backend.training_run_progress import project_training_run_progress


class TrainingJobProjector:
    def __init__(self, monitor_locator: TrainingMonitorLocator | None = None) -> None:
        self._monitor_locator = monitor_locator or TrainingMonitorLocator()

    def project(
        self,
        job: TrainingJobRecord,
        *,
        events: list[dict[str, Any]],
        summarize,
    ) -> dict[str, Any]:
        latest_event = events[-1] if events else {}
        metrics_event = next(
            (
                event
                for event in reversed(events)
                if isinstance(event.get("metrics"), dict)
            ),
            {},
        )
        result_events = [
            event for event in events if event.get("type") == "dataset_completed"
        ]
        latest_preset = self._monitor_locator.event_preset_name(latest_event)
        run_plan = project_training_run_progress(
            job.run_plan,
            events,
            job.status,
            summarize,
        )
        return {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "preset": job.preset,
            "presets": job.presets,
            "datasets": job.datasets,
            "overrides": job.overrides,
            "search": job.search,
            "plannedRunCount": job.planned_run_count,
            "runPlan": run_plan,
            "monitors": job.monitors,
            "logFolder": job.log_folder,
            "createdAt": job.created_at,
            "updatedAt": job.updated_at,
            "exitCode": job.exit_code,
            "pid": job.pid,
            "currentPreset": latest_preset,
            "currentDataset": latest_event.get("dataset"),
            "epoch": latest_event.get("epoch"),
            "step": latest_event.get("step"),
            "metrics": metrics_event.get("metrics") or {},
            "logDir": latest_event.get("logDir"),
            "events": events,
            "logTail": self.log_tail(job),
            "resultLinks": [
                {
                    "preset": self._monitor_locator.event_preset_name(event),
                    "dataset": event.get("dataset"),
                    "logDir": event.get("logDir"),
                }
                for event in result_events
            ],
        }

    def log_tail(
        self,
        job: TrainingJobRecord,
        line_count: int = 80,
    ) -> list[str]:
        if not job.log_path.exists():
            return []
        return job.log_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()[-line_count:]


__all__ = ["TrainingJobProjector"]

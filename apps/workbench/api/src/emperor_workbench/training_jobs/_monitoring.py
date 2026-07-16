from __future__ import annotations

from typing import Any

from emperor_workbench.training_jobs._records import TrainingJobRecord


class TrainingMonitorLocator:
    def preset_in_job(self, job: TrainingJobRecord, preset: str) -> bool:
        return preset in job.presets

    def log_dir_for_monitor_data(
        self,
        *,
        events: list[dict[str, Any]],
        dataset: str | None,
        preset: str | None,
    ) -> str | None:
        for event in reversed(events):
            log_dir = event.get("logDir")
            if not log_dir:
                continue
            if dataset is None or event.get("dataset") == dataset:
                if self.event_matches_preset(event, preset):
                    return str(log_dir)
        return None

    def event_preset_name(self, event: dict[str, Any]) -> str | None:
        preset = event.get("preset")
        return preset if isinstance(preset, str) and preset else None

    def event_matches_preset(
        self,
        event: dict[str, Any],
        preset: str | None,
    ) -> bool:
        if preset is None:
            return True
        return self.event_preset_name(event) == preset


__all__ = ["TrainingMonitorLocator"]

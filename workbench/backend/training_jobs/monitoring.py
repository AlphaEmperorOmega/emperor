"""Locate trusted monitor inputs within one Training Job."""

from __future__ import annotations

from typing import Any

from workbench.backend.model_identity import normalize_preset_token
from workbench.backend.training_jobs.store import TrainingJobRecord


class TrainingMonitorLocator:
    def preset_in_job(self, job: TrainingJobRecord, preset: str) -> bool:
        normalized = normalize_preset_token(preset)
        return normalized in {
            normalize_preset_token(item)
            for item in job.presets
        }

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
        return normalize_preset_token(event.get("preset"))

    def event_matches_preset(
        self,
        event: dict[str, Any],
        preset: str | None,
    ) -> bool:
        if preset is None:
            return True
        return self.event_preset_name(event) == normalize_preset_token(preset)


__all__ = ["TrainingMonitorLocator"]

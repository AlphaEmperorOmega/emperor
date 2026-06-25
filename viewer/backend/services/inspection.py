"""Model inspection use cases."""

from __future__ import annotations

from typing import Any

from models.catalog import model_id_from_parts
from models.dataset_naming import normalize_dataset_name

from viewer.backend.inspector.discovery import load_model_parts
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.overrides import parse_override_mapping
from viewer.backend.repositories.log_runs import LogRunRepository
from viewer.backend.training_monitor_locator import normalize_preset_token


def _model_id(model_type: str, model: str) -> str:
    model_id = model_id_from_parts(model_type, model)
    if model_id is None:
        raise InspectorError(
            f"Unknown model: --model-type {model_type} --model {model}"
        )
    return model_id


def _normalized_dataset(dataset: str | None) -> str | None:
    return normalize_dataset_name(dataset) if dataset else None


class InspectionService:
    def __init__(self, log_runs: LogRunRepository | None = None) -> None:
        self._log_runs = log_runs

    def inspect(
        self,
        *,
        model_type: str,
        model: str,
        preset: str,
        overrides: dict[str, Any],
        dataset: str | None,
        log_run_id: str | None = None,
    ) -> dict[str, Any]:
        from viewer.backend.inspector.service import inspect_model

        model_id = _model_id(model_type, model)
        parts = load_model_parts(model_id)
        effective_overrides = parse_override_mapping(parts.config_module, overrides)
        if log_run_id:
            effective_overrides = {
                **parse_override_mapping(
                    parts.config_module,
                    self._saved_run_overrides(
                        log_run_id=log_run_id,
                        model_id=model_id,
                        preset=preset,
                        dataset=dataset,
                    ),
                    ignore_unknown=True,
                ),
                **effective_overrides,
            }

        return inspect_model(
            model_id,
            preset,
            dataset=dataset,
            parsed_overrides=effective_overrides,
        )

    def _saved_run_overrides(
        self,
        *,
        log_run_id: str,
        model_id: str,
        preset: str,
        dataset: str | None,
    ) -> dict[str, Any]:
        if self._log_runs is None:
            raise InspectorError("Log run inspection is not configured.")

        run = next(
            (
                candidate
                for candidate in self._log_runs.list_runs()
                if candidate.id == log_run_id
            ),
            None,
        )
        if run is None:
            raise InspectorError(f"Unknown log run id: {log_run_id}")
        if run.model != model_id:
            raise InspectorError(
                f"Log run '{log_run_id}' belongs to model '{run.model}', "
                f"not '{model_id}'."
            )
        if normalize_preset_token(run.preset) != normalize_preset_token(preset):
            raise InspectorError(
                f"Log run '{log_run_id}' belongs to preset '{run.preset}', "
                f"not '{preset}'."
            )
        if dataset and _normalized_dataset(run.dataset) != _normalized_dataset(dataset):
            raise InspectorError(
                f"Log run '{log_run_id}' belongs to dataset '{run.dataset}', "
                f"not '{dataset}'."
            )

        artifacts = self._log_runs.artifacts_for_run(log_run_id)
        params = artifacts.get("params")
        return dict(params) if isinstance(params, dict) else {}

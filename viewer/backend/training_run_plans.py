from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

from models.catalog import model_identity_payload_from_id
from models.config_overrides import config_key_to_model_param, normalize_key

from viewer.backend.inspector.discovery import (
    dataset_name,
    load_model_parts,
    option_cli_name,
    resolve_datasets,
)
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.overrides import parse_override_mapping
from viewer.backend.inspector.schema import config_schema, search_space_schema
from viewer.backend.inspector.search import (
    parse_training_search,
    strip_search_overrides,
)
from viewer.backend.inspector.service import reject_locked_overrides
from viewer.backend.inspector.values import serialize_config_value
from viewer.backend.log_runs import is_valid_log_experiment_name


def _shell_quote(value: str) -> str:
    if value == "":
        return "''"
    safe = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_@%+=:,./-"
    )
    if all(character in safe for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _command_value(value: Any) -> str:
    serialized = serialize_config_value(value)
    if serialized is None:
        return "None"
    return str(serialized)


def _build_training_command(
    *,
    fields: list[dict[str, Any]],
    by_key: dict[str, dict[str, Any]],
    model: str,
    preset: str,
    dataset: str,
    overrides: dict[str, Any],
    log_folder: str,
) -> str:
    values_by_field_key: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        field = by_key.get(normalize_key(str(raw_key)))
        if field is not None:
            values_by_field_key[str(field["key"])] = raw_value

    identity = model_identity_payload_from_id(model)
    parts = [
        "source",
        "experiment.sh",
        "--model-type",
        _shell_quote(identity["modelType"]),
        "--model",
        _shell_quote(identity["model"]),
        "--preset",
        _shell_quote(preset),
        "--datasets",
        _shell_quote(dataset),
    ]
    if log_folder:
        parts.extend(["--logdir", _shell_quote(log_folder)])

    config_parts: list[str] = []
    for field in fields:
        field_key = str(field["key"])
        if field_key not in values_by_field_key:
            continue
        config_parts.extend(
            [
                str(field["flag"]),
                _shell_quote(_command_value(values_by_field_key[field_key])),
            ]
        )
    if config_parts:
        parts.append("--config")
        parts.extend(config_parts)
    return " ".join(parts)


@dataclass(frozen=True)
class SelectedTrainingInputs:
    parts: Any
    selected_preset_names: list[str]
    selected_options: list[Any]
    selected_datasets: list[type]
    parsed_searches: list[Any]
    parsed_search: Any | None
    effective_overrides: dict[str, Any]
    parsed_overrides: dict[str, Any]


class TrainingRunPlanBuilder:
    def __init__(self, random_source: random.Random | None = None) -> None:
        self._random = random_source or random

    def resolve_inputs(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None,
        datasets: list[str],
        overrides: dict[str, Any],
        search: dict[str, Any] | None,
    ) -> SelectedTrainingInputs:
        return self._resolve_selected_training_inputs(
            model=model,
            preset=preset,
            presets=presets,
            datasets=datasets,
            overrides=overrides,
            search=search,
        )

    def _resolve_selected_training_inputs(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None,
        datasets: list[str],
        overrides: dict[str, Any],
        search: dict[str, Any] | None,
    ) -> SelectedTrainingInputs:
        if not datasets:
            raise InspectorError("Training requires at least one selected dataset.")

        parts = load_model_parts(model)
        selected_preset_names, selected_options = self._resolve_presets(
            parts,
            model,
            preset,
            presets,
        )
        selected_datasets = resolve_datasets(parts, datasets)
        parsed_searches = self._parse_selected_searches(
            model=model,
            selected_preset_names=selected_preset_names,
            search=search,
            dataset_count=len(selected_datasets),
        )
        parsed_search = next(
            (candidate for candidate in parsed_searches if candidate is not None),
            None,
        )
        search_model_params = self._search_model_params(parsed_searches)
        effective_overrides = self._effective_overrides_for_search(
            parts=parts,
            overrides=overrides,
            search_model_params=search_model_params,
        )
        parsed_overrides = self._parse_and_validate_overrides(
            model=model,
            parts=parts,
            selected_preset_names=selected_preset_names,
            effective_overrides=effective_overrides,
        )
        return SelectedTrainingInputs(
            parts=parts,
            selected_preset_names=selected_preset_names,
            selected_options=selected_options,
            selected_datasets=selected_datasets,
            parsed_searches=parsed_searches,
            parsed_search=parsed_search,
            effective_overrides=effective_overrides,
            parsed_overrides=parsed_overrides,
        )

    def _parse_selected_searches(
        self,
        *,
        model: str,
        selected_preset_names: list[str],
        search: dict[str, Any] | None,
        dataset_count: int,
    ) -> list[Any]:
        return [
            parse_training_search(
                model,
                selected_preset,
                search,
                dataset_count=dataset_count,
            )
            for selected_preset in selected_preset_names
        ]

    def _search_model_params(self, parsed_searches: list[Any]) -> set[str]:
        search_model_params: set[str] = set()
        for parsed in parsed_searches:
            if parsed is not None:
                search_model_params.update(parsed.model_params)
        return search_model_params

    def _effective_overrides_for_search(
        self,
        *,
        parts,
        overrides: dict[str, Any],
        search_model_params: set[str],
    ) -> dict[str, Any]:
        return strip_search_overrides(
            parts.config_module,
            overrides,
            search_model_params,
        )

    def _parse_and_validate_overrides(
        self,
        *,
        model: str,
        parts,
        selected_preset_names: list[str],
        effective_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        parsed_overrides = parse_override_mapping(
            parts.config_module,
            effective_overrides,
        )
        for selected_preset in selected_preset_names:
            reject_locked_overrides(model, selected_preset, parsed_overrides)
        return parsed_overrides

    def valid_plan_log_folder(self, log_folder: str) -> str:
        return (
            log_folder
            if log_folder and is_valid_log_experiment_name(log_folder)
            else ""
        )

    def summarize(self, runs: list[dict[str, Any]]) -> dict[str, int]:
        statuses = [str(run.get("status", "Pending")) for run in runs]
        completed_epochs = 0
        remaining_epochs = 0
        total_epochs = 0
        for run in runs:
            row_total = int(run.get("totalEpochs") or 0)
            row_current = int(run.get("currentEpoch") or 0)
            row_status = str(run.get("status", "Pending"))
            total_epochs += row_total
            if row_status == "Completed":
                row_done = row_total
            elif row_status in {"Running", "Failed", "Cancelled"}:
                row_done = min(row_current, row_total)
            else:
                row_done = 0
            completed_epochs += row_done
            if row_status in {"Pending", "Running"}:
                remaining_epochs += max(0, row_total - row_done)

        return {
            "totalRuns": len(runs),
            "completedRuns": statuses.count("Completed"),
            "runningRuns": statuses.count("Running"),
            "pendingRuns": statuses.count("Pending"),
            "failedRuns": statuses.count("Failed"),
            "cancelledRuns": statuses.count("Cancelled"),
            "skippedRuns": statuses.count("Skipped"),
            "totalEpochs": total_epochs,
            "completedEpochs": completed_epochs,
            "remainingEpochs": remaining_epochs,
        }

    def create(
        self,
        *,
        model: str,
        selected: SelectedTrainingInputs,
        log_folder: str,
    ) -> dict[str, Any]:
        total_epochs = self._total_epochs(selected.parts, selected.parsed_overrides)
        runs: list[dict[str, Any]] = []
        for selected_preset, selected_search in zip(
            selected.selected_preset_names,
            selected.parsed_searches,
            strict=True,
        ):
            fixed_changes, fixed_overrides = self._ordered_override_changes(
                model=model,
                preset=selected_preset,
                overrides=selected.effective_overrides,
            )
            for dataset in selected.selected_datasets:
                dataset_display_name = dataset_name(dataset)
                for search_changes, search_overrides in self._search_combinations(
                    model=model,
                    preset=selected_preset,
                    parsed_search=selected_search,
                ):
                    row_overrides = {**fixed_overrides, **search_overrides}
                    row_index = len(runs) + 1
                    runs.append(
                        self._pending_run(
                            model=model,
                            index=row_index,
                            preset=selected_preset,
                            dataset=dataset_display_name,
                            changes=[*fixed_changes, *search_changes],
                            overrides=row_overrides,
                            log_folder=log_folder,
                            total_epochs=total_epochs,
                        )
                    )

        return self._plan_payload(
            model=model,
            selected=selected,
            log_folder=log_folder,
            runs=runs,
        )

    def from_submitted(
        self,
        *,
        model: str,
        selected: SelectedTrainingInputs,
        run_plan: dict[str, Any],
        log_folder: str,
    ) -> dict[str, Any]:
        valid_presets = set(selected.selected_preset_names)
        valid_datasets = {
            dataset_name(dataset) for dataset in selected.selected_datasets
        }
        runs = []
        for index, row in enumerate(run_plan.get("runs") or [], start=1):
            preset = str(row.get("preset") or "")
            dataset = str(row.get("dataset") or "")
            if preset not in valid_presets:
                raise InspectorError(f"Run plan contains unknown preset '{preset}'.")
            if dataset not in valid_datasets:
                raise InspectorError(f"Run plan contains unknown dataset '{dataset}'.")

            row_overrides = dict(row.get("overrides") or {})
            snapshot_id = row.get("snapshotId")
            snapshot_name = row.get("snapshotName")
            parsed_row_overrides = parse_override_mapping(
                selected.parts.config_module,
                row_overrides,
            )
            reject_locked_overrides(model, preset, parsed_row_overrides)
            runs.append(
                {
                    **row,
                    "id": str(row.get("id") or f"run-{index:04d}"),
                    "index": index,
                    "status": "Pending",
                    "preset": preset,
                    "snapshotId": str(snapshot_id) if snapshot_id is not None else None,
                    "snapshotName": str(snapshot_name)
                    if snapshot_name is not None
                    else None,
                    "dataset": dataset,
                    "overrides": row_overrides,
                    "command": self._training_command(
                        model=model,
                        preset=preset,
                        dataset=dataset,
                        overrides=row_overrides,
                        log_folder=log_folder,
                    ),
                    "totalEpochs": int(row.get("totalEpochs") or 0),
                    "currentEpoch": 0,
                    "metrics": {},
                    "logDir": None,
                    "error": None,
                    "errorTraceback": None,
                }
            )
        if not runs:
            raise InspectorError("Run plan requires at least one training run.")

        return self._plan_payload(
            model=model,
            selected=selected,
            log_folder=log_folder,
            runs=runs,
        )

    def _resolve_presets(
        self,
        parts,
        model: str,
        preset: str,
        presets: list[str] | None,
    ):
        raw_presets = presets if presets else [preset]
        selected = []
        seen = set()
        unknown = []
        for raw_preset in raw_presets:
            if not isinstance(raw_preset, str) or not raw_preset.strip():
                continue
            try:
                option = parts.experiment_options.get_option(raw_preset)
            except Exception:
                unknown.append(raw_preset)
                continue
            if option.name in seen:
                continue
            seen.add(option.name)
            selected.append(
                (
                    option_cli_name(parts.experiment_options, option),
                    option,
                )
            )
        if unknown:
            raise InspectorError(f"Unknown preset '{unknown[0]}' for model '{model}'.")
        if not selected:
            raise InspectorError("Training requires at least one selected preset.")
        return [name for name, _option in selected], [
            option for _name, option in selected
        ]

    def _field_maps(
        self,
        model: str,
        preset: str,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        fields = config_schema(model, preset)["fields"]
        by_key: dict[str, dict[str, Any]] = {}
        for field in fields:
            by_key[normalize_key(str(field["key"]))] = field
            by_key[normalize_key(str(field["configKey"]))] = field
            by_key[
                normalize_key(config_key_to_model_param(str(field["configKey"])))
            ] = field
        return fields, by_key

    def _ordered_override_changes(
        self,
        *,
        model: str,
        preset: str,
        overrides: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        fields, by_key = self._field_maps(model, preset)
        values_by_field_key: dict[str, Any] = {}
        for raw_key, raw_value in overrides.items():
            field = by_key.get(normalize_key(str(raw_key)))
            if field is None:
                continue
            values_by_field_key[str(field["key"])] = serialize_config_value(raw_value)

        changes = []
        ordered_overrides: dict[str, Any] = {}
        for field in fields:
            field_key = str(field["key"])
            if field_key not in values_by_field_key:
                continue
            value = values_by_field_key[field_key]
            ordered_overrides[field_key] = value
            changes.append(
                {
                    "key": field_key,
                    "label": str(field["label"]),
                    "value": value,
                    "source": "override",
                }
            )
        return changes, ordered_overrides

    def _search_axis_maps(
        self,
        *,
        model: str,
        preset: str,
    ) -> dict[str, dict[str, Any]]:
        axes = search_space_schema(model, preset)["axes"]
        return {normalize_key(str(axis["key"])): axis for axis in axes}

    def _search_combinations(
        self,
        *,
        model: str,
        preset: str,
        parsed_search,
    ) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
        if parsed_search is None:
            return [([], {})]

        axis_order = list(parsed_search.values)
        model_param_order = list(parsed_search.search_overrides)
        axis_maps = self._search_axis_maps(model=model, preset=preset)
        indexed_values = []
        for axis_key, model_param in zip(axis_order, model_param_order, strict=True):
            serialized_values = parsed_search.values[axis_key]
            parsed_values = parsed_search.search_overrides[model_param]
            indexed_values.append(
                [
                    (axis_key, model_param, serialized_value, parsed_value)
                    for serialized_value, parsed_value in zip(
                        serialized_values,
                        parsed_values,
                        strict=True,
                    )
                ]
            )

        combinations = list(itertools.product(*indexed_values))
        if parsed_search.mode == "random":
            combinations = self._random.sample(
                combinations,
                min(parsed_search.random_samples or 10, len(combinations)),
            )

        materialized = []
        for combination in combinations:
            changes = []
            overrides = {}
            for axis_key, _model_param, serialized_value, parsed_value in combination:
                axis = axis_maps.get(normalize_key(axis_key), {})
                field_key = normalize_key(str(axis.get("configKey", axis_key)))
                overrides[field_key] = serialize_config_value(parsed_value)
                changes.append(
                    {
                        "key": field_key,
                        "label": str(axis.get("label", axis_key)),
                        "value": serialized_value,
                        "source": "search",
                    }
                )
            materialized.append((changes, overrides))
        return materialized

    def _training_command(
        self,
        *,
        model: str,
        preset: str,
        dataset: str,
        overrides: dict[str, Any],
        log_folder: str,
    ) -> str:
        fields, by_key = self._field_maps(model, preset)
        return _build_training_command(
            fields=fields,
            by_key=by_key,
            model=model,
            preset=preset,
            dataset=dataset,
            overrides=overrides,
            log_folder=log_folder,
        )

    def _total_epochs(self, parts, parsed_overrides: dict[str, Any]) -> int:
        raw_epochs = parsed_overrides.get(
            "num_epochs",
            getattr(parts.config_module, "NUM_EPOCHS", 10),
        )
        try:
            return max(0, int(raw_epochs))
        except (TypeError, ValueError):
            return 0

    def _pending_run(
        self,
        *,
        model: str,
        index: int,
        preset: str,
        dataset: str,
        changes: list[dict[str, Any]],
        overrides: dict[str, Any],
        log_folder: str,
        total_epochs: int,
    ) -> dict[str, Any]:
        return {
            "id": f"run-{index:04d}",
            "index": index,
            "status": "Pending",
            "preset": preset,
            "dataset": dataset,
            "changes": changes,
            "overrides": overrides,
            "command": self._training_command(
                model=model,
                preset=preset,
                dataset=dataset,
                overrides=overrides,
                log_folder=log_folder,
            ),
            "totalEpochs": total_epochs,
            "currentEpoch": 0,
            "metrics": {},
            "logDir": None,
            "error": None,
            "errorTraceback": None,
        }

    def _plan_payload(
        self,
        *,
        model: str,
        selected: SelectedTrainingInputs,
        log_folder: str,
        runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            **model_identity_payload_from_id(model),
            "preset": selected.selected_preset_names[0],
            "presets": selected.selected_preset_names,
            "datasets": [
                dataset_name(dataset) for dataset in selected.selected_datasets
            ],
            "overrides": selected.effective_overrides,
            "search": (
                selected.parsed_search.to_payload()
                if selected.parsed_search is not None
                else None
            ),
            "logFolder": log_folder,
            "isRandomSearch": bool(
                selected.parsed_search and selected.parsed_search.mode == "random"
            ),
            "runs": runs,
            "summary": self.summarize(runs),
        }

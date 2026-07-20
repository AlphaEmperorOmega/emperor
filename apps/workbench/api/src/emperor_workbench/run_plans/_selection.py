from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model_runtime.inspection import ConfigurationField, resolve_override_key
from model_runtime.packages import (
    config_key_to_model_param,
    dataset_name,
    iter_supported_config_keys,
    normalize_key,
)
from model_runtime.runs import RunRequest

from emperor_workbench.model_packages import (
    ModelPackageCatalog,
    ModelPackageFailure,
    SelectedModelPackage,
)
from emperor_workbench.project_adapter import (
    ModelPackageReference,
    ProjectAdapterFailure,
)
from emperor_workbench.run_plans._errors import RunPlanFailure
from emperor_workbench.run_plans._records import TrainingSearch
from emperor_workbench.run_plans._search import adapt_search


@dataclass(frozen=True)
class SelectedTrainingInputs:
    package: ModelPackageReference
    request: RunRequest


def require_package(
    model_packages: ModelPackageCatalog,
    model: str,
) -> ModelPackageReference:
    try:
        return model_packages.select(model).reference
    except ModelPackageFailure as exc:
        raise RunPlanFailure(exc.detail) from exc


def resolve_presets(
    package: ModelPackageReference,
    *,
    model: str,
    preset: str,
    presets: list[str] | None,
) -> list[str]:
    raw_presets = presets if presets else [preset]
    selected: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for raw_preset in raw_presets:
        if not isinstance(raw_preset, str) or not raw_preset.strip():
            continue
        try:
            preset_member = package.resolve_preset(raw_preset)
        except (ProjectAdapterFailure, ValueError):
            unknown.append(raw_preset)
            continue
        if preset_member.name in seen:
            continue
        seen.add(preset_member.name)
        selected.append(package.preset_name(preset_member))
    if unknown:
        raise RunPlanFailure(f"Unknown preset '{unknown[0]}' for model '{model}'.")
    if not selected:
        raise RunPlanFailure("Training requires at least one selected preset.")
    return selected


def _effective_overrides_for_search(
    *,
    package: ModelPackageReference,
    overrides: dict[str, Any],
    search_model_params: set[str],
) -> dict[str, Any]:
    if not overrides or not search_model_params:
        return dict(overrides)
    supported = {
        normalize_key(config_key): config_key
        for config_key in iter_supported_config_keys(package.runtime_defaults)
    }
    filtered: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        canonical_key = resolve_override_key(
            normalize_key(str(raw_key)),
            supported,
        )
        if (
            canonical_key is not None
            and config_key_to_model_param(canonical_key) in search_model_params
        ):
            continue
        filtered[raw_key] = raw_value
    return filtered


def _validate_overrides(
    *,
    package: ModelPackageReference,
    selected_preset_names: list[str],
    effective_overrides: dict[str, Any],
) -> None:
    selected = SelectedModelPackage(package)
    try:
        parsed_overrides = selected.parse_overrides(effective_overrides).values
        for selected_preset in selected_preset_names:
            selected.reject_locked_overrides(
                selected_preset,
                parsed_overrides,
            )
    except ModelPackageFailure as exc:
        raise RunPlanFailure(exc.detail) from exc


def resolve_inputs(
    model_packages: ModelPackageCatalog,
    *,
    model: str,
    preset: str,
    presets: list[str] | None,
    experiment_task: str | None,
    datasets: list[str],
    overrides: dict[str, Any],
    search: TrainingSearch | None,
) -> SelectedTrainingInputs:
    if not datasets:
        raise RunPlanFailure("Training requires at least one selected dataset.")

    package = require_package(model_packages, model)
    try:
        selected_experiment_task = package.resolve_experiment_task(experiment_task)
        selected_datasets = package.resolve_datasets(
            datasets,
            selected_experiment_task,
        )
    except ProjectAdapterFailure as exc:
        raise RunPlanFailure(exc.detail, kind=exc.kind) from exc
    except ValueError as exc:
        raise RunPlanFailure(str(exc)) from exc
    selected_experiment_task_name = package.task_name(selected_experiment_task)
    selected_preset_names = resolve_presets(
        package,
        model=model,
        preset=preset,
        presets=presets,
    )
    parsed_search, search_model_params = adapt_search(
        package,
        selected_preset_names[0],
        search,
    )
    effective_overrides = _effective_overrides_for_search(
        package=package,
        overrides=overrides,
        search_model_params=search_model_params,
    )
    _validate_overrides(
        package=package,
        selected_preset_names=selected_preset_names,
        effective_overrides=effective_overrides,
    )
    return SelectedTrainingInputs(
        package=package,
        request=RunRequest(
            presets=tuple(selected_preset_names),
            datasets=tuple(dataset_name(dataset) for dataset in selected_datasets),
            experiment_task=selected_experiment_task_name,
            overrides=effective_overrides,
            search=parsed_search,
        ),
    )


def resolve_monitor_names(
    package: ModelPackageReference,
    monitor_names: list[str] | None,
) -> list[str]:
    try:
        return [monitor.name for monitor in package.resolve_monitors(monitor_names)]
    except ProjectAdapterFailure as exc:
        raise RunPlanFailure(exc.detail, kind=exc.kind) from exc
    except ValueError as exc:
        raise RunPlanFailure(str(exc)) from exc


def configuration_fields(
    model_packages: ModelPackageCatalog,
    *,
    model: str,
    preset: str,
) -> tuple[tuple[ConfigurationField, ...], dict[str, ConfigurationField]]:
    try:
        fields = model_packages.select(model).configuration(preset).fields
    except ModelPackageFailure as exc:
        raise RunPlanFailure(exc.detail) from exc
    by_key: dict[str, ConfigurationField] = {}
    for config_field in fields:
        by_key[normalize_key(config_field.key)] = config_field
    for config_field in fields:
        model_param_key = normalize_key(config_key_to_model_param(config_field.key))
        by_key.setdefault(model_param_key, config_field)
    return fields, by_key


__all__ = [
    "SelectedTrainingInputs",
    "configuration_fields",
    "require_package",
    "resolve_inputs",
    "resolve_monitor_names",
    "resolve_presets",
]

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from model_runtime.packages import ModelPackage, RuntimeDefaultsError, dataset_name
from model_runtime.runs._search_parsing import ParsedSearch, SearchValue, parse_search
from model_runtime.runs.errors import InvalidRunPlan, InvalidRunRequest, PlanTooLarge
from model_runtime.runs.records import (
    PlanningBudget,
    PresetSearch,
    RandomSource,
    RunParameter,
    RunParameterSource,
    RunPlan,
    RunRequest,
    RunSpec,
    SubmittedRun,
)


@dataclass(frozen=True, slots=True)
class _ResolvedRequest:
    experiment_task_name: str
    preset_names: tuple[str, ...]
    datasets: tuple[type, ...]
    searches: tuple[ParsedSearch | None, ...]
    effective_overrides_by_preset: tuple[Mapping[str, Any], ...]
    serialized_overrides: Mapping[str, Any]

    def preset_searches(self) -> tuple[PresetSearch, ...]:
        return tuple(
            PresetSearch(
                preset=preset,
                search=search.spec if search is not None else None,
            )
            for preset, search in zip(
                self.preset_names,
                self.searches,
                strict=True,
            )
        )


def _request_error(exc: Exception) -> InvalidRunRequest:
    return InvalidRunRequest(str(exc))


def _plan_error(exc: Exception) -> InvalidRunPlan:
    return InvalidRunPlan(str(exc))


def _require_model_package(value: object) -> ModelPackage:
    if not isinstance(value, ModelPackage):
        raise TypeError("Runs require a selected ModelPackage.")
    return value


def _selected_planning_budget(value: object) -> PlanningBudget:
    if value is None:
        return PlanningBudget()
    if not isinstance(value, PlanningBudget):
        raise TypeError("Run planning budget must be a PlanningBudget.")
    return value


def _resolve_presets(
    package: ModelPackage,
    raw_presets: Sequence[object],
) -> tuple[str, ...]:
    selected_names: list[str] = []
    seen: set[str] = set()
    for raw_preset in raw_presets:
        if not isinstance(raw_preset, str) or not raw_preset.strip():
            continue
        try:
            preset = package.resolve_preset(raw_preset)
        except ValueError as exc:
            raise _request_error(exc) from exc
        if preset.name in seen:
            continue
        seen.add(preset.name)
        selected_names.append(package.preset_name(preset))
    if not selected_names:
        raise InvalidRunRequest("Training requires at least one selected preset.")
    return tuple(selected_names)


def _strip_searched_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any],
    searched_model_params: set[str],
) -> dict[str, Any]:
    if not searched_model_params:
        return dict(overrides)
    try:
        runtime_defaults = package.runtime_defaults_spec
        canonical = runtime_defaults.canonicalize_overrides(overrides)
    except RuntimeDefaultsError as exc:
        raise _request_error(exc) from exc
    return {
        key: value
        for key, value in canonical.items()
        if runtime_defaults.model_parameter(key) not in searched_model_params
    }


def _reject_conflicting_locks(
    package: ModelPackage,
    preset_name: str,
    parsed_overrides: Mapping[str, Any],
) -> None:
    try:
        package.runtime_defaults_spec.reject_conflicting_locked_overrides(
            preset_name,
            parsed_overrides,
        )
    except RuntimeDefaultsError as exc:
        raise _request_error(exc) from exc


def _resolve_task_and_datasets(
    package: ModelPackage,
    request: RunRequest,
) -> tuple[str, tuple[type[Any], ...]]:
    if not request.datasets:
        raise InvalidRunRequest("Training requires at least one selected dataset.")
    try:
        experiment_task = package.resolve_experiment_task(request.experiment_task)
        datasets = tuple(
            package.resolve_datasets(list(request.datasets), experiment_task)
        )
    except ValueError as exc:
        raise _request_error(exc) from exc
    return package.task_name(experiment_task), datasets


def _resolve_request_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any],
    preset_names: tuple[str, ...],
    searches: tuple[ParsedSearch | None, ...],
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    searched_model_params = {
        model_param
        for search in searches
        if search is not None
        for model_param in search.model_params
    }
    top_level_effective_overrides = _strip_searched_overrides(
        package,
        overrides,
        searched_model_params,
    )
    try:
        runtime_defaults = package.runtime_defaults_spec
        serialized_overrides = runtime_defaults.serialize_overrides(
            top_level_effective_overrides,
        )
    except RuntimeDefaultsError as exc:
        raise _request_error(exc) from exc

    effective_overrides_by_preset: list[Mapping[str, Any]] = []
    for preset_name, search in zip(preset_names, searches, strict=True):
        effective_overrides = _strip_searched_overrides(
            package,
            overrides,
            search.model_params if search is not None else set(),
        )
        try:
            parsed_overrides = runtime_defaults.parse_overrides(effective_overrides)
        except RuntimeDefaultsError as exc:
            raise _request_error(exc) from exc
        _reject_conflicting_locks(package, preset_name, parsed_overrides)
        effective_overrides_by_preset.append(effective_overrides)
    return tuple(effective_overrides_by_preset), serialized_overrides


def _resolve_request(
    package: ModelPackage,
    request: RunRequest,
    budget: PlanningBudget,
) -> _ResolvedRequest:
    experiment_task_name, datasets = _resolve_task_and_datasets(package, request)
    preset_names = _resolve_presets(package, request.presets)
    searches = tuple(
        parse_search(package, preset_name, request.search, budget)
        for preset_name in preset_names
    )
    effective_overrides_by_preset, serialized_overrides = _resolve_request_overrides(
        package,
        request.overrides,
        preset_names,
        searches,
    )
    return _ResolvedRequest(
        experiment_task_name=experiment_task_name,
        preset_names=preset_names,
        datasets=datasets,
        searches=searches,
        effective_overrides_by_preset=effective_overrides_by_preset,
        serialized_overrides=serialized_overrides,
    )


def _ordered_parameters(
    package: ModelPackage,
    overrides: Mapping[str, Any],
    *,
    source: RunParameterSource,
) -> tuple[RunParameter, ...]:
    try:
        runtime_defaults = package.runtime_defaults_spec
        canonical = runtime_defaults.canonicalize_overrides(overrides)
    except RuntimeDefaultsError as exc:
        raise _request_error(exc) from exc
    supported_keys = runtime_defaults.supported_keys
    missing_keys = sorted(set(canonical) - set(supported_keys))
    if missing_keys:
        raise InvalidRunRequest(
            "Canonical Run overrides are not supported Runtime Defaults: "
            f"{', '.join(missing_keys)}."
        )
    ordered_keys = runtime_defaults.ordered_configuration_keys()
    return tuple(
        RunParameter(
            key=key,
            value=runtime_defaults.serialize_value(canonical[key]),
            source=source,
        )
        for key in ordered_keys
        if key in canonical
    )


def _search_parameters(
    search: ParsedSearch,
    combination: tuple[SearchValue, ...],
) -> tuple[RunParameter, ...]:
    return tuple(
        RunParameter(
            key=axis.key,
            value=value.serialized,
            source="search",
        )
        for axis, value in zip(search.axes, combination, strict=True)
    )


def _planned_run_count(resolved: _ResolvedRequest) -> int:
    dataset_count = len(resolved.datasets)
    return sum(
        (search.prepared.selected_count if search is not None else 1) * dataset_count
        for search in resolved.searches
    )


def _reject_plan_budget(
    planned_run_count: int,
    budget: PlanningBudget,
    resolved: _ResolvedRequest,
) -> None:
    limit = budget.max_materialized_runs
    if limit is not None and planned_run_count > limit:
        searches = tuple(search for search in resolved.searches if search is not None)
        axis_value_counts = tuple(
            len(axis.values) for search in searches for axis in search.axes
        )
        details = (
            f" across {len(axis_value_counts)} axes with value counts "
            f"{', '.join(str(count) for count in axis_value_counts)}"
            if axis_value_counts
            else ""
        )
        raise PlanTooLarge(
            "Training run plan is too large: "
            f"{planned_run_count} planned runs exceeds limit {limit}{details}. "
            "Narrow the sweep with --search-keys or --search-set."
        )


def _materialize_planned_runs(
    package: ModelPackage,
    resolved: _ResolvedRequest,
    random_source: RandomSource | None,
) -> tuple[RunSpec, ...]:
    runs: list[RunSpec] = []
    for preset_name, search, effective_overrides in zip(
        resolved.preset_names,
        resolved.searches,
        resolved.effective_overrides_by_preset,
        strict=True,
    ):
        fixed_parameters = _ordered_parameters(
            package,
            effective_overrides,
            source="override",
        )
        for dataset in resolved.datasets:
            combinations = (
                search.prepared.combinations(random_source)
                if search is not None
                else iter(((),))
            )
            for combination in combinations:
                index = len(runs) + 1
                runs.append(
                    RunSpec(
                        id=f"run-{index:04d}",
                        experiment_task=resolved.experiment_task_name,
                        preset=preset_name,
                        dataset=dataset_name(dataset),
                        parameters=(
                            fixed_parameters
                            if search is None
                            else fixed_parameters
                            + _search_parameters(search, combination)
                        ),
                    )
                )
    return tuple(runs)


def plan_runs(
    package: ModelPackage,
    request: RunRequest,
    *,
    random_source: RandomSource | None = None,
    budget: PlanningBudget | None = None,
) -> RunPlan:
    package = _require_model_package(package)
    planning_budget = _selected_planning_budget(budget)
    resolved = _resolve_request(package, request, planning_budget)
    _reject_plan_budget(_planned_run_count(resolved), planning_budget, resolved)
    if random_source is None and any(
        search is not None and search.spec.mode == "random"
        for search in resolved.searches
    ):
        raise InvalidRunRequest("Random search requires an explicit random source.")

    return RunPlan(
        identity=package.identity,
        presets=resolved.preset_names,
        experiment_task=resolved.experiment_task_name,
        datasets=tuple(dataset_name(dataset) for dataset in resolved.datasets),
        overrides=resolved.serialized_overrides,
        search=None,
        runs=_materialize_planned_runs(package, resolved, random_source),
        preset_searches=resolved.preset_searches(),
    )


def _submitted_parameters(
    package: ModelPackage,
    preset_name: str,
    overrides: Mapping[str, Any],
) -> tuple[RunParameter, ...]:
    try:
        runtime_defaults = package.runtime_defaults_spec
        parsed = runtime_defaults.parse_overrides(overrides)
        runtime_defaults.reject_locked_overrides(preset_name, parsed)
        serialized = runtime_defaults.serialize_overrides(overrides)
    except (RuntimeDefaultsError, ValueError) as exc:
        raise _plan_error(exc) from exc
    return _ordered_parameters(
        package,
        serialized,
        source="override",
    )


def accept_run_plan(
    package: ModelPackage,
    request: RunRequest,
    submitted_runs: Sequence[SubmittedRun],
    *,
    budget: PlanningBudget | None = None,
) -> RunPlan:
    package = _require_model_package(package)
    planning_budget = _selected_planning_budget(budget)
    resolved = _resolve_request(package, request, planning_budget)
    if not submitted_runs:
        raise InvalidRunPlan("Run plan requires at least one training run.")
    limit = planning_budget.max_materialized_runs
    if limit is not None and len(submitted_runs) > limit:
        raise InvalidRunPlan(
            "Submitted run plan is too large: "
            f"{len(submitted_runs)} submitted runs exceeds {limit}."
        )

    valid_presets = set(resolved.preset_names)
    valid_datasets = {dataset_name(dataset) for dataset in resolved.datasets}
    accepted: list[RunSpec] = []
    seen_ids: set[str] = set()
    for index, submitted in enumerate(submitted_runs, start=1):
        run_id = submitted.id or f"run-{index:04d}"
        if run_id in seen_ids:
            raise InvalidRunPlan(f"Run plan contains duplicate run id '{run_id}'.")
        seen_ids.add(run_id)
        if submitted.preset not in valid_presets:
            raise InvalidRunPlan(
                f"Run plan contains unknown preset '{submitted.preset}'."
            )
        if submitted.dataset not in valid_datasets:
            raise InvalidRunPlan(
                f"Run plan contains unknown dataset '{submitted.dataset}'."
            )
        accepted.append(
            RunSpec(
                id=run_id,
                experiment_task=resolved.experiment_task_name,
                preset=submitted.preset,
                dataset=submitted.dataset,
                parameters=_submitted_parameters(
                    package,
                    submitted.preset,
                    submitted.overrides,
                ),
            )
        )

    return RunPlan(
        identity=package.identity,
        presets=resolved.preset_names,
        experiment_task=resolved.experiment_task_name,
        datasets=tuple(dataset_name(dataset) for dataset in resolved.datasets),
        overrides=resolved.serialized_overrides,
        search=None,
        runs=tuple(accepted),
        preset_searches=resolved.preset_searches(),
    )


__all__ = ["accept_run_plan", "plan_runs"]

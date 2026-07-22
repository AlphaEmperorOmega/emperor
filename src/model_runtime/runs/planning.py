from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from model_runtime.inspection import (
    InspectionError,
    canonicalize_overrides,
    configuration_schema,
    parse_overrides,
    preset_locks,
    reject_locked_overrides,
    search_space_schema,
    serialize_overrides,
)
from model_runtime.packages import (
    ModelPackage,
    abstract_config_class_error,
    config_key_to_model_param,
    dataset_name,
    iter_supported_config_keys,
    normalize_key,
    parse_config_value,
    serialize_config_value,
)
from model_runtime.runs.errors import InvalidRunPlan, InvalidRunRequest, PlanTooLarge
from model_runtime.runs.records import (
    PlanningBudget,
    RandomSource,
    RunParameter,
    RunPlan,
    RunRequest,
    RunSpec,
    SearchAxisSelection,
    SearchSpec,
    SubmittedRun,
)
from model_runtime.runs.search import PreparedSearch


@dataclass(frozen=True, slots=True)
class _SearchValue:
    serialized: Any
    parsed: Any


@dataclass(frozen=True, slots=True)
class _ParsedSearchAxis:
    key: str
    model_param: str
    values: tuple[_SearchValue, ...]


@dataclass(frozen=True, slots=True)
class _ParsedSearch:
    spec: SearchSpec
    axes: tuple[_ParsedSearchAxis, ...]
    prepared: PreparedSearch

    @property
    def model_params(self) -> set[str]:
        return {axis.model_param for axis in self.axes}


@dataclass(frozen=True, slots=True)
class _ResolvedRequest:
    experiment_task_name: str
    preset_names: tuple[str, ...]
    datasets: tuple[type, ...]
    searches: tuple[_ParsedSearch | None, ...]
    effective_overrides_by_preset: tuple[Mapping[str, Any], ...]
    serialized_overrides: Mapping[str, Any]

    @property
    def normalized_search(self) -> SearchSpec | None:
        return next(
            (search.spec for search in self.searches if search is not None),
            None,
        )


def _request_error(exc: Exception) -> InvalidRunRequest:
    return InvalidRunRequest(str(exc))


def _plan_error(exc: Exception) -> InvalidRunPlan:
    return InvalidRunPlan(str(exc))


def _resolve_presets(
    package: ModelPackage,
    raw_presets: Sequence[str],
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


def _parse_search_value(
    *,
    axis_key: str,
    config_module: Any,
    parse_key: str,
    raw_value: Any,
) -> _SearchValue:
    if raw_value is None:
        return _SearchValue(serialized=None, parsed=None)
    try:
        parsed = parse_config_value(
            config_module,
            parse_key,
            str(serialize_config_value(raw_value)),
        )
        if isinstance(parsed, type):
            abstract_error = abstract_config_class_error(parsed)
            if abstract_error is not None:
                raise ValueError(abstract_error)
    except Exception as exc:
        raise InvalidRunRequest(
            f"Invalid search value for axis '{axis_key}': {raw_value!r}. {exc}"
        ) from exc
    return _SearchValue(
        serialized=serialize_config_value(parsed),
        parsed=parsed,
    )


def _reject_axis_budget(
    selection: SearchAxisSelection,
    budget: PlanningBudget,
) -> None:
    if (
        selection.values is not None
        and budget.max_values_per_axis is not None
        and len(selection.values) > budget.max_values_per_axis
    ):
        raise PlanTooLarge(
            f"Search axis '{selection.key}' accepts at most "
            f"{budget.max_values_per_axis} selected values."
        )


def _parse_search(
    package: ModelPackage,
    preset_name: str,
    spec: SearchSpec | None,
    budget: PlanningBudget,
) -> _ParsedSearch | None:
    if spec is None:
        return None
    if spec.mode not in {"grid", "random"}:
        raise InvalidRunRequest("Training search mode must be 'grid' or 'random'.")
    if spec.axes == ():
        raise InvalidRunRequest("Training search requires at least one selected axis.")
    if (
        spec.axes is not None
        and budget.max_axes is not None
        and len(spec.axes) > budget.max_axes
    ):
        raise PlanTooLarge(
            f"Training search accepts at most {budget.max_axes} selected axes."
        )

    try:
        search_space = search_space_schema(package, preset_name)
    except InspectionError as exc:
        raise _request_error(exc) from exc
    axes_by_key = {normalize_key(axis.key): axis for axis in search_space.axes}
    config_keys_by_selection: dict[str, str] = {}
    for config_key in iter_supported_config_keys(package.runtime_defaults):
        config_keys_by_selection[normalize_key(config_key)] = config_key
        config_keys_by_selection[
            normalize_key(config_key_to_model_param(config_key))
        ] = config_key
    try:
        locks = preset_locks(package, preset_name)
    except InspectionError as exc:
        raise _request_error(exc) from exc
    if spec.axes is None:
        selections = tuple(
            SearchAxisSelection(key=axis.key)
            for axis in search_space.axes
            if not axis.locked
        )
    else:
        selections = spec.axes
    if spec.axes is None and budget.max_axes is not None:
        semantic_axes = {
            normalize_key(config_key_to_model_param(selection.key))
            for selection in selections
        }
        if len(semantic_axes) > budget.max_axes:
            raise PlanTooLarge(
                f"Training search accepts at most {budget.max_axes} selected axes."
            )

    parsed_axes: list[_ParsedSearchAxis] = []
    normalized_selections: list[SearchAxisSelection] = []
    axis_positions: dict[str, int] = {}
    implicit_full_search = spec.axes is None
    for selection in selections:
        _reject_axis_budget(selection, budget)
        normalized_key = normalize_key(selection.key)
        axis = axes_by_key.get(normalized_key)
        if axis is None and selection.allow_custom_values:
            config_key = config_keys_by_selection.get(normalized_key)
        else:
            config_key = None
        if axis is None and config_key is None:
            raise InvalidRunRequest(f"Unknown search axis '{selection.key}'.")

        if axis is not None:
            axis_key = axis.key
            model_param = config_key_to_model_param(axis.key)
            parse_module = package.metadata.search_space
            parse_key = axis.search_key
            default_values = axis.values
            allowed_values: tuple[Any, ...] | None = axis.values
            locked = axis.locked
            locked_value = axis.locked_value
        else:
            assert config_key is not None
            axis_key = config_key
            model_param = config_key_to_model_param(config_key)
            parse_module = package.runtime_defaults
            parse_key = config_key
            default_values = ()
            allowed_values = None
            lock = locks.get(model_param)
            locked = lock is not None
            locked_value = serialize_config_value(getattr(lock, "value", None))

        semantic_axis_key = normalize_key(model_param)
        existing_position = axis_positions.get(semantic_axis_key)
        if existing_position is not None and implicit_full_search:
            continue
        raw_values = default_values if selection.values is None else selection.values
        if not raw_values:
            raise InvalidRunRequest(
                f"Search axis '{axis_key}' requires at least one selected value."
            )
        if (
            budget.max_values_per_axis is not None
            and len(raw_values) > budget.max_values_per_axis
        ):
            raise PlanTooLarge(
                f"Search axis '{axis_key}' accepts at most "
                f"{budget.max_values_per_axis} selected values."
            )
        parsed_values = tuple(
            _parse_search_value(
                axis_key=axis_key,
                config_module=parse_module,
                parse_key=parse_key,
                raw_value=raw_value,
            )
            for raw_value in raw_values
        )
        serialized_values = tuple(value.serialized for value in parsed_values)
        if locked:
            lock_is_unchanged = selection.allow_custom_values and all(
                value == locked_value for value in serialized_values
            )
            if not lock_is_unchanged:
                raise InvalidRunRequest(
                    f"Search axis '{axis_key}' is locked by preset '{preset_name}'."
                )
        if not selection.allow_custom_values and allowed_values is not None:
            allowed_value_set = set(allowed_values)
            invalid_values = [
                value for value in serialized_values if value not in allowed_value_set
            ]
            if invalid_values:
                raise InvalidRunRequest(
                    f"Search axis '{axis_key}' received values outside its "
                    f"search space: {invalid_values}."
                )
        normalized_selection = SearchAxisSelection(
            key=axis_key,
            values=serialized_values,
        )
        parsed_axis = _ParsedSearchAxis(
            key=axis_key,
            model_param=model_param,
            values=parsed_values,
        )
        if existing_position is None:
            axis_positions[semantic_axis_key] = len(parsed_axes)
            normalized_selections.append(normalized_selection)
            parsed_axes.append(parsed_axis)
        else:
            normalized_selections[existing_position] = normalized_selection
            parsed_axes[existing_position] = parsed_axis

    random_samples: int | None = None
    if spec.mode == "random":
        random_samples = 10 if spec.random_samples is None else spec.random_samples
    prepared = PreparedSearch(
        axes=tuple(axis.values for axis in parsed_axes),
        mode=spec.mode,
        random_samples=random_samples,
    )
    return _ParsedSearch(
        spec=SearchSpec(
            mode=spec.mode,
            axes=tuple(normalized_selections),
            random_samples=random_samples,
        ),
        axes=tuple(parsed_axes),
        prepared=prepared,
    )


def _strip_searched_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any],
    searched_model_params: set[str],
) -> dict[str, Any]:
    if not searched_model_params:
        return dict(overrides)
    try:
        canonical = canonicalize_overrides(package, overrides)
    except InspectionError as exc:
        raise _request_error(exc) from exc
    return {
        key: value
        for key, value in canonical.items()
        if config_key_to_model_param(key) not in searched_model_params
    }


def _reject_conflicting_locks(
    package: ModelPackage,
    preset_name: str,
    parsed_overrides: Mapping[str, Any],
) -> None:
    try:
        locks = preset_locks(package, preset_name)
    except InspectionError as exc:
        raise _request_error(exc) from exc
    conflicts = sorted(
        key
        for key, value in parsed_overrides.items()
        if key in locks and value != getattr(locks[key], "value", None)
    )
    if not conflicts:
        return
    details = ", ".join(
        f"{key} ({getattr(locks[key], 'reason', '')})" for key in conflicts
    )
    raise InvalidRunRequest(
        f"Preset '{preset_name}' does not allow overriding locked fields: {details}"
    )


def _resolve_request(
    package: ModelPackage,
    request: RunRequest,
    budget: PlanningBudget,
) -> _ResolvedRequest:
    if not request.datasets:
        raise InvalidRunRequest("Training requires at least one selected dataset.")
    try:
        experiment_task = package.resolve_experiment_task(request.experiment_task)
        datasets = tuple(
            package.resolve_datasets(list(request.datasets), experiment_task)
        )
    except ValueError as exc:
        raise _request_error(exc) from exc
    preset_names = _resolve_presets(package, request.presets)
    searches = tuple(
        _parse_search(package, preset_name, request.search, budget)
        for preset_name in preset_names
    )
    searched_model_params = {
        model_param
        for search in searches
        if search is not None
        for model_param in search.model_params
    }
    top_level_effective_overrides = _strip_searched_overrides(
        package,
        request.overrides,
        searched_model_params,
    )
    try:
        serialized_overrides = serialize_overrides(
            package,
            top_level_effective_overrides,
        )
    except InspectionError as exc:
        raise _request_error(exc) from exc

    effective_overrides_by_preset: list[Mapping[str, Any]] = []
    for preset_name, search in zip(preset_names, searches, strict=True):
        effective_overrides = _strip_searched_overrides(
            package,
            request.overrides,
            search.model_params if search is not None else set(),
        )
        try:
            parsed_overrides = parse_overrides(
                package,
                effective_overrides,
            ).values
        except InspectionError as exc:
            raise _request_error(exc) from exc
        _reject_conflicting_locks(package, preset_name, parsed_overrides)
        effective_overrides_by_preset.append(effective_overrides)
    return _ResolvedRequest(
        experiment_task_name=package.task_name(experiment_task),
        preset_names=preset_names,
        datasets=datasets,
        searches=searches,
        effective_overrides_by_preset=tuple(effective_overrides_by_preset),
        serialized_overrides=serialized_overrides,
    )


def _ordered_parameters(
    package: ModelPackage,
    overrides: Mapping[str, Any],
    *,
    source: str,
) -> tuple[RunParameter, ...]:
    try:
        canonical = canonicalize_overrides(package, overrides)
        schema = configuration_schema(package)
    except InspectionError as exc:
        raise _request_error(exc) from exc
    supported_keys = iter_supported_config_keys(package.runtime_defaults)
    missing_keys = sorted(set(canonical) - set(supported_keys))
    if missing_keys:
        raise InvalidRunRequest(
            "Canonical Run overrides are not supported Runtime Defaults: "
            f"{', '.join(missing_keys)}."
        )
    schema_keys = [field.key for field in schema.fields]
    visible_keys = set(schema_keys)
    ordered_keys = schema_keys + [
        key for key in supported_keys if key not in visible_keys
    ]
    return tuple(
        RunParameter(
            key=key,
            value=serialize_config_value(canonical[key]),
            source=source,  # type: ignore[arg-type]
        )
        for key in ordered_keys
        if key in canonical
    )


def _search_parameters(
    search: _ParsedSearch,
    combination: tuple[_SearchValue, ...],
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
) -> None:
    limit = budget.max_materialized_runs
    if limit is not None and planned_run_count > limit:
        raise PlanTooLarge(
            "Training run plan is too large: "
            f"{planned_run_count} planned runs exceeds {limit}."
        )


def plan_runs(
    package: ModelPackage,
    request: RunRequest,
    *,
    random_source: RandomSource | None = None,
    budget: PlanningBudget | None = None,
) -> RunPlan:
    if not isinstance(package, ModelPackage):
        raise TypeError("Runs require a selected ModelPackage.")
    planning_budget = budget or PlanningBudget()
    resolved = _resolve_request(package, request, planning_budget)
    _reject_plan_budget(_planned_run_count(resolved), planning_budget)
    if random_source is None and any(
        search is not None and search.spec.mode == "random"
        for search in resolved.searches
    ):
        raise InvalidRunRequest("Random search requires an explicit random source.")

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

    return RunPlan(
        identity=package.identity,
        presets=resolved.preset_names,
        experiment_task=resolved.experiment_task_name,
        datasets=tuple(dataset_name(dataset) for dataset in resolved.datasets),
        overrides=resolved.serialized_overrides,
        search=resolved.normalized_search,
        runs=tuple(runs),
    )


def _submitted_parameters(
    package: ModelPackage,
    preset_name: str,
    overrides: Mapping[str, Any],
) -> tuple[RunParameter, ...]:
    try:
        parsed = parse_overrides(package, overrides).values
        reject_locked_overrides(package, preset_name, parsed)
        serialized = serialize_overrides(package, overrides)
    except (InspectionError, ValueError) as exc:
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
    if not isinstance(package, ModelPackage):
        raise TypeError("Runs require a selected ModelPackage.")
    planning_budget = budget or PlanningBudget()
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
        search=resolved.normalized_search,
        runs=tuple(accepted),
    )


__all__ = ["accept_run_plan", "plan_runs"]

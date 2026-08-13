from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, cast

from model_runtime.cli._wire_packages import (
    experiment_task_from_wire,
    identity_from_wire,
    identity_to_wire,
)
from model_runtime.cli._wire_search import (
    search_spec_from_wire,
    search_spec_from_wire_at,
    search_spec_to_wire,
    search_spec_to_wire_at,
)
from model_runtime.cli._wire_shared import (
    WireCodecError,
    json_mapping_from_wire,
    json_value_from_wire,
    json_value_to_wire,
    require_sequence_limit,
    wire_fields,
    wire_list,
    wire_literal,
    wire_optional_int,
    wire_optional_string,
    wire_string,
    wire_string_list,
)
from model_runtime.runs import (
    PlanningBudget,
    PresetSearch,
    RunParameter,
    RunPlan,
    RunRequest,
    RunResult,
    RunSpec,
    SubmittedRun,
)
from model_runtime.runs.records import RunParameterSource

_RUN_PARAMETER_SOURCES = {"override", "search"}
_DEFAULT_BUDGET = PlanningBudget()
_MAX_WIRE_RUNS = _DEFAULT_BUDGET.max_materialized_runs or 2_000
_MAX_WIRE_SELECTIONS = _MAX_WIRE_RUNS
_MAX_WIRE_PARAMETERS_PER_RUN = 1_024


def _mapping_to_wire(value: object, path: str) -> dict[str, Any]:
    encoded = json_value_to_wire(value, path=path)
    if not isinstance(encoded, dict):
        raise WireCodecError(f"{path} must be an object.")
    return cast(dict[str, Any], encoded)


def run_request_to_wire(request: RunRequest) -> dict[str, Any]:
    require_sequence_limit(
        request.presets,
        "$.presets",
        _MAX_WIRE_SELECTIONS,
    )
    require_sequence_limit(
        request.datasets,
        "$.datasets",
        _MAX_WIRE_SELECTIONS,
    )
    payload = {
        "presets": list(request.presets),
        "datasets": list(request.datasets),
        "experiment_task": (
            experiment_task_from_wire(request.experiment_task, "$.experiment_task")
            if request.experiment_task is not None
            else None
        ),
        "overrides": _mapping_to_wire(request.overrides, "$.overrides"),
        "search": search_spec_to_wire(request.search),
    }
    run_request_from_wire(payload)
    return payload


def run_request_from_wire(payload: object) -> RunRequest:
    raw = wire_fields(
        payload,
        path="$",
        required=("presets", "datasets"),
        optional=("experiment_task", "overrides", "search"),
    )
    experiment_task = raw.get("experiment_task")
    return RunRequest(
        presets=wire_string_list(
            wire_list(
                raw["presets"],
                "$.presets",
                maximum_items=_MAX_WIRE_SELECTIONS,
            ),
            "$.presets",
        ),
        datasets=wire_string_list(
            wire_list(
                raw["datasets"],
                "$.datasets",
                maximum_items=_MAX_WIRE_SELECTIONS,
            ),
            "$.datasets",
        ),
        experiment_task=(
            None
            if experiment_task is None
            else experiment_task_from_wire(experiment_task, "$.experiment_task")
        ),
        overrides=json_mapping_from_wire(
            raw.get("overrides", {}),
            path="$.overrides",
        ),
        search=search_spec_from_wire(raw.get("search")),
    )


def planning_budget_to_wire(budget: PlanningBudget) -> dict[str, int | None]:
    payload = {
        "max_axes": budget.max_axes,
        "max_values_per_axis": budget.max_values_per_axis,
        "max_materialized_runs": budget.max_materialized_runs,
    }
    planning_budget_from_wire(payload)
    return payload


def planning_budget_from_wire(payload: object) -> PlanningBudget:
    raw = wire_fields(
        payload,
        path="$",
        required=(),
        optional=("max_axes", "max_values_per_axis", "max_materialized_runs"),
    )
    defaults = PlanningBudget()
    budget = PlanningBudget(
        max_axes=(
            defaults.max_axes
            if "max_axes" not in raw
            else wire_optional_int(raw["max_axes"], "$.max_axes", minimum=1)
        ),
        max_values_per_axis=(
            defaults.max_values_per_axis
            if "max_values_per_axis" not in raw
            else wire_optional_int(
                raw["max_values_per_axis"],
                "$.max_values_per_axis",
                minimum=1,
            )
        ),
        max_materialized_runs=(
            defaults.max_materialized_runs
            if "max_materialized_runs" not in raw
            else wire_optional_int(
                raw["max_materialized_runs"],
                "$.max_materialized_runs",
                minimum=1,
            )
        ),
    )
    run_limit = budget.max_materialized_runs
    if run_limit is None or run_limit > _MAX_WIRE_RUNS:
        raise WireCodecError(
            "Run plan transport requires max_materialized_runs at most "
            f"{_MAX_WIRE_RUNS}."
        )
    return budget


def submitted_run_to_wire(run: SubmittedRun) -> dict[str, Any]:
    payload = {
        "id": run.id,
        "preset": run.preset,
        "dataset": run.dataset,
        "overrides": _mapping_to_wire(run.overrides, "$.overrides"),
    }
    submitted_run_from_wire(payload)
    return payload


def submitted_run_from_wire(payload: object) -> SubmittedRun:
    raw = wire_fields(
        payload,
        path="$",
        required=("preset", "dataset"),
        optional=("id", "overrides"),
    )
    return SubmittedRun(
        id=wire_optional_string(raw.get("id"), "$.id"),
        preset=wire_string(raw["preset"], "$.preset"),
        dataset=wire_string(raw["dataset"], "$.dataset"),
        overrides=json_mapping_from_wire(
            raw.get("overrides", {}),
            path="$.overrides",
        ),
    )


def submitted_runs_to_wire(runs: Sequence[SubmittedRun]) -> list[dict[str, Any]]:
    require_sequence_limit(runs, "$.runs", _MAX_WIRE_RUNS)
    return [submitted_run_to_wire(run) for run in runs]


def submitted_runs_from_wire(payload: object) -> tuple[SubmittedRun, ...]:
    return tuple(
        submitted_run_from_wire(item)
        for item in wire_list(
            payload,
            "$.runs",
            maximum_items=_MAX_WIRE_RUNS,
        )
    )


def _run_parameter_to_wire(parameter: RunParameter) -> dict[str, Any]:
    return {
        "key": parameter.key,
        "value": json_value_to_wire(parameter.value),
        "source": wire_literal(
            parameter.source,
            "$.runs[].parameters[].source",
            _RUN_PARAMETER_SOURCES,
        ),
    }


def _run_spec_to_wire(run: RunSpec) -> dict[str, Any]:
    require_sequence_limit(
        run.parameters,
        "$.runs[].parameters",
        _MAX_WIRE_PARAMETERS_PER_RUN,
    )
    return {
        "id": run.id,
        "experiment_task": experiment_task_from_wire(
            run.experiment_task,
            "$.runs[].experiment_task",
        ),
        "preset": run.preset,
        "dataset": run.dataset,
        "parameters": [
            _run_parameter_to_wire(parameter) for parameter in run.parameters
        ],
    }


def _preset_search_to_wire(
    entry: PresetSearch,
    index: int,
) -> dict[str, Any]:
    path = f"$.preset_searches[{index}]"
    return {
        "preset": entry.preset,
        "search": search_spec_to_wire_at(entry.search, f"{path}.search"),
    }


def run_plan_to_wire(plan: RunPlan) -> dict[str, Any]:
    require_sequence_limit(plan.runs, "$.runs", _MAX_WIRE_RUNS)
    require_sequence_limit(
        plan.presets,
        "$.presets",
        _MAX_WIRE_SELECTIONS,
    )
    require_sequence_limit(
        plan.datasets,
        "$.datasets",
        _MAX_WIRE_SELECTIONS,
    )
    require_sequence_limit(
        plan.preset_searches,
        "$.preset_searches",
        _MAX_WIRE_SELECTIONS,
    )
    payload = {
        "identity": identity_to_wire(plan.identity),
        "presets": list(plan.presets),
        "experiment_task": experiment_task_from_wire(
            plan.experiment_task,
            "$.experiment_task",
        ),
        "datasets": list(plan.datasets),
        "overrides": _mapping_to_wire(plan.overrides, "$.overrides"),
        "search": search_spec_to_wire(plan.search),
        "preset_searches": [
            _preset_search_to_wire(entry, index)
            for index, entry in enumerate(plan.preset_searches)
        ],
        "runs": [_run_spec_to_wire(run) for run in plan.runs],
    }
    run_plan_from_wire(payload)
    return payload


def _run_parameter_from_wire(item: object, path: str) -> RunParameter:
    parameter = wire_fields(
        item,
        path=path,
        required=("key", "value", "source"),
    )
    return RunParameter(
        key=wire_string(parameter["key"], f"{path}.key"),
        value=json_value_from_wire(parameter["value"], path=f"{path}.value"),
        source=cast(
            RunParameterSource,
            wire_literal(
                parameter["source"],
                f"{path}.source",
                _RUN_PARAMETER_SOURCES,
            ),
        ),
    )


def _run_parameters_from_wire(payload: object, path: str) -> list[RunParameter]:
    parameters: list[RunParameter] = []
    for index, item in enumerate(
        wire_list(
            payload,
            path,
            maximum_items=_MAX_WIRE_PARAMETERS_PER_RUN,
        )
    ):
        parameters.append(_run_parameter_from_wire(item, f"{path}[{index}]"))
    return parameters


def _run_spec_from_wire(item: object, index: int) -> RunSpec:
    path = f"$.runs[{index}]"
    run = wire_fields(
        item,
        path=path,
        required=("id", "experiment_task", "preset", "dataset", "parameters"),
    )
    parameters = _run_parameters_from_wire(run["parameters"], f"{path}.parameters")
    return RunSpec(
        id=wire_string(run["id"], f"{path}.id"),
        experiment_task=experiment_task_from_wire(
            run["experiment_task"],
            f"{path}.experiment_task",
        ),
        preset=wire_string(run["preset"], f"{path}.preset"),
        dataset=wire_string(run["dataset"], f"{path}.dataset"),
        parameters=tuple(parameters),
    )


def _run_specs_from_wire(payload: object) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for index, item in enumerate(
        wire_list(payload, "$.runs", maximum_items=_MAX_WIRE_RUNS)
    ):
        runs.append(_run_spec_from_wire(item, index))
    return runs


def _preset_searches_from_wire(payload: object) -> tuple[PresetSearch, ...]:
    entries: list[PresetSearch] = []
    for index, item in enumerate(
        wire_list(
            payload,
            "$.preset_searches",
            maximum_items=_MAX_WIRE_SELECTIONS,
        )
    ):
        path = f"$.preset_searches[{index}]"
        raw = wire_fields(
            item,
            path=path,
            required=("preset", "search"),
        )
        entries.append(
            PresetSearch(
                preset=wire_string(raw["preset"], f"{path}.preset"),
                search=search_spec_from_wire_at(
                    raw["search"],
                    f"{path}.search",
                ),
            )
        )
    return tuple(entries)


def run_plan_from_wire(payload: object) -> RunPlan:
    raw = wire_fields(
        payload,
        path="$",
        required=(
            "identity",
            "presets",
            "experiment_task",
            "datasets",
            "overrides",
            "search",
            "runs",
        ),
        optional=("preset_searches",),
    )
    runs = _run_specs_from_wire(raw["runs"])
    identity = identity_from_wire(raw["identity"])
    presets = wire_string_list(
        wire_list(
            raw["presets"],
            "$.presets",
            maximum_items=_MAX_WIRE_SELECTIONS,
        ),
        "$.presets",
    )
    experiment_task = experiment_task_from_wire(
        raw["experiment_task"],
        "$.experiment_task",
    )
    datasets = wire_string_list(
        wire_list(
            raw["datasets"],
            "$.datasets",
            maximum_items=_MAX_WIRE_SELECTIONS,
        ),
        "$.datasets",
    )
    overrides = json_mapping_from_wire(raw["overrides"], path="$.overrides")
    search = search_spec_from_wire(raw["search"])
    preset_searches = (
        ()
        if "preset_searches" not in raw
        else _preset_searches_from_wire(raw["preset_searches"])
    )
    return RunPlan(
        identity=identity,
        presets=presets,
        experiment_task=experiment_task,
        datasets=datasets,
        overrides=overrides,
        search=search,
        runs=tuple(runs),
        preset_searches=preset_searches,
    )


def run_result_to_wire(result: RunResult) -> dict[str, Any]:
    payload = {
        "run_id": result.run_id,
        "experiment_task": experiment_task_from_wire(
            result.experiment_task,
            "$.experiment_task",
        ),
        "preset": result.preset,
        "dataset": result.dataset,
        "log_dir": result.log_dir,
        "payload": _mapping_to_wire(result.payload, "$.payload"),
    }
    run_result_from_wire(payload)
    return payload


def run_result_from_wire(payload: object) -> RunResult:
    raw = wire_fields(
        payload,
        path="$",
        required=(
            "run_id",
            "experiment_task",
            "preset",
            "dataset",
            "log_dir",
            "payload",
        ),
    )
    return RunResult(
        run_id=wire_string(raw["run_id"], "$.run_id"),
        experiment_task=experiment_task_from_wire(
            raw["experiment_task"],
            "$.experiment_task",
        ),
        preset=wire_string(raw["preset"], "$.preset"),
        dataset=wire_string(raw["dataset"], "$.dataset"),
        log_dir=wire_string(raw["log_dir"], "$.log_dir"),
        payload=json_mapping_from_wire(raw["payload"], path="$.payload"),
    )


def run_results_to_wire(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    require_sequence_limit(results, "$.results", _MAX_WIRE_RUNS)
    return [run_result_to_wire(result) for result in results]


def random_state_to_wire(state: tuple[Any, ...]) -> list[Any]:
    encoded = json_value_to_wire(state, path="$.random_state")
    if not isinstance(encoded, list):
        raise WireCodecError("$.random_state must be a list.")
    encoded_list = cast(list[Any], encoded)
    random_state_from_wire(encoded_list)
    return encoded_list


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in cast(list[Any], value))
    return value


def random_state_from_wire(payload: object) -> tuple[Any, ...]:
    decoded = json_value_from_wire(payload, path="$.random_state")
    if not isinstance(decoded, list):
        raise WireCodecError("$.random_state must be a list.")
    state = _tuple_tree(decoded)
    try:
        random.Random().setstate(state)
    except (TypeError, ValueError) as exc:
        raise WireCodecError("$.random_state is invalid.") from exc
    return state


__all__ = [
    "planning_budget_from_wire",
    "planning_budget_to_wire",
    "random_state_from_wire",
    "random_state_to_wire",
    "run_plan_from_wire",
    "run_plan_to_wire",
    "run_request_from_wire",
    "run_request_to_wire",
    "run_result_from_wire",
    "run_result_to_wire",
    "run_results_to_wire",
    "search_spec_from_wire",
    "search_spec_to_wire",
    "submitted_run_from_wire",
    "submitted_run_to_wire",
    "submitted_runs_from_wire",
    "submitted_runs_to_wire",
]

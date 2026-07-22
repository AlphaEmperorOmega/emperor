from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from emperor.experiments import ExperimentTask, experiment_task_name
from model_runtime.cli._wire_packages import identity_from_wire, identity_to_wire
from model_runtime.cli._wire_shared import (
    WireCodecError,
    json_mapping_from_wire,
    json_value_from_wire,
    json_value_to_wire,
    wire_bool,
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
    RunParameter,
    RunPlan,
    RunRequest,
    RunResult,
    RunSpec,
    SearchAxisSelection,
    SearchSpec,
    SubmittedRun,
)

_EXPERIMENT_TASK_NAMES = {experiment_task_name(task) for task in ExperimentTask}
_RUN_PARAMETER_SOURCES = {"override", "search"}
_SEARCH_MODES = {"grid", "random"}


def _experiment_task(value: object, path: str) -> str:
    selected = wire_string(value, path)
    if selected not in _EXPERIMENT_TASK_NAMES:
        raise WireCodecError(f"{path} is not a supported Experiment Task.")
    return selected


def _mapping_to_wire(value: object, path: str) -> dict[str, Any]:
    encoded = json_value_to_wire(value, path=path)
    if not isinstance(encoded, dict):
        raise WireCodecError(f"{path} must be an object.")
    return encoded


def search_spec_to_wire(search: SearchSpec | None) -> dict[str, Any] | None:
    if search is None:
        return None
    payload = {
        "mode": wire_literal(search.mode, "$.search.mode", _SEARCH_MODES),
        "axes": (
            None
            if search.axes is None
            else [
                {
                    "key": axis.key,
                    "values": (
                        None
                        if axis.values is None
                        else [
                            json_value_to_wire(
                                value,
                                path="$.search.axes[].values[]",
                            )
                            for value in axis.values
                        ]
                    ),
                    "allow_custom_values": axis.allow_custom_values,
                }
                for axis in search.axes
            ]
        ),
        "random_samples": search.random_samples,
    }
    search_spec_from_wire(payload)
    return payload


def search_spec_from_wire(payload: object) -> SearchSpec | None:
    if payload is None:
        return None
    raw = wire_fields(
        payload,
        path="$.search",
        required=("mode",),
        optional=("axes", "random_samples"),
    )
    raw_axes = raw.get("axes")
    axes: tuple[SearchAxisSelection, ...] | None = None
    if raw_axes is not None:
        decoded_axes: list[SearchAxisSelection] = []
        for index, item in enumerate(wire_list(raw_axes, "$.search.axes")):
            path = f"$.search.axes[{index}]"
            axis = wire_fields(
                item,
                path=path,
                required=("key",),
                optional=("values", "allow_custom_values"),
            )
            raw_values = axis.get("values")
            decoded_axes.append(
                SearchAxisSelection(
                    key=wire_string(axis["key"], f"{path}.key"),
                    values=(
                        None
                        if raw_values is None
                        else tuple(
                            json_value_from_wire(
                                value,
                                path=f"{path}.values[{value_index}]",
                            )
                            for value_index, value in enumerate(
                                wire_list(raw_values, f"{path}.values")
                            )
                        )
                    ),
                    allow_custom_values=wire_bool(
                        axis.get("allow_custom_values", False),
                        f"{path}.allow_custom_values",
                    ),
                )
            )
        axes = tuple(decoded_axes)
    return SearchSpec(
        mode=wire_literal(raw["mode"], "$.search.mode", _SEARCH_MODES),
        axes=axes,
        random_samples=wire_optional_int(
            raw.get("random_samples"),
            "$.search.random_samples",
            minimum=1,
        ),
    )


def run_request_to_wire(request: RunRequest) -> dict[str, Any]:
    payload = {
        "presets": list(request.presets),
        "datasets": list(request.datasets),
        "experiment_task": (
            _experiment_task(request.experiment_task, "$.experiment_task")
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
        presets=wire_string_list(raw["presets"], "$.presets"),
        datasets=wire_string_list(raw["datasets"], "$.datasets"),
        experiment_task=(
            None
            if experiment_task is None
            else _experiment_task(experiment_task, "$.experiment_task")
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
    return PlanningBudget(
        max_axes=wire_optional_int(raw.get("max_axes"), "$.max_axes", minimum=1),
        max_values_per_axis=wire_optional_int(
            raw.get("max_values_per_axis"),
            "$.max_values_per_axis",
            minimum=1,
        ),
        max_materialized_runs=wire_optional_int(
            raw.get("max_materialized_runs"),
            "$.max_materialized_runs",
            minimum=1,
        ),
    )


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
    return [submitted_run_to_wire(run) for run in runs]


def submitted_runs_from_wire(payload: object) -> tuple[SubmittedRun, ...]:
    return tuple(submitted_run_from_wire(item) for item in wire_list(payload, "$.runs"))


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
    return {
        "id": run.id,
        "experiment_task": _experiment_task(
            run.experiment_task,
            "$.runs[].experiment_task",
        ),
        "preset": run.preset,
        "dataset": run.dataset,
        "parameters": [
            _run_parameter_to_wire(parameter) for parameter in run.parameters
        ],
    }


def run_plan_to_wire(plan: RunPlan) -> dict[str, Any]:
    payload = {
        "identity": identity_to_wire(plan.identity),
        "presets": list(plan.presets),
        "experiment_task": _experiment_task(
            plan.experiment_task,
            "$.experiment_task",
        ),
        "datasets": list(plan.datasets),
        "overrides": _mapping_to_wire(plan.overrides, "$.overrides"),
        "search": search_spec_to_wire(plan.search),
        "runs": [_run_spec_to_wire(run) for run in plan.runs],
    }
    run_plan_from_wire(payload)
    return payload


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
    )
    runs: list[RunSpec] = []
    for run_index, item in enumerate(wire_list(raw["runs"], "$.runs")):
        path = f"$.runs[{run_index}]"
        run = wire_fields(
            item,
            path=path,
            required=(
                "id",
                "experiment_task",
                "preset",
                "dataset",
                "parameters",
            ),
        )
        parameters: list[RunParameter] = []
        for parameter_index, parameter_item in enumerate(
            wire_list(run["parameters"], f"{path}.parameters")
        ):
            parameter_path = f"{path}.parameters[{parameter_index}]"
            parameter = wire_fields(
                parameter_item,
                path=parameter_path,
                required=("key", "value", "source"),
            )
            parameters.append(
                RunParameter(
                    key=wire_string(parameter["key"], f"{parameter_path}.key"),
                    value=json_value_from_wire(
                        parameter["value"],
                        path=f"{parameter_path}.value",
                    ),
                    source=wire_literal(
                        parameter["source"],
                        f"{parameter_path}.source",
                        _RUN_PARAMETER_SOURCES,
                    ),
                )
            )
        runs.append(
            RunSpec(
                id=wire_string(run["id"], f"{path}.id"),
                experiment_task=_experiment_task(
                    run["experiment_task"],
                    f"{path}.experiment_task",
                ),
                preset=wire_string(run["preset"], f"{path}.preset"),
                dataset=wire_string(run["dataset"], f"{path}.dataset"),
                parameters=tuple(parameters),
            )
        )
    return RunPlan(
        identity=identity_from_wire(raw["identity"]),
        presets=wire_string_list(raw["presets"], "$.presets"),
        experiment_task=_experiment_task(
            raw["experiment_task"],
            "$.experiment_task",
        ),
        datasets=wire_string_list(raw["datasets"], "$.datasets"),
        overrides=json_mapping_from_wire(raw["overrides"], path="$.overrides"),
        search=search_spec_from_wire(raw["search"]),
        runs=tuple(runs),
    )


def run_result_to_wire(result: RunResult) -> dict[str, Any]:
    payload = {
        "run_id": result.run_id,
        "experiment_task": _experiment_task(
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
        experiment_task=_experiment_task(
            raw["experiment_task"],
            "$.experiment_task",
        ),
        preset=wire_string(raw["preset"], "$.preset"),
        dataset=wire_string(raw["dataset"], "$.dataset"),
        log_dir=wire_string(raw["log_dir"], "$.log_dir"),
        payload=json_mapping_from_wire(raw["payload"], path="$.payload"),
    )


def run_results_to_wire(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    return [run_result_to_wire(result) for result in results]


def random_state_to_wire(state: tuple[Any, ...]) -> list[Any]:
    encoded = json_value_to_wire(state, path="$.random_state")
    if not isinstance(encoded, list):
        raise WireCodecError("$.random_state must be a list.")
    random_state_from_wire(encoded)
    return encoded


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
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

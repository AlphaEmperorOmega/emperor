from __future__ import annotations

from typing import Any

from model_runtime.cli._wire_inspection import (
    configuration_schema_from_wire,
    configuration_schema_to_wire,
    configuration_values_to_wire,
    inspection_result_from_wire,
    inspection_result_to_wire,
    preset_locks_to_wire,
    search_space_from_wire,
    search_space_to_wire,
)
from model_runtime.cli._wire_packages import (
    identity_from_wire,
    identity_to_wire,
    package_metadata_from_wire,
    package_metadata_to_wire,
)
from model_runtime.cli._wire_runs import (
    planning_budget_from_wire,
    planning_budget_to_wire,
    random_state_from_wire,
    random_state_to_wire,
    run_plan_from_wire,
    run_plan_to_wire,
    run_request_from_wire,
    run_request_to_wire,
    run_result_from_wire,
    run_result_to_wire,
    run_results_to_wire,
    search_spec_from_wire,
    search_spec_to_wire,
    submitted_run_from_wire,
    submitted_run_to_wire,
    submitted_runs_from_wire,
    submitted_runs_to_wire,
)
from model_runtime.cli._wire_shared import (
    PROTOCOL_VERSION,
    WireCodecError,
    json_value_from_wire,
    json_value_to_wire,
)
from model_runtime.inspection import (
    ConfigurationSchema,
    InspectionResult,
    SearchSpace,
)
from model_runtime.packages import ModelIdentity
from model_runtime.runs import (
    PlanningBudget,
    RunPlan,
    RunRequest,
    RunResult,
    SearchSpec,
    SubmittedRun,
)


def to_wire(value: Any) -> Any:
    """Encode one supported protocol record or an already JSON-shaped value."""

    encoders = (
        (ModelIdentity, identity_to_wire),
        (ConfigurationSchema, configuration_schema_to_wire),
        (SearchSpace, search_space_to_wire),
        (InspectionResult, inspection_result_to_wire),
        (SearchSpec, search_spec_to_wire),
        (RunRequest, run_request_to_wire),
        (PlanningBudget, planning_budget_to_wire),
        (SubmittedRun, submitted_run_to_wire),
        (RunPlan, run_plan_to_wire),
        (RunResult, run_result_to_wire),
    )
    for record_type, encoder in encoders:
        if isinstance(value, record_type):
            return encoder(value)
    if isinstance(value, (list, tuple)) and value:
        if all(isinstance(item, SubmittedRun) for item in value):
            return submitted_runs_to_wire(value)
        if all(isinstance(item, RunResult) for item in value):
            return run_results_to_wire(value)
    return json_value_to_wire(value)


__all__ = [
    "PROTOCOL_VERSION",
    "WireCodecError",
    "configuration_schema_from_wire",
    "configuration_schema_to_wire",
    "configuration_values_to_wire",
    "identity_from_wire",
    "identity_to_wire",
    "inspection_result_from_wire",
    "inspection_result_to_wire",
    "json_value_from_wire",
    "json_value_to_wire",
    "package_metadata_from_wire",
    "package_metadata_to_wire",
    "planning_budget_from_wire",
    "planning_budget_to_wire",
    "preset_locks_to_wire",
    "random_state_from_wire",
    "random_state_to_wire",
    "run_plan_from_wire",
    "run_plan_to_wire",
    "run_request_from_wire",
    "run_request_to_wire",
    "run_result_from_wire",
    "run_result_to_wire",
    "run_results_to_wire",
    "search_space_from_wire",
    "search_space_to_wire",
    "search_spec_from_wire",
    "search_spec_to_wire",
    "submitted_run_from_wire",
    "submitted_run_to_wire",
    "submitted_runs_from_wire",
    "submitted_runs_to_wire",
    "to_wire",
]

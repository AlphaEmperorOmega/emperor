from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any

from models.config_overrides import (
    config_key_to_model_param,
    normalize_key,
    parse_config_value,
)

from workbench.backend.inspector.config_classes import abstract_config_class_error
from workbench.backend.inspector.discovery import load_model_parts
from workbench.backend.inspector.errors import InspectorError
from workbench.backend.inspector.overrides import (
    resolve_override_key,
    supported_config_keys,
)
from workbench.backend.inspector.schema import search_space_schema
from workbench.backend.inspector.values import serialize_config_value
from workbench.backend.training_limits import (
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)


@dataclass(frozen=True)
class ParsedTrainingSearch:
    mode: str
    values: dict[str, list[Any]]
    search_overrides: dict[str, list[Any]]
    axis_keys: set[str]
    model_params: set[str]
    combination_count: int
    planned_run_count: int
    random_samples: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "values": self.values,
        }
        if self.random_samples is not None:
            payload["randomSamples"] = self.random_samples
        return payload


def _axis_by_key(model_name: str, preset_name: str) -> dict[str, dict[str, Any]]:
    axes = search_space_schema(model_name, preset_name)["axes"]
    return {normalize_key(axis["key"]): axis for axis in axes}


def _parse_search_value(
    search_space_module,
    axis: Mapping[str, Any],
    raw_value: Any,
) -> Any:
    if raw_value is None:
        return None
    try:
        parsed_value = parse_config_value(
            search_space_module,
            str(axis["searchKey"]),
            str(raw_value),
        )
        if isinstance(parsed_value, type):
            abstract_error = abstract_config_class_error(parsed_value)
            if abstract_error is not None:
                raise ValueError(abstract_error)
        return parsed_value
    except Exception as exc:
        raise InspectorError(
            f"Invalid search value for axis '{axis['key']}': {raw_value!r}. {exc}"
        ) from exc


def _deduplicate_axis_values(
    parsed_values: list[Any],
) -> tuple[list[Any], list[Any]]:
    deduplicated_parsed: list[Any] = []
    deduplicated_serialized: list[Any] = []
    seen: set[Any] = set()
    for parsed_value in parsed_values:
        serialized_value = serialize_config_value(parsed_value)
        if serialized_value in seen:
            continue
        seen.add(serialized_value)
        deduplicated_parsed.append(parsed_value)
        deduplicated_serialized.append(serialized_value)
    return deduplicated_parsed, deduplicated_serialized


def parse_training_search(
    model_name: str,
    preset_name: str,
    search: Mapping[str, Any] | None,
    *,
    dataset_count: int,
) -> ParsedTrainingSearch | None:
    if not search:
        return None

    mode = search.get("mode")
    if mode not in {"grid", "random"}:
        raise InspectorError("Training search mode must be 'grid' or 'random'.")

    values_payload = search.get("values")
    if not isinstance(values_payload, Mapping) or not values_payload:
        raise InspectorError("Training search requires at least one selected axis.")

    random_samples: int | None = None
    if mode == "random":
        raw_samples = search.get("randomSamples", 10)
        if isinstance(raw_samples, bool) or not isinstance(raw_samples, int):
            raise InspectorError("Random search sample count must be an integer.")
        if raw_samples < 1:
            raise InspectorError("Random search sample count must be at least 1.")
        random_samples = raw_samples

    parts = load_model_parts(model_name)
    axes = _axis_by_key(model_name, preset_name)
    parsed_values: dict[str, list[Any]] = {}
    serialized_values: dict[str, list[Any]] = {}
    search_overrides: dict[str, list[Any]] = {}
    model_params: set[str] = set()

    for raw_key, raw_values in values_payload.items():
        axis_key = normalize_key(str(raw_key))
        axis = axes.get(axis_key)
        if axis is None:
            raise InspectorError(f"Unknown search axis '{raw_key}'.")
        if axis.get("locked"):
            raise InspectorError(
                f"Search axis '{axis['key']}' is locked by preset '{preset_name}'."
            )
        if not isinstance(raw_values, list) or not raw_values:
            raise InspectorError(
                f"Search axis '{axis['key']}' requires at least one selected value."
            )
        if len(raw_values) > MAX_TRAINING_SEARCH_AXIS_VALUES:
            raise InspectorError(
                f"Search axis '{axis['key']}' accepts at most "
                f"{MAX_TRAINING_SEARCH_AXIS_VALUES} selected values."
            )

        allowed_values = {
            serialize_config_value(value) for value in axis.get("values", [])
        }
        parsed_axis_values = [
            _parse_search_value(
                getattr(parts, "search_space_module", parts.config_module),
                axis,
                raw_value,
            )
            for raw_value in raw_values
        ]
        (
            parsed_axis_values,
            serialized_axis_values,
        ) = _deduplicate_axis_values(parsed_axis_values)
        invalid_values = [
            value for value in serialized_axis_values if value not in allowed_values
        ]
        if invalid_values:
            raise InspectorError(
                f"Search axis '{axis['key']}' received values outside its "
                "search space: "
                f"{invalid_values}."
            )

        model_param = config_key_to_model_param(str(axis["configKey"]))
        parsed_values[str(axis["key"])] = parsed_axis_values
        serialized_values[str(axis["key"])] = serialized_axis_values
        search_overrides[model_param] = parsed_axis_values
        model_params.add(model_param)

    combination_count = prod(len(values) for values in parsed_values.values())
    run_count = combination_count
    if random_samples is not None:
        run_count = min(random_samples, combination_count)
    planned_run_count = run_count * dataset_count
    if planned_run_count > MAX_TRAINING_PLANNED_RUNS:
        raise InspectorError(
            "Training run plan is too large: "
            f"{planned_run_count} planned runs exceeds "
            f"{MAX_TRAINING_PLANNED_RUNS}."
        )

    return ParsedTrainingSearch(
        mode=str(mode),
        values=serialized_values,
        search_overrides=search_overrides,
        axis_keys=set(serialized_values),
        model_params=model_params,
        combination_count=combination_count,
        planned_run_count=planned_run_count,
        random_samples=random_samples,
    )


def strip_search_overrides(
    config_module,
    overrides: Mapping[str, Any] | None,
    search_model_params: set[str],
) -> dict[str, Any]:
    if not overrides or not search_model_params:
        return dict(overrides or {})

    supported = supported_config_keys(config_module)
    filtered: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        config_key, _legacy_residual_flag = resolve_override_key(
            normalize_key(str(raw_key)),
            supported,
        )
        if (
            config_key is not None
            and config_key_to_model_param(config_key) in search_model_params
        ):
            continue
        filtered[raw_key] = raw_value
    return filtered

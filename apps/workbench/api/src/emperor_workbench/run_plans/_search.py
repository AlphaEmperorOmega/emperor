from __future__ import annotations

from typing import Any

from model_runtime.inspection import SearchAxis
from model_runtime.packages import (
    config_key_to_model_param,
    normalize_key,
    serialize_config_value,
)
from model_runtime.runs import SearchAxisSelection, SearchSpec

from emperor_workbench.model_packages import ModelPackageFailure, SelectedModelPackage
from emperor_workbench.project_adapter import ModelPackageReference
from emperor_workbench.run_plans._errors import RunPlanFailure
from emperor_workbench.run_plans._limits import (
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_SEARCH_AXES,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)
from emperor_workbench.run_plans._records import TrainingSearch


def _parse_search_value(
    package: ModelPackageReference,
    axis: SearchAxis,
    raw_value: Any,
) -> Any:
    if raw_value is None:
        return None
    try:
        return package.client.call(
            "parse_search_value",
            {
                "model_id": package.catalog_key,
                "search_key": axis.search_key,
                "value": raw_value,
            },
        )
    except Exception as exc:
        raise RunPlanFailure(
            f"Invalid search value for axis '{axis.key}': {raw_value!r}. {exc}"
        ) from exc


def _deduplicate(values: list[Any]) -> tuple[Any, ...]:
    deduplicated: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return tuple(deduplicated)


def adapt_search(
    package: ModelPackageReference,
    preset_name: str,
    search: TrainingSearch | None,
) -> tuple[SearchSpec | None, set[str]]:
    if search is None:
        return None, set()
    if search.mode not in {"grid", "random"}:
        raise RunPlanFailure("Training search mode must be 'grid' or 'random'.")
    values_payload = search.values
    if not values_payload:
        raise RunPlanFailure("Training search requires at least one selected axis.")
    if len(values_payload) > MAX_TRAINING_SEARCH_AXES:
        raise RunPlanFailure(
            f"Training search accepts at most {MAX_TRAINING_SEARCH_AXES} selected axes."
        )

    random_samples: int | None = None
    if search.mode == "random":
        raw_samples = search.random_samples
        if raw_samples is None:
            raw_samples = 10
        if isinstance(raw_samples, bool) or not isinstance(raw_samples, int):
            raise RunPlanFailure("Random search sample count must be an integer.")
        if raw_samples < 1:
            raise RunPlanFailure("Random search sample count must be at least 1.")
        if raw_samples > MAX_TRAINING_PLANNED_RUNS:
            raise RunPlanFailure(
                "Random search sample count must be an integer between 1 and "
                f"{MAX_TRAINING_PLANNED_RUNS}."
            )
        random_samples = raw_samples

    try:
        semantic_search = SelectedModelPackage(package).search_space(preset_name)
    except ModelPackageFailure as exc:
        raise RunPlanFailure(exc.detail) from exc
    axes_by_key = {normalize_key(axis.key): axis for axis in semantic_search.axes}
    ordered_keys: list[str] = []
    selections: dict[str, SearchAxisSelection] = {}
    model_params: set[str] = set()
    for raw_key, raw_values in values_payload.items():
        axis = axes_by_key.get(normalize_key(str(raw_key)))
        if axis is None:
            raise RunPlanFailure(f"Unknown search axis '{raw_key}'.")
        if axis.locked:
            raise RunPlanFailure(
                f"Search axis '{axis.key}' is locked by preset '{preset_name}'."
            )
        if not isinstance(raw_values, list) or not raw_values:
            raise RunPlanFailure(
                f"Search axis '{axis.key}' requires at least one selected value."
            )
        if len(raw_values) > MAX_TRAINING_SEARCH_AXIS_VALUES:
            raise RunPlanFailure(
                f"Search axis '{axis.key}' accepts at most "
                f"{MAX_TRAINING_SEARCH_AXIS_VALUES} selected values."
            )

        serialized_values = _deduplicate(
            [_parse_search_value(package, axis, raw_value) for raw_value in raw_values]
        )
        allowed_values = {serialize_config_value(value) for value in axis.values}
        invalid_values = [
            value for value in serialized_values if value not in allowed_values
        ]
        if invalid_values:
            raise RunPlanFailure(
                f"Search axis '{axis.key}' received values outside its "
                f"search space: {invalid_values}."
            )

        canonical_key = normalize_key(axis.key)
        if canonical_key not in selections:
            ordered_keys.append(canonical_key)
        selections[canonical_key] = SearchAxisSelection(
            key=axis.key,
            values=serialized_values,
        )
        model_params.add(config_key_to_model_param(axis.key))

    return (
        SearchSpec(
            mode=search.mode,
            axes=tuple(selections[key] for key in ordered_keys),
            random_samples=random_samples,
        ),
        model_params,
    )


def search_from_spec(search: SearchSpec | None) -> TrainingSearch | None:
    if search is None:
        return None
    return TrainingSearch(
        mode=search.mode,
        values={axis.key: list(axis.values or ()) for axis in (search.axes or ())},
        random_samples=search.random_samples,
    )


__all__ = ["adapt_search", "search_from_spec"]

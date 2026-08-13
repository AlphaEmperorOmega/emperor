from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from model_runtime.packages import (
    ModelPackage,
    RuntimeDefaultsError,
    RuntimeDefaultsSpec,
    abstract_config_class_error,
    normalize_key,
)
from model_runtime.runs.errors import InvalidRunRequest, PlanTooLarge
from model_runtime.runs.records import PlanningBudget, SearchAxisSelection, SearchSpec
from model_runtime.runs.search import PreparedSearch


@dataclass(frozen=True, slots=True)
class SearchValue:
    serialized: Any
    parsed: Any


@dataclass(frozen=True, slots=True)
class ParsedSearchAxis:
    key: str
    model_param: str
    values: tuple[SearchValue, ...]


@dataclass(frozen=True, slots=True)
class ParsedSearch:
    spec: SearchSpec
    axes: tuple[ParsedSearchAxis, ...]
    prepared: PreparedSearch

    @property
    def model_params(self) -> set[str]:
        return {axis.model_param for axis in self.axes}


@dataclass(frozen=True, slots=True)
class _SearchContext:
    preset_name: str
    budget: PlanningBudget
    runtime_defaults: RuntimeDefaultsSpec
    search_axes: tuple[_AxisDefinition, ...]
    axes_by_key: Mapping[str, _AxisDefinition]
    locks: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _AxisDefinition:
    key: str
    config_key: str
    model_param: str
    search_key: str | None
    default_values: tuple[Any, ...]
    allowed_values: tuple[Any, ...] | None
    locked: bool
    locked_value: Any

    @property
    def semantic_key(self) -> str:
        return normalize_key(self.model_param)


@dataclass(frozen=True, slots=True)
class _ParsedAxisSelection:
    semantic_key: str
    normalized: SearchAxisSelection
    axis: ParsedSearchAxis


def _request_error(exc: Exception) -> InvalidRunRequest:
    return InvalidRunRequest(str(exc))


def _validate_search_spec(spec: SearchSpec, budget: PlanningBudget) -> None:
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


def _search_context(
    package: ModelPackage,
    preset_name: str,
    budget: PlanningBudget,
) -> _SearchContext:
    try:
        runtime_defaults = package.runtime_defaults_spec
        locks = runtime_defaults.preset_locks(preset_name)
    except RuntimeDefaultsError as exc:
        raise _request_error(exc) from exc
    search_axes = tuple(
        _search_axis_definition(runtime_defaults, locks, search_key, values)
        for search_key, values in runtime_defaults.ordered_search_items()
    )
    return _SearchContext(
        preset_name=preset_name,
        budget=budget,
        runtime_defaults=runtime_defaults,
        search_axes=search_axes,
        axes_by_key={normalize_key(axis.key): axis for axis in search_axes},
        locks=locks,
    )


def _selected_axes(
    context: _SearchContext,
    spec: SearchSpec,
) -> tuple[SearchAxisSelection, ...]:
    if spec.axes is not None:
        return spec.axes
    selections = tuple(
        SearchAxisSelection(key=axis.key)
        for axis in context.search_axes
        if not axis.locked
    )
    limit = context.budget.max_axes
    if limit is not None:
        semantic_axes = {
            normalize_key(context.runtime_defaults.model_parameter(selection.key))
            for selection in selections
        }
        if len(semantic_axes) > limit:
            raise PlanTooLarge(
                f"Training search accepts at most {limit} selected axes."
            )
    return selections


def _reject_selected_value_budget(
    selection: SearchAxisSelection,
    budget: PlanningBudget,
) -> None:
    values = selection.values
    limit = budget.max_values_per_axis
    if values is not None and limit is not None and len(values) > limit:
        raise PlanTooLarge(
            f"Search axis '{selection.key}' accepts at most {limit} selected values."
        )


def _search_axis_definition(
    runtime_defaults: RuntimeDefaultsSpec,
    locks: Mapping[str, Any],
    search_key: str,
    values: tuple[Any, ...],
) -> _AxisDefinition:
    config_key = search_key.removeprefix("SEARCH_SPACE_")
    model_param = runtime_defaults.model_parameter(config_key)
    lock = locks.get(model_param)
    serialized_values = tuple(
        runtime_defaults.serialize_value(value) for value in values
    )
    return _AxisDefinition(
        key=config_key,
        config_key=config_key,
        model_param=model_param,
        search_key=search_key,
        default_values=serialized_values,
        allowed_values=serialized_values,
        locked=lock is not None,
        locked_value=runtime_defaults.serialize_value(
            getattr(lock, "value", None)
        ),
    )


def _custom_axis_definition(
    context: _SearchContext,
    config_key: str,
) -> _AxisDefinition:
    model_param = context.runtime_defaults.model_parameter(config_key)
    lock = context.locks.get(model_param)
    return _AxisDefinition(
        key=config_key,
        config_key=config_key,
        model_param=model_param,
        search_key=None,
        default_values=(),
        allowed_values=None,
        locked=lock is not None,
        locked_value=context.runtime_defaults.serialize_value(
            getattr(lock, "value", None)
        ),
    )


def _axis_definition(
    context: _SearchContext,
    selection: SearchAxisSelection,
) -> _AxisDefinition:
    normalized_key = normalize_key(selection.key)
    axis = context.axes_by_key.get(normalized_key)
    if axis is not None:
        return axis
    if selection.allow_custom_values:
        config_key = context.runtime_defaults.keys_by_alias.get(normalized_key)
        if config_key is not None:
            return _custom_axis_definition(context, config_key)
    raise InvalidRunRequest(f"Unknown search axis '{selection.key}'.")


def _parse_search_value(
    context: _SearchContext,
    definition: _AxisDefinition,
    raw_value: Any,
) -> SearchValue:
    try:
        parsed = context.runtime_defaults.parse_search_value(
            definition.config_key,
            raw_value,
            search_key=definition.search_key,
        )
        if isinstance(parsed, type):
            abstract_error = abstract_config_class_error(parsed)
            if abstract_error is not None:
                raise ValueError(abstract_error)
    except Exception as exc:
        raise InvalidRunRequest(
            f"Invalid search value for axis '{definition.key}': {raw_value!r}. {exc}"
        ) from exc
    return SearchValue(
        serialized=context.runtime_defaults.serialize_value(parsed),
        parsed=parsed,
    )


def _raw_axis_values(
    context: _SearchContext,
    selection: SearchAxisSelection,
    definition: _AxisDefinition,
) -> Sequence[Any]:
    values = definition.default_values if selection.values is None else selection.values
    if not values:
        raise InvalidRunRequest(
            f"Search axis '{definition.key}' requires at least one selected value."
        )
    limit = context.budget.max_values_per_axis
    if limit is not None and len(values) > limit:
        raise PlanTooLarge(
            f"Search axis '{definition.key}' accepts at most {limit} selected values."
        )
    return values


def _reject_changed_lock(
    context: _SearchContext,
    selection: SearchAxisSelection,
    definition: _AxisDefinition,
    serialized_values: tuple[Any, ...],
) -> None:
    if not definition.locked:
        return
    lock_is_unchanged = selection.allow_custom_values and all(
        value == definition.locked_value for value in serialized_values
    )
    if not lock_is_unchanged:
        raise InvalidRunRequest(
            f"Search axis '{definition.key}' is locked by preset "
            f"'{context.preset_name}'."
        )


def _reject_disallowed_values(
    selection: SearchAxisSelection,
    definition: _AxisDefinition,
    serialized_values: tuple[Any, ...],
) -> None:
    if selection.allow_custom_values or definition.allowed_values is None:
        return
    allowed_values = set(definition.allowed_values)
    invalid_values = [
        value for value in serialized_values if value not in allowed_values
    ]
    if invalid_values:
        raise InvalidRunRequest(
            f"Search axis '{definition.key}' received values outside its "
            f"search space: {invalid_values}."
        )


def _requires_custom_value_authorization(
    selection: SearchAxisSelection,
    definition: _AxisDefinition,
    serialized_values: tuple[Any, ...],
) -> bool:
    if not selection.allow_custom_values:
        return False
    if definition.locked or definition.allowed_values is None:
        return True
    return any(value not in definition.allowed_values for value in serialized_values)


def _parse_axis_selection(
    context: _SearchContext,
    selection: SearchAxisSelection,
    definition: _AxisDefinition,
) -> _ParsedAxisSelection:
    raw_values = _raw_axis_values(context, selection, definition)
    parsed_values = tuple(
        _parse_search_value(context, definition, value) for value in raw_values
    )
    serialized_values = tuple(value.serialized for value in parsed_values)
    _reject_changed_lock(context, selection, definition, serialized_values)
    _reject_disallowed_values(selection, definition, serialized_values)
    return _ParsedAxisSelection(
        semantic_key=definition.semantic_key,
        normalized=SearchAxisSelection(
            key=definition.key,
            values=serialized_values,
            allow_custom_values=_requires_custom_value_authorization(
                selection,
                definition,
                serialized_values,
            ),
        ),
        axis=ParsedSearchAxis(
            key=definition.key,
            model_param=definition.model_param,
            values=parsed_values,
        ),
    )


def _parse_axes(
    context: _SearchContext,
    selections: Sequence[SearchAxisSelection],
    *,
    implicit_full_search: bool,
) -> tuple[tuple[SearchAxisSelection, ...], tuple[ParsedSearchAxis, ...]]:
    normalized: list[SearchAxisSelection] = []
    parsed: list[ParsedSearchAxis] = []
    positions: dict[str, int] = {}
    for selection in selections:
        _reject_selected_value_budget(selection, context.budget)
        definition = _axis_definition(context, selection)
        existing_position = positions.get(definition.semantic_key)
        if existing_position is not None and implicit_full_search:
            continue
        parsed_selection = _parse_axis_selection(context, selection, definition)
        if existing_position is None:
            positions[parsed_selection.semantic_key] = len(parsed)
            normalized.append(parsed_selection.normalized)
            parsed.append(parsed_selection.axis)
        else:
            normalized[existing_position] = parsed_selection.normalized
            parsed[existing_position] = parsed_selection.axis
    return tuple(normalized), tuple(parsed)


def _random_samples(spec: SearchSpec) -> int | None:
    if spec.mode != "random":
        return None
    return 10 if spec.random_samples is None else spec.random_samples


def parse_search(
    package: ModelPackage,
    preset_name: str,
    spec: SearchSpec | None,
    budget: PlanningBudget,
) -> ParsedSearch | None:
    if spec is None:
        return None
    _validate_search_spec(spec, budget)
    context = _search_context(package, preset_name, budget)
    selections = _selected_axes(context, spec)
    normalized, parsed_axes = _parse_axes(
        context,
        selections,
        implicit_full_search=spec.axes is None,
    )
    random_samples = _random_samples(spec)
    return ParsedSearch(
        spec=SearchSpec(
            mode=spec.mode,
            axes=normalized,
            random_samples=random_samples,
        ),
        axes=parsed_axes,
        prepared=PreparedSearch(
            axes=tuple(axis.values for axis in parsed_axes),
            mode=spec.mode,
            random_samples=random_samples,
        ),
    )


__all__ = ["ParsedSearch", "SearchValue", "parse_search"]

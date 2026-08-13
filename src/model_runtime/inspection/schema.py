from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from enum import Enum
from types import NoneType, UnionType
from typing import Any, Union, cast, get_args, get_origin

from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.field_descriptions import config_field_description
from model_runtime.inspection.records import (
    ConfigurationField,
    ConfigurationFieldCondition,
    ConfigurationSchema,
    SearchAxis,
    SearchSpace,
)
from model_runtime.inspection.runtime_defaults import (
    RuntimeDefaultsSpec,
    apply_runtime_defaults,
    raise_runtime_defaults_inspection_error,
    runtime_defaults_spec,
)
from model_runtime.packages import (
    ModelPackage,
    RuntimeDefaultsError,
    abstract_config_class_error,
    config_key_to_flag,
)

DEFAULT_SECTION = "General"
PRIMITIVE_ANNOTATION_KINDS = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "string",
}


def _field_section_path(
    key: str,
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    raw_path = metadata.get("sectionPath")
    if isinstance(raw_path, list):
        raw_path_items = cast(list[object], raw_path)
        path = tuple(item for item in raw_path_items if isinstance(item, str) and item)
        if path:
            return path
    raise InspectionError(
        f"Config field {key!r} is missing source heading metadata. "
        "Add a markdown heading comment above the field or preserve imported "
        "config field names."
    )


def _annotation_classes(annotation: Any) -> list[type[Any]]:
    if annotation is None:
        return []
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {UnionType, Union}:
        return [item for arg in args for item in _annotation_classes(arg)]
    if annotation is NoneType:
        return []
    if origin is type and args and isinstance(args[0], type):
        return [cast(type[Any], args[0])]
    if inspect.isclass(annotation):
        return [annotation]
    return []


def _annotation_is_nullable(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in {UnionType, Union} and any(
        arg is NoneType for arg in get_args(annotation)
    )


def _annotation_primitive_kind(annotation: Any) -> str | None:
    classes = _annotation_classes(annotation)
    for primitive_type, kind in PRIMITIVE_ANNOTATION_KINDS.items():
        if primitive_type in classes:
            return kind
    return None


def _concrete_value_kind(value: Any) -> str | None:
    kind: str | None = None
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int) and not isinstance(value, bool):
        kind = "int"
    elif isinstance(value, float):
        kind = "float"
    elif isinstance(value, str):
        kind = "string"
    elif isinstance(value, Enum):
        kind = "enum"
    elif inspect.isclass(value):
        kind = "class"
    return kind


def _none_value_kind(annotation: Any, annotation_classes: list[type[Any]]) -> str:
    primitive_kind = _annotation_primitive_kind(annotation)
    if primitive_kind is not None:
        return primitive_kind
    kind = "unknown"
    if any(issubclass(cls, Enum) for cls in annotation_classes):
        kind = "enum"
    elif any(cls not in PRIMITIVE_ANNOTATION_KINDS for cls in annotation_classes):
        kind = "class"
    return kind


def _value_kind(value: Any, annotation: Any) -> str:
    annotation_classes = _annotation_classes(annotation)
    kind = _concrete_value_kind(value)
    if kind is not None:
        return kind
    if value is None:
        return _none_value_kind(annotation, annotation_classes)
    return "list" if isinstance(value, list) else "unknown"


def _enum_choices(value: Any, annotation: Any) -> list[str]:
    enum_type = type(value) if isinstance(value, Enum) else None
    if enum_type is None:
        for cls in _annotation_classes(annotation):
            if issubclass(cls, Enum):
                enum_type = cls
                break
    return list(enum_type.__members__) if enum_type is not None else []


def _class_choice_name(value: Any) -> str | None:
    if value is None:
        return None
    return value.__name__ if inspect.isclass(value) else None


class _ChoiceResolver:
    def __init__(self, spec: RuntimeDefaultsSpec) -> None:
        self._spec = spec

    def _search_space_class_choices(
        self,
        key: str | None,
        available_choices: set[str],
    ) -> list[str]:
        if key is None:
            return []
        values = self._spec.search_source_value(f"SEARCH_SPACE_{key}")
        if not isinstance(values, list):
            return []
        ordered: list[str] = []
        seen: set[str] = set()
        for value in cast(list[object], values):
            choice = _class_choice_name(value)
            if choice is None or choice not in available_choices or choice in seen:
                continue
            ordered.append(choice)
            seen.add(choice)
        return ordered

    def _class_choices(
        self,
        annotation: Any,
        current_value: Any,
        key: str | None,
    ) -> list[str]:
        expected: list[type[Any]] = [
            cls
            for cls in _annotation_classes(annotation)
            if not issubclass(cls, Enum) and cls not in PRIMITIVE_ANNOTATION_KINDS
        ]
        if inspect.isclass(current_value):
            expected.append(current_value)

        choices: list[str] = []
        for candidate in self._spec.configuration_values():
            if not inspect.isclass(candidate):
                continue
            if abstract_config_class_error(candidate) is not None:
                continue
            if not expected or any(
                candidate is expected_type or issubclass(candidate, expected_type)
                for expected_type in expected
            ):
                choices.append(candidate.__name__)
        available = set(choices)
        ordered = self._search_space_class_choices(key, available)
        if ordered:
            return ordered + sorted(available - set(ordered))
        return sorted(available)

    def choices_for(
        self,
        value: Any,
        annotation: Any,
        kind: str,
        key: str | None = None,
    ) -> list[Any]:
        if kind == "bool":
            return [True, False]
        if kind == "enum":
            return _enum_choices(value, annotation)
        if kind == "class":
            return self._class_choices(annotation, value, key)
        return []


def preset_locks(
    package: ModelPackage,
    preset_name: str | None,
) -> dict[str, Any]:
    return apply_runtime_defaults(
        package,
        lambda spec: spec.preset_locks(preset_name),
    )


def _unique_presets(
    preset_name: str | None,
    preset_names: tuple[str, ...] | list[str] | None,
) -> list[str]:
    raw_names = preset_names if preset_names else ([preset_name] if preset_name else [])
    names: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_names:
        name = raw_name.strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _preset_lock_details(
    spec: RuntimeDefaultsSpec,
    preset_name: str | None,
    preset_names: tuple[str, ...] | list[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    details: dict[str, list[dict[str, Any]]] = {}
    for selected_name in _unique_presets(preset_name, preset_names):
        preset, locks = spec.resolve_preset_locks(selected_name)
        assert preset is not None
        for field, lock in locks.items():
            details.setdefault(field, []).append(
                {
                    "preset": preset.name,
                    "value": lock.value,
                    "reason": lock.reason,
                }
            )
    return details


def _shared_locked_value(
    spec: RuntimeDefaultsSpec,
    lock_details: list[dict[str, Any]],
) -> Any:
    if not lock_details:
        return None
    values = [spec.serialize_value(detail["value"]) for detail in lock_details]
    first = values[0]
    return first if all(value == first for value in values) else None


class _ApplicabilityParser:
    def __init__(
        self,
        spec: RuntimeDefaultsSpec,
        supported_keys: Sequence[str],
    ) -> None:
        self._spec = spec
        self._supported_keys = supported_keys
        self._known_keys: set[str] = set()
        self._dependencies: dict[str, tuple[str, ...]] = {}
        self._visiting: list[str] = []
        self._visited: set[str] = set()

    def parse(self) -> dict[str, tuple[ConfigurationFieldCondition, ...]]:
        raw_applicability = self._raw_applicability()
        self._known_keys = set(self._supported_keys)
        applicability: dict[str, tuple[ConfigurationFieldCondition, ...]] = {}
        for target_value, raw_conditions in raw_applicability.items():
            target_key = self._target_key(target_value)
            conditions, controller_keys = self._conditions_for(
                target_key,
                raw_conditions,
            )
            applicability[target_key] = conditions
            self._dependencies[target_key] = controller_keys
        self._reject_cycles()
        return applicability

    def _raw_applicability(self) -> Mapping[object, object]:
        raw_applicability = self._spec.configuration_applicability()
        if not isinstance(raw_applicability, Mapping):
            raise InspectionError(
                f"Model {self._spec.package.catalog_key!r} "
                "CONFIG_FIELD_APPLICABILITY must be a mapping."
            )
        return cast(Mapping[object, object], raw_applicability)

    def _target_key(self, target_value: object) -> str:
        if not isinstance(target_value, str) or target_value not in self._known_keys:
            raise InspectionError(
                f"Model {self._spec.package.catalog_key!r} "
                "CONFIG_FIELD_APPLICABILITY contains unknown target key "
                f"{target_value!r}."
            )
        return target_value

    def _conditions_for(
        self,
        target_key: str,
        raw_conditions: object,
    ) -> tuple[tuple[ConfigurationFieldCondition, ...], tuple[str, ...]]:
        if not isinstance(raw_conditions, Mapping):
            raise InspectionError(
                "Applicability metadata for Runtime Defaults field "
                f"{target_key!r} must be a mapping of controller keys to values."
            )
        conditions: list[ConfigurationFieldCondition] = []
        controller_keys: list[str] = []
        conditions_mapping = cast(Mapping[object, object], raw_conditions)
        for controller_value, raw_values in conditions_mapping.items():
            condition = self._condition(target_key, controller_value, raw_values)
            conditions.append(condition)
            controller_keys.append(condition.key)
        return tuple(conditions), tuple(controller_keys)

    def _condition(
        self,
        target_key: str,
        controller_value: object,
        raw_values: object,
    ) -> ConfigurationFieldCondition:
        if (
            not isinstance(controller_value, str)
            or controller_value not in self._known_keys
        ):
            raise InspectionError(
                "Applicability metadata for Runtime Defaults field "
                f"{target_key!r} contains unknown controller key "
                f"{controller_value!r}."
            )
        if controller_value == target_key:
            raise InspectionError(
                f"Runtime Defaults field {target_key!r} cannot make its "
                "applicability depend on itself."
            )
        if not isinstance(raw_values, (list, tuple)):
            raise InspectionError(
                "Applicability values for Runtime Defaults field "
                f"{target_key!r} controlled by {controller_value!r} must be a "
                "list or tuple."
            )
        if not raw_values:
            raise InspectionError(
                "Applicability values for Runtime Defaults field "
                f"{target_key!r} controlled by {controller_value!r} cannot be "
                "empty."
            )
        serialized_values = cast(Sequence[object], raw_values)
        return ConfigurationFieldCondition(
            key=controller_value,
            values=tuple(
                self._spec.serialize_value(value) for value in serialized_values
            ),
        )

    def _reject_cycles(self) -> None:
        for target_key in self._dependencies:
            self._visit(target_key)

    def _visit(self, key: str) -> None:
        if key in self._visited:
            return
        if key in self._visiting:
            cycle_start = self._visiting.index(key)
            cycle = [*self._visiting[cycle_start:], key]
            raise InspectionError(
                "CONFIG_FIELD_APPLICABILITY contains a dependency cycle: "
                + " -> ".join(cycle)
                + "."
            )
        self._visiting.append(key)
        for controller_key in self._dependencies.get(key, ()):
            self._visit(controller_key)
        self._visiting.pop()
        self._visited.add(key)


def _configuration_field_applicability(
    spec: RuntimeDefaultsSpec,
    supported_keys: Sequence[str],
) -> dict[str, tuple[ConfigurationFieldCondition, ...]]:
    return _ApplicabilityParser(spec, supported_keys).parse()


def configuration_schema(
    package: ModelPackage,
    preset: str | None = None,
) -> ConfigurationSchema:
    spec = runtime_defaults_spec(package)
    try:
        return _configuration_schema(spec, preset)
    except RuntimeDefaultsError as exc:
        raise_runtime_defaults_inspection_error(exc)


def _supported_configuration_keys(
    spec: RuntimeDefaultsSpec,
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[
    list[str],
    dict[str, tuple[ConfigurationFieldCondition, ...]],
]:
    supported_keys = [
        key for key in spec.supported_keys if key not in spec.skipped_schema_keys
    ]
    applicability = _configuration_field_applicability(spec, supported_keys)
    missing = [key for key in supported_keys if key not in metadata]
    if missing:
        raise InspectionError(
            f"Config fields for model {spec.package.catalog_key!r} are missing "
            f"source heading metadata: {', '.join(missing)}"
        )
    ordered_keys = [
        key
        for key in spec.ordered_configuration_keys()
        if key not in spec.skipped_schema_keys
    ]
    return ordered_keys, applicability


def _configuration_schema(
    spec: RuntimeDefaultsSpec,
    preset: str | None,
) -> ConfigurationSchema:
    package = spec.package
    locks = spec.preset_locks(preset)
    metadata = spec.configuration_metadata
    supported_keys, applicability = _supported_configuration_keys(spec, metadata)

    choice_resolver = _ChoiceResolver(spec)
    fields: list[ConfigurationField] = []
    for key in supported_keys:
        value = spec.current_value(key)
        annotation = spec.annotations.get(key)
        kind = _value_kind(value, annotation)
        section_path = _field_section_path(key, metadata.get(key, {}))
        nullable = value is None or _annotation_is_nullable(annotation)
        lock = locks.get(spec.model_parameter(key))
        locked_value = lock.value if lock is not None else None
        fields.append(
            ConfigurationField(
                key=key,
                flag=config_key_to_flag(key),
                section_path=section_path,
                description=config_field_description(
                    key,
                    section=section_path[-1],
                    kind=kind,
                    nullable=nullable,
                    default=value,
                ),
                value_type=kind,
                default=spec.serialize_value(value),
                nullable=nullable,
                choices=tuple(
                    spec.serialize_value(choice)
                    for choice in choice_resolver.choices_for(
                        value,
                        annotation,
                        kind,
                        key,
                    )
                ),
                applicable_when=applicability.get(key, ()),
                maximum=spec.maximum_for(key),
                locked=lock is not None,
                locked_value=(
                    spec.serialize_value(locked_value) if lock is not None else None
                ),
                locked_reason=(lock.reason if lock is not None else ""),
            )
        )
    return ConfigurationSchema(identity=package.identity, fields=tuple(fields))


def _search_axis_kind(
    spec: RuntimeDefaultsSpec,
    config_key: str,
    values: Sequence[Any],
) -> str:
    if spec.has_current_value(config_key):
        return _value_kind(
            spec.current_value(config_key),
            spec.annotations.get(config_key),
        )
    sample = next((value for value in values if value is not None), None)
    return _value_kind(
        sample,
        spec.annotations.get(config_key) or spec.search_annotations.get(config_key),
    )


def search_space_schema(
    package: ModelPackage,
    preset: str | None = None,
    presets: tuple[str, ...] | list[str] | None = None,
) -> SearchSpace:
    spec = runtime_defaults_spec(package)
    try:
        lock_details_by_param = _preset_lock_details(spec, preset, presets)
        config_fields = {
            field.key: field for field in _configuration_schema(spec, preset).fields
        }
    except RuntimeDefaultsError as exc:
        raise_runtime_defaults_inspection_error(exc)
    metadata = spec.search_metadata
    axes: list[SearchAxis] = []
    for search_key, values in spec.ordered_search_items():
        config_key = search_key.removeprefix("SEARCH_SPACE_")
        field = config_fields.get(config_key)
        lock_details = lock_details_by_param.get(
            spec.model_parameter(config_key),
            [],
        )
        lock_reasons = tuple(
            str(detail["reason"]) for detail in lock_details if str(detail["reason"])
        )
        locked_by_presets = tuple(
            str(detail["preset"]) for detail in lock_details if str(detail["preset"])
        )
        axes.append(
            SearchAxis(
                key=config_key,
                search_key=search_key,
                section=(
                    field.section_path[-1]
                    if field is not None
                    else str(
                        metadata.get(search_key, {}).get("section", DEFAULT_SECTION)
                    )
                ),
                value_type=_search_axis_kind(spec, config_key, values),
                values=tuple(spec.serialize_value(value) for value in values),
                locked=bool(lock_details),
                locked_value=(
                    _shared_locked_value(spec, lock_details) if lock_details else None
                ),
                locked_reason=" ".join(lock_reasons),
                locked_by_presets=locked_by_presets,
                lock_reasons=lock_reasons,
            )
        )
    return SearchSpace(identity=package.identity, preset=preset, axes=tuple(axes))


__all__ = [
    "configuration_schema",
    "preset_locks",
    "search_space_schema",
]

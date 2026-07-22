from __future__ import annotations

import inspect
from enum import Enum
from types import ModuleType, NoneType, UnionType
from typing import Any, Union, get_args, get_origin

from model_runtime.inspection.errors import InspectionError, _model_package_failure
from model_runtime.inspection.field_descriptions import config_field_description
from model_runtime.inspection.records import (
    ConfigurationField,
    ConfigurationSchema,
    SearchAxis,
    SearchSpace,
)
from model_runtime.packages import (
    ModelPackage,
    abstract_config_class_error,
    config_key_to_flag,
    config_key_to_model_param,
    iter_supported_config_keys,
    serialize_config_value,
)

DEFAULT_SECTION = "General"
PRIMITIVE_ANNOTATION_KINDS = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "string",
}


def _field_section_path(key: str, metadata: dict[str, Any]) -> tuple[str, ...]:
    raw_path = metadata.get("sectionPath")
    if isinstance(raw_path, list):
        path = tuple(item for item in raw_path if isinstance(item, str) and item)
        if path:
            return path
    raise InspectionError(
        f"Config field {key!r} is missing source heading metadata. "
        "Add a markdown heading comment above the field or preserve imported "
        "config field names."
    )


def _annotation_classes(annotation: Any) -> list[type]:
    if annotation is None:
        return []
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {UnionType, Union}:
        return [item for arg in args for item in _annotation_classes(arg)]
    if annotation is NoneType:
        return []
    if origin is type and args and isinstance(args[0], type):
        return [args[0]]
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


def _value_kind(value: Any, annotation: Any) -> str:
    annotation_classes = _annotation_classes(annotation)
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Enum):
        return "enum"
    if inspect.isclass(value):
        return "class"
    if value is None:
        primitive_kind = _annotation_primitive_kind(annotation)
        if primitive_kind is not None:
            return primitive_kind
        if any(issubclass(cls, Enum) for cls in annotation_classes):
            return "enum"
        if any(cls not in PRIMITIVE_ANNOTATION_KINDS for cls in annotation_classes):
            return "class"
        return "unknown"
    if isinstance(value, list):
        return "list"
    return "unknown"


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


def _search_space_class_choices(
    search_space_module: ModuleType,
    key: str | None,
    available_choices: set[str],
) -> list[str]:
    if key is None:
        return []
    values = getattr(search_space_module, f"SEARCH_SPACE_{key}", None)
    if not isinstance(values, list):
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        choice = _class_choice_name(value)
        if choice is None or choice not in available_choices or choice in seen:
            continue
        ordered.append(choice)
        seen.add(choice)
    return ordered


def _class_choices(
    config_module: ModuleType,
    search_space_module: ModuleType,
    annotation: Any,
    current_value: Any,
    key: str | None = None,
) -> list[str]:
    expected = [
        cls
        for cls in _annotation_classes(annotation)
        if not issubclass(cls, Enum) and cls not in PRIMITIVE_ANNOTATION_KINDS
    ]
    if inspect.isclass(current_value):
        expected.append(current_value)

    choices = []
    for candidate in vars(config_module).values():
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
    ordered = _search_space_class_choices(search_space_module, key, available)
    if ordered:
        return ordered + sorted(available - set(ordered))
    return sorted(available)


def _choices_for(
    config_module: ModuleType,
    search_space_module: ModuleType,
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
        return _class_choices(
            config_module,
            search_space_module,
            annotation,
            value,
            key,
        )
    return []


def preset_locks(
    package: ModelPackage,
    preset_name: str | None,
) -> dict[str, Any]:
    if preset_name is None:
        return {}
    try:
        preset = package.resolve_preset(preset_name)
    except ValueError as exc:
        raise InspectionError(str(exc)) from exc
    except Exception as exc:
        raise _model_package_failure(package.catalog_key, exc) from exc
    try:
        locks = package.preset_locks(preset)
    except Exception as exc:
        raise _model_package_failure(package.catalog_key, exc) from exc
    canonical: dict[str, Any] = {}
    source_fields: dict[str, str] = {}
    for field, lock in locks.items():
        model_param = config_key_to_model_param(field)
        previous = canonical.get(model_param)
        if previous is not None:
            previous_value = serialize_config_value(getattr(previous, "value", None))
            value = serialize_config_value(getattr(lock, "value", None))
            if previous_value != value:
                raise InspectionError(
                    f"Preset '{preset_name}' for model "
                    f"'{package.catalog_key}' defines conflicting locks for "
                    f"Runtime Defaults parameter '{model_param}' through "
                    f"'{source_fields[model_param]}' and '{field}'."
                )
            continue
        canonical[model_param] = lock
        source_fields[model_param] = field
    return canonical


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
    package: ModelPackage,
    preset_name: str | None,
    preset_names: tuple[str, ...] | list[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    details: dict[str, list[dict[str, Any]]] = {}
    for selected_name in _unique_presets(preset_name, preset_names):
        try:
            preset = package.resolve_preset(selected_name)
        except ValueError as exc:
            raise InspectionError(str(exc)) from exc
        except Exception as exc:
            raise _model_package_failure(package.catalog_key, exc) from exc
        locks = preset_locks(package, selected_name)
        for field, lock in locks.items():
            details.setdefault(field, []).append(
                {
                    "preset": preset.name,
                    "value": getattr(lock, "value", None),
                    "reason": getattr(lock, "reason", ""),
                }
            )
    return details


def _shared_locked_value(lock_details: list[dict[str, Any]]) -> Any:
    if not lock_details:
        return None
    values = [serialize_config_value(detail["value"]) for detail in lock_details]
    first = values[0]
    return first if all(value == first for value in values) else None


def configuration_schema(
    package: ModelPackage,
    preset: str | None = None,
) -> ConfigurationSchema:
    try:
        config_module = package.runtime_defaults
        search_space_module = package.metadata.search_space
    except Exception as exc:
        raise _model_package_failure(package.catalog_key, exc) from exc
    locks = preset_locks(package, preset)
    annotations = getattr(config_module, "__annotations__", {})
    try:
        metadata = package.configuration_field_metadata()
    except ValueError as exc:
        raise InspectionError(str(exc)) from exc
    except Exception as exc:
        raise _model_package_failure(package.catalog_key, exc) from exc
    skip_keys = {
        key
        for key in getattr(config_module, "CONFIG_SCHEMA_SKIP_KEYS", ())
        if isinstance(key, str)
    }
    supported_keys = [
        key for key in iter_supported_config_keys(config_module) if key not in skip_keys
    ]
    missing = [key for key in supported_keys if key not in metadata]
    if missing:
        raise InspectionError(
            f"Config fields for model {package.catalog_key!r} are missing source "
            f"heading metadata: {', '.join(missing)}"
        )
    supported_keys.sort(key=lambda key: tuple(metadata[key].get("sortKey", [10**9])))

    fields: list[ConfigurationField] = []
    for key in supported_keys:
        value = getattr(config_module, key, None)
        annotation = annotations.get(key)
        kind = _value_kind(value, annotation)
        section_path = _field_section_path(key, metadata.get(key, {}))
        nullable = value is None or _annotation_is_nullable(annotation)
        lock = locks.get(config_key_to_model_param(key))
        locked_value = getattr(lock, "value", None) if lock is not None else None
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
                default=serialize_config_value(value),
                nullable=nullable,
                choices=tuple(
                    serialize_config_value(choice)
                    for choice in _choices_for(
                        config_module,
                        search_space_module,
                        value,
                        annotation,
                        kind,
                        key,
                    )
                ),
                maximum=package.inspection_construction_limits.maximum_for(key),
                locked=lock is not None,
                locked_value=(
                    serialize_config_value(locked_value) if lock is not None else None
                ),
                locked_reason=(getattr(lock, "reason", "") if lock else ""),
            )
        )
    return ConfigurationSchema(identity=package.identity, fields=tuple(fields))


def _search_axis_kind(
    config_module: ModuleType,
    search_space_module: ModuleType,
    config_key: str,
    values: list[Any],
) -> str:
    annotations = getattr(config_module, "__annotations__", {})
    if hasattr(config_module, config_key):
        return _value_kind(
            getattr(config_module, config_key, None),
            annotations.get(config_key),
        )
    search_annotations = getattr(search_space_module, "__annotations__", {})
    sample = next((value for value in values if value is not None), None)
    return _value_kind(
        sample,
        annotations.get(config_key) or search_annotations.get(config_key),
    )


def search_space_schema(
    package: ModelPackage,
    preset: str | None = None,
    presets: tuple[str, ...] | list[str] | None = None,
) -> SearchSpace:
    lock_details_by_param = _preset_lock_details(package, preset, presets)
    try:
        config_module = package.runtime_defaults
        search_space_module = package.metadata.search_space
        metadata = package.configuration_field_metadata(include_search_space=True)
    except ValueError as exc:
        raise InspectionError(str(exc)) from exc
    except Exception as exc:
        raise _model_package_failure(package.catalog_key, exc) from exc
    config_fields = {
        field.key: field for field in configuration_schema(package, preset).fields
    }
    prefix = "SEARCH_SPACE_"
    search_keys = sorted(
        (
            key
            for key, value in vars(search_space_module).items()
            if key.startswith(prefix) and isinstance(value, list)
        ),
        key=lambda key: int(metadata.get(key, {}).get("line", 10**9)),
    )

    axes: list[SearchAxis] = []
    for search_key in search_keys:
        config_key = search_key[len(prefix) :]
        values = getattr(search_space_module, search_key, [])
        field = config_fields.get(config_key)
        lock_details = lock_details_by_param.get(
            config_key_to_model_param(config_key),
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
                value_type=_search_axis_kind(
                    config_module,
                    search_space_module,
                    config_key,
                    values,
                ),
                values=tuple(serialize_config_value(value) for value in values),
                locked=bool(lock_details),
                locked_value=(
                    _shared_locked_value(lock_details) if lock_details else None
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

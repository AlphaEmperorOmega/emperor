from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from enum import Enum
from pathlib import Path
from types import ModuleType, NoneType, UnionType
from typing import Any, Union, get_args, get_origin

from models.catalog import model_identity_payload_from_id
from models.config_overrides import (
    config_key_to_flag,
    config_key_to_model_param,
    iter_supported_config_keys,
)

from workbench.backend.inspector.config_classes import abstract_config_class_error
from workbench.backend.inspector.errors import InspectorError
from workbench.backend.inspector.field_descriptions import config_field_description
from workbench.backend.inspector.values import serialize_config_value

DEFAULT_SECTION = "General"
PRIMITIVE_ANNOTATION_KINDS = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "string",
}


def load_model_parts(model_name: str):
    from workbench.backend.inspector.discovery import load_model_parts as load_parts

    return load_parts(model_name)


def _assignment_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _section_title(title: str) -> str | None:
    title = title.strip()
    if not title or set(title) <= {"#", "-", "=", "_", "*"}:
        return None
    if "=" in title or "`" in title or ":" in title or title.endswith("."):
        return None
    if title.lower().startswith(("if ", "these ", "this ", "when ", "for ")):
        return None

    title = title.split("(", 1)[0].strip()
    if not title:
        return None
    lower_title = title.lower()
    simple_headings = {
        "global",
        "trainer",
        "callback",
        "model",
        "preset",
        "adaptive preset",
    }
    if (
        title.isupper()
        or title.istitle()
        or "options" in lower_title
        or lower_title in simple_headings
    ):
        return title.title()
    return None


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _markdown_heading(line: str) -> tuple[int, str] | None:
    match = HEADING_RE.match(line.strip())
    if match is None:
        return None
    title = _section_title(match.group(2))
    if title is None:
        return None
    return len(match.group(1)), title


def _absolute_import_module_name(
    node: ast.ImportFrom,
    current_module_name: str,
) -> str | None:
    if node.level == 0:
        return node.module
    package_parts = current_module_name.split(".")[:-node.level]
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(package_parts) if package_parts else None


def _star_import_module_names(
    tree: ast.Module,
    current_module_name: str,
    *,
    include_search_space: bool,
) -> list[tuple[int, str]]:
    module_names = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if not any(alias.name == "*" for alias in node.names):
            continue
        module_name = _absolute_import_module_name(node, current_module_name)
        if not module_name:
            continue
        if include_search_space:
            if module_name.endswith(".search_space"):
                module_names.append((node.lineno, module_name))
            continue
        if module_name.endswith(
            (".dataset_options", ".monitor_options", ".search_space")
        ):
            continue
        module_names.append((node.lineno, module_name))
    return module_names


def _explicit_uppercase_imports(
    tree: ast.Module,
    current_module_name: str,
) -> list[tuple[int, str, list[str]]]:
    imports = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        imported_names = []
        for alias in node.names:
            if alias.name == "*" or not alias.name.isupper():
                continue
            if alias.asname is not None:
                raise InspectorError(
                    "Config fields imported with uppercase names cannot use "
                    f"`as` aliases: {current_module_name}:{node.lineno} imports "
                    f"{alias.name} as {alias.asname}."
                )
            imported_names.append(alias.name)
        if not imported_names:
            continue
        module_name = _absolute_import_module_name(node, current_module_name)
        if module_name:
            imports.append((node.lineno, module_name, imported_names))
    return imports


def _config_module_alias_imports(
    tree: ast.Module,
    current_module_name: str,
) -> list[tuple[int, str]]:
    module_names = []
    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.asname is None:
                continue
            if not alias.name.endswith(".config"):
                continue
            module_names.append((node.lineno, alias.name))
    return module_names


def _module_path(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin in {"built-in", "frozen"}:
        return None
    return Path(origin)


def _source_metadata(
    config_module: ModuleType,
    *,
    include_search_space: bool = False,
) -> dict[str, dict[str, Any]]:
    return _source_metadata_for_module(
        config_module.__name__,
        include_search_space=include_search_space,
        visited=set(),
    )


def _source_metadata_for_module(
    module_name: str,
    *,
    include_search_space: bool,
    visited: set[str],
) -> dict[str, dict[str, int | str | list[str]]]:
    if module_name in visited:
        return {}
    visited.add(module_name)

    config_path = _module_path(module_name)
    if config_path is None:
        return {}

    try:
        source = config_path.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}

    metadata: dict[str, dict[str, Any]] = {}

    def import_metadata(
        line_number: int,
        source_metadata: dict[str, dict[str, Any]],
        imported_names: set[str] | None = None,
        *,
        overwrite: bool,
    ) -> None:
        for key, entry in source_metadata.items():
            if imported_names is not None and key not in imported_names:
                continue
            if not overwrite and key in metadata:
                continue
            metadata[key] = {
                **entry,
                "sortKey": [line_number, *entry.get("sortKey", [entry.get("line", 0)])],
            }

    for line_number, import_module_name in _star_import_module_names(
        tree,
        module_name,
        include_search_space=include_search_space,
    ):
        import_metadata(
            line_number,
            _source_metadata_for_module(
                import_module_name,
                include_search_space=include_search_space,
                visited=set(visited),
            ),
            overwrite=True,
        )

    for line_number, import_module_name, imported_names in _explicit_uppercase_imports(
        tree,
        module_name,
    ):
        import_metadata(
            line_number,
            _source_metadata_for_module(
                import_module_name,
                include_search_space=include_search_space,
                visited=set(visited),
            ),
            set(imported_names),
            overwrite=True,
        )

    for line_number, import_module_name in _config_module_alias_imports(
        tree,
        module_name,
    ):
        import_metadata(
            line_number,
            _source_metadata_for_module(
                import_module_name,
                include_search_space=include_search_space,
                visited=set(visited),
            ),
            overwrite=False,
        )

    assignments_by_line: dict[int, list[str]] = {}
    for node in tree.body:
        key = _assignment_key(node)
        if key is None or not key.isupper():
            continue
        if key.startswith("SEARCH_SPACE_") and not include_search_space:
            continue
        assignments_by_line.setdefault(node.lineno, []).append(key)

    current_path: list[str] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        heading = _markdown_heading(stripped)
        if heading is not None:
            level, title = heading
            current_path = [*current_path[: level - 1], title]
        for key in assignments_by_line.get(line_number, []):
            if not current_path:
                continue
            section_path = list(current_path)
            metadata[key] = {
                "line": line_number,
                "sortKey": [line_number],
                "section": section_path[-1],
                "sectionPath": section_path,
            }
    return metadata


def _field_section_path(key: str, field_metadata: dict[str, Any]) -> list[str]:
    raw_section_path = field_metadata.get("sectionPath")
    if isinstance(raw_section_path, list):
        section_path = [
            section
            for section in raw_section_path
            if isinstance(section, str) and section
        ]
        if section_path:
            return section_path
    raise InspectorError(
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
    if origin in {UnionType, Union}:
        return any(arg is NoneType for arg in get_args(annotation))
    return False


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
        if any(
            cls not in PRIMITIVE_ANNOTATION_KINDS
            for cls in annotation_classes
        ):
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
    if enum_type is None:
        return []
    return list(enum_type.__members__)


def _class_choice_name(value: Any) -> str | None:
    if value is None:
        return None
    if inspect.isclass(value):
        return value.__name__
    return None


def _search_space_class_choices(
    search_space_module: ModuleType,
    key: str | None,
    available_choices: set[str],
) -> list[str]:
    if key is None:
        return []
    search_key = f"SEARCH_SPACE_{key}"
    search_values = getattr(search_space_module, search_key, None)
    if not isinstance(search_values, list):
        return []

    ordered_choices = []
    seen = set()
    for search_value in search_values:
        choice = _class_choice_name(search_value)
        if choice is None or choice not in available_choices or choice in seen:
            continue
        ordered_choices.append(choice)
        seen.add(choice)
    return ordered_choices


def _class_choices(
    config_module: ModuleType,
    search_space_module: ModuleType,
    annotation: Any,
    current_value: Any,
    key: str | None = None,
) -> list[str]:
    expected_classes = [
        cls
        for cls in _annotation_classes(annotation)
        if not issubclass(cls, Enum) and cls not in PRIMITIVE_ANNOTATION_KINDS
    ]
    if inspect.isclass(current_value):
        expected_classes.append(current_value)

    choices = []
    for candidate in vars(config_module).values():
        if not inspect.isclass(candidate):
            continue
        if abstract_config_class_error(candidate) is not None:
            continue
        if not expected_classes:
            choices.append(candidate.__name__)
            continue
        if any(
            candidate is expected or issubclass(candidate, expected)
            for expected in expected_classes
        ):
            choices.append(candidate.__name__)
    available_choices = set(choices)
    ordered_choices = _search_space_class_choices(
        search_space_module,
        key,
        available_choices,
    )
    if ordered_choices:
        ordered_choice_set = set(ordered_choices)
        return ordered_choices + sorted(available_choices - ordered_choice_set)
    return sorted(available_choices)


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


def preset_locks(model_name: str, preset_name: str | None) -> dict[str, Any]:
    parts = load_model_parts(model_name)
    if preset_name is None:
        return {}
    try:
        preset = parts.experiment_preset_enum.get_member(preset_name)
    except Exception as exc:
        raise InspectorError(
            f"Unknown preset '{preset_name}' for model '{model_name}'."
        ) from exc
    locked_fields = getattr(parts.presets, "locked_fields", None)
    if not callable(locked_fields):
        return {}
    return locked_fields(preset)


def _unique_presets(
    preset_name: str | None,
    preset_names: list[str] | None,
) -> list[str]:
    raw_names = preset_names if preset_names else ([preset_name] if preset_name else [])
    names = []
    seen = set()
    for raw_name in raw_names:
        name = raw_name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _preset_lock_details(
    model_name: str,
    preset_name: str | None,
    preset_names: list[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    parts = load_model_parts(model_name)
    selected_presets = _unique_presets(preset_name, preset_names)
    if not selected_presets:
        return {}
    locked_fields = getattr(parts.presets, "locked_fields", None)
    if not callable(locked_fields):
        return {}

    details: dict[str, list[dict[str, Any]]] = {}
    for selected_preset in selected_presets:
        try:
            preset = parts.experiment_preset_enum.get_member(selected_preset)
        except Exception as exc:
            raise InspectorError(
                f"Unknown preset '{selected_preset}' for model '{model_name}'."
            ) from exc
        locks = locked_fields(preset)
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
    first_value = values[0]
    if all(value == first_value for value in values):
        return first_value
    return None


def config_schema(model_name: str, preset_name: str | None = None) -> dict[str, Any]:
    parts = load_model_parts(model_name)
    locks = preset_locks(model_name, preset_name)
    annotations = getattr(parts.config_module, "__annotations__", {})
    metadata = _source_metadata(parts.config_module)
    supported_keys = iter_supported_config_keys(parts.config_module)
    missing_metadata_keys = [key for key in supported_keys if key not in metadata]
    if missing_metadata_keys:
        raise InspectorError(
            f"Config fields for model {model_name!r} are missing source heading "
            f"metadata: {', '.join(missing_metadata_keys)}"
        )
    supported_keys = sorted(
        supported_keys,
        key=lambda key: tuple(metadata[key].get("sortKey", [10**9])),
    )
    fields = []
    for key in supported_keys:
        value = getattr(parts.config_module, key, None)
        annotation = annotations.get(key)
        kind = _value_kind(value, annotation)
        field_metadata = metadata.get(key, {})
        section_path = _field_section_path(key, field_metadata)
        section = section_path[-1]
        nullable = value is None or _annotation_is_nullable(annotation)
        model_param = config_key_to_model_param(key)
        lock = locks.get(model_param)
        locked_value = getattr(lock, "value", None) if lock is not None else None
        locked_reason = getattr(lock, "reason", "") if lock is not None else ""
        fields.append(
            {
                "key": key,
                "configKey": key,
                "flag": config_key_to_flag(key),
                "label": key.lower().replace("_", " "),
                "section": section,
                "sectionPath": section_path,
                "description": config_field_description(
                    key,
                    section=section,
                    kind=kind,
                    nullable=nullable,
                    default=value,
                ),
                "type": kind,
                "default": serialize_config_value(value),
                "nullable": nullable,
                "choices": _choices_for(
                    parts.config_module,
                    getattr(parts, "search_space_module", parts.config_module),
                    value,
                    annotation,
                    kind,
                    key,
                ),
                "locked": lock is not None,
                "lockedValue": serialize_config_value(locked_value)
                if lock is not None
                else None,
                "lockedReason": locked_reason,
            }
        )
    return {**model_identity_payload_from_id(model_name), "fields": fields}


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
    model_name: str,
    preset_name: str | None = None,
    preset_names: list[str] | None = None,
) -> dict[str, Any]:
    parts = load_model_parts(model_name)
    lock_details_by_param = _preset_lock_details(
        model_name,
        preset_name,
        preset_names,
    )
    search_space_module = getattr(parts, "search_space_module", parts.config_module)
    metadata = _source_metadata(search_space_module, include_search_space=True)
    config_fields = {
        field["configKey"]: field
        for field in config_schema(model_name, preset_name)["fields"]
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

    axes = []
    for search_key in search_keys:
        config_key = search_key[len(prefix) :]
        values = getattr(search_space_module, search_key, [])
        field = config_fields.get(config_key)
        model_param = config_key_to_model_param(config_key)
        lock_details = lock_details_by_param.get(model_param, [])
        locked_value = _shared_locked_value(lock_details)
        lock_reasons = [
            str(detail["reason"])
            for detail in lock_details
            if str(detail["reason"])
        ]
        locked_reason = " ".join(lock_reasons)
        locked_by_presets = [
            str(detail["preset"])
            for detail in lock_details
            if str(detail["preset"])
        ]
        axes.append(
            {
                "key": config_key,
                "configKey": config_key,
                "searchKey": search_key,
                "label": (
                    field["label"]
                    if field is not None
                    else config_key.lower().replace("_", " ")
                ),
                "section": (
                    field["section"]
                    if field is not None
                    else metadata.get(search_key, {}).get("section", DEFAULT_SECTION)
                ),
                "type": _search_axis_kind(
                    parts.config_module,
                    search_space_module,
                    config_key,
                    values,
                ),
                "values": [serialize_config_value(value) for value in values],
                "locked": len(lock_details) > 0,
                "lockedValue": locked_value if lock_details else None,
                "lockedReason": locked_reason,
                "lockedByPresets": locked_by_presets,
                "lockReasons": lock_reasons,
            }
        )

    return {
        **model_identity_payload_from_id(model_name),
        "preset": preset_name,
        "axes": axes,
    }

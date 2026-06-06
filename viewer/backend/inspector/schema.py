from __future__ import annotations

import inspect
import ast
from enum import Enum
from pathlib import Path
from types import ModuleType, NoneType, UnionType
from typing import Any, Union, get_args, get_origin

from models.config_overrides import (
    config_key_to_flag,
    config_key_to_model_param,
    config_key_to_param,
    iter_supported_config_keys,
    normalize_key,
)

from viewer.backend.inspector.config_classes import abstract_config_class_error
from viewer.backend.inspector.discovery import load_model_parts
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.values import serialize_config_value

DEFAULT_SECTION = "General"


def _assignment_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _section_title(comment: str) -> str | None:
    title = comment.strip()
    if not title or set(title) <= {"#", "-", "=", "_", "*"}:
        return None
    if "=" in title or "`" in title or title.endswith("."):
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


def _source_metadata(
    config_module: ModuleType,
    *,
    include_search_space: bool = False,
) -> dict[str, dict[str, int | str]]:
    config_file = getattr(config_module, "__file__", None)
    if not config_file:
        return {}

    config_path = Path(config_file)
    try:
        source = config_path.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}

    assignments_by_line: dict[int, list[str]] = {}
    for node in tree.body:
        key = _assignment_key(node)
        if key is None or not key.isupper():
            continue
        if key.startswith("SEARCH_SPACE_") and not include_search_space:
            continue
        assignments_by_line.setdefault(node.lineno, []).append(key)

    metadata: dict[str, dict[str, int | str]] = {}
    current_section = DEFAULT_SECTION
    for line_number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            title = _section_title(stripped.lstrip("#"))
            if title:
                current_section = title
        for key in assignments_by_line.get(line_number, []):
            metadata[key] = {"line": line_number, "section": current_section}
    return metadata


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
        if any(issubclass(cls, Enum) for cls in annotation_classes):
            return "enum"
        if annotation_classes:
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


def _class_choices(config_module: ModuleType, annotation: Any, current_value: Any) -> list[str]:
    expected_classes = [
        cls for cls in _annotation_classes(annotation) if not issubclass(cls, Enum)
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
    return sorted(set(choices))


def _choices_for(
    config_module: ModuleType,
    value: Any,
    annotation: Any,
    kind: str,
) -> list[Any]:
    if kind == "bool":
        return [True, False]
    if kind == "enum":
        return _enum_choices(value, annotation)
    if kind == "class":
        return _class_choices(config_module, annotation, value)
    return []


def preset_locks(model_name: str, preset_name: str | None) -> dict[str, Any]:
    parts = load_model_parts(model_name)
    if preset_name is None:
        return {}
    try:
        option = parts.experiment_options.get_option(preset_name)
    except Exception as exc:
        raise InspectorError(f"Unknown preset '{preset_name}' for model '{model_name}'.") from exc
    locked_fields = getattr(parts.presets, "locked_fields", None)
    if not callable(locked_fields):
        return {}
    return locked_fields(option)


def config_schema(model_name: str, preset_name: str | None = None) -> dict[str, Any]:
    parts = load_model_parts(model_name)
    locks = preset_locks(model_name, preset_name)
    annotations = getattr(parts.config_module, "__annotations__", {})
    metadata = _source_metadata(parts.config_module)
    supported_keys = sorted(
        iter_supported_config_keys(parts.config_module),
        key=lambda key: int(metadata.get(key, {}).get("line", 10**9)),
    )
    fields = []
    for key in supported_keys:
        value = getattr(parts.config_module, key, None)
        annotation = annotations.get(key)
        kind = _value_kind(value, annotation)
        field_metadata = metadata.get(key, {})
        model_param = config_key_to_model_param(key)
        lock = locks.get(model_param)
        locked_value = getattr(lock, "value", None) if lock is not None else None
        locked_reason = getattr(lock, "reason", "") if lock is not None else ""
        fields.append(
            {
                "key": config_key_to_param(key),
                "configKey": key,
                "flag": config_key_to_flag(key),
                "label": key.lower().replace("_", " "),
                "section": field_metadata.get("section", DEFAULT_SECTION),
                "type": kind,
                "default": serialize_config_value(value),
                "nullable": value is None or _annotation_is_nullable(annotation),
                "choices": _choices_for(
                    parts.config_module,
                    value,
                    annotation,
                    kind,
                ),
                "locked": lock is not None,
                "lockedValue": serialize_config_value(locked_value) if lock is not None else None,
                "lockedReason": locked_reason,
            }
        )
    return {"model": model_name, "fields": fields}


def _search_axis_kind(config_module: ModuleType, config_key: str, values: list[Any]) -> str:
    annotations = getattr(config_module, "__annotations__", {})
    if hasattr(config_module, config_key):
        return _value_kind(
            getattr(config_module, config_key, None),
            annotations.get(config_key),
        )
    sample = next((value for value in values if value is not None), None)
    return _value_kind(sample, annotations.get(config_key))


def search_space_schema(
    model_name: str,
    preset_name: str | None = None,
) -> dict[str, Any]:
    parts = load_model_parts(model_name)
    locks = preset_locks(model_name, preset_name)
    metadata = _source_metadata(parts.config_module, include_search_space=True)
    config_fields = {
        field["configKey"]: field
        for field in config_schema(model_name, preset_name)["fields"]
    }
    prefix = "SEARCH_SPACE_"
    search_keys = sorted(
        (
            key
            for key, value in vars(parts.config_module).items()
            if key.startswith(prefix) and isinstance(value, list)
        ),
        key=lambda key: int(metadata.get(key, {}).get("line", 10**9)),
    )

    axes = []
    for search_key in search_keys:
        config_key = search_key[len(prefix) :]
        values = getattr(parts.config_module, search_key, [])
        field = config_fields.get(config_key)
        model_param = config_key_to_model_param(config_key)
        lock = locks.get(model_param)
        locked_value = getattr(lock, "value", None) if lock is not None else None
        locked_reason = getattr(lock, "reason", "") if lock is not None else ""
        axes.append(
            {
                "key": normalize_key(config_key),
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
                "type": _search_axis_kind(parts.config_module, config_key, values),
                "values": [serialize_config_value(value) for value in values],
                "locked": lock is not None,
                "lockedValue": serialize_config_value(locked_value) if lock is not None else None,
                "lockedReason": locked_reason,
            }
        )

    return {"model": model_name, "preset": preset_name, "axes": axes}

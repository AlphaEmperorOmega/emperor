from __future__ import annotations

import inspect
import ast
from enum import Enum
from pathlib import Path
from types import ModuleType, NoneType, UnionType
from typing import Any, Union, get_args, get_origin

from models.config_overrides import (
    config_key_to_flag,
    config_key_to_param,
    iter_supported_config_keys,
)

from viewer.backend.inspector.config_classes import abstract_config_class_error
from viewer.backend.inspector.discovery import load_model_parts

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


def _source_metadata(config_module: ModuleType) -> dict[str, dict[str, int | str]]:
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
        if key is None or not key.isupper() or key.startswith("SEARCH_SPACE_"):
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


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if inspect.isclass(value):
        return value.__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
    key: str,
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
    search_values = getattr(config_module, f"SEARCH_SPACE_{key}", None)
    if isinstance(search_values, list):
        return [_serialize_value(item) for item in search_values]
    return []


def _search_choices(config_module: ModuleType, key: str) -> list[Any]:
    search_values = getattr(config_module, f"SEARCH_SPACE_{key}", None)
    if not isinstance(search_values, list):
        return []
    return [_serialize_value(item) for item in search_values]


def config_schema(model_name: str) -> dict[str, Any]:
    parts = load_model_parts(model_name)
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
        fields.append(
            {
                "key": config_key_to_param(key),
                "configKey": key,
                "flag": config_key_to_flag(key),
                "label": key.lower().replace("_", " "),
                "section": field_metadata.get("section", DEFAULT_SECTION),
                "type": kind,
                "default": _serialize_value(value),
                "nullable": value is None or _annotation_is_nullable(annotation),
                "choices": _choices_for(
                    parts.config_module,
                    key,
                    value,
                    annotation,
                    kind,
                ),
                "searchChoices": _search_choices(parts.config_module, key),
            }
        )
    return {"model": model_name, "fields": fields}

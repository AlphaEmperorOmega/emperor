from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_type_hints

from models.catalog import discover_model_packages


def _allows_none(annotation: object) -> bool:
    return annotation in (Any, object) or type(None) in get_args(annotation)


def _none_annotation_mismatches(root: object) -> list[str]:
    mismatches: list[str] = []
    pending = [("runtime", root)]
    visited: set[int] = set()
    while pending:
        path, value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        if not is_dataclass(value) or isinstance(value, type):
            continue
        annotations = get_type_hints(type(value))
        for dataclass_field in fields(value):
            field_value = getattr(value, dataclass_field.name)
            field_path = f"{path}.{dataclass_field.name}"
            annotation = annotations[dataclass_field.name]
            if field_value is None and not _allows_none(annotation):
                mismatches.append(f"{field_path}: {annotation}")
            elif is_dataclass(field_value) and not isinstance(field_value, type):
                pending.append((field_path, field_value))
    return mismatches


def test_runtime_option_annotations_accept_every_default_none_value() -> None:
    mismatches: list[str] = []
    for package in discover_model_packages():
        for mismatch in _none_annotation_mismatches(package.bind_runtime_defaults()):
            mismatches.append(f"{package.catalog_key}: {mismatch}")

    assert not mismatches, "Runtime annotation mismatches:\n" + "\n".join(mismatches)

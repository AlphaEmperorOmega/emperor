from __future__ import annotations

import ast
import importlib.util
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

_RuntimeDefaultsField = tuple[str, int, tuple[int, ...]]
_RuntimeDefaultsFieldInput = tuple[str, int, Sequence[int]]


def _snapshot_section_path(path: object) -> tuple[str, ...]:
    if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
        raise ValueError("Runtime Defaults metadata section path must be a sequence.")
    snapshot = tuple(cast(Sequence[object], path))
    if any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in snapshot
    ):
        raise ValueError(
            "Runtime Defaults metadata section path items must be non-empty "
            "trimmed strings."
        )
    return cast(tuple[str, ...], snapshot)


def _snapshot_metadata_field(field: object) -> _RuntimeDefaultsField:
    if isinstance(field, (str, bytes)) or not isinstance(field, Sequence):
        raise ValueError(
            "Runtime Defaults metadata fields require key, line, and sort key."
        )
    values = tuple(cast(Sequence[object], field))
    if len(values) != 3:
        raise ValueError(
            "Runtime Defaults metadata fields require key, line, and sort key."
        )
    key, line, raw_sort_key = values
    if not isinstance(key, str) or not key or key != key.strip():
        raise ValueError(
            "Runtime Defaults metadata field keys must be non-empty trimmed strings."
        )
    if type(line) is not int or line < 1:
        raise ValueError(
            "Runtime Defaults metadata field lines must be positive integers."
        )
    if isinstance(raw_sort_key, (str, bytes)) or not isinstance(raw_sort_key, Sequence):
        raise ValueError(
            "Runtime Defaults metadata sort keys must be integer sequences."
        )
    sort_key = tuple(cast(Sequence[object], raw_sort_key))
    if not sort_key or any(type(item) is not int or item < 0 for item in sort_key):
        raise ValueError(
            "Runtime Defaults metadata sort keys must contain non-negative integers."
        )
    return key, line, cast(tuple[int, ...], sort_key)


def _snapshot_metadata_fields(fields: object) -> tuple[_RuntimeDefaultsField, ...]:
    if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
        raise ValueError("Runtime Defaults metadata fields must be a sequence.")
    return tuple(
        _snapshot_metadata_field(field) for field in cast(Sequence[object], fields)
    )


@dataclass(frozen=True, slots=True, init=False)
class RuntimeDefaultsSection:
    """Immutable package-owned declaration of ordered Inspection metadata."""

    path: tuple[str, ...]
    fields: tuple[_RuntimeDefaultsField, ...]

    def __init__(
        self,
        path: Sequence[str],
        fields: Sequence[_RuntimeDefaultsFieldInput],
    ) -> None:
        snapshot = _snapshot_metadata_fields(fields)
        if not snapshot:
            raise ValueError("Runtime Defaults metadata sections require fields.")
        object.__setattr__(self, "path", _snapshot_section_path(path))
        object.__setattr__(self, "fields", snapshot)


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
    package_parts = current_module_name.split(".")[: -node.level]
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(package_parts) if package_parts else None


def _star_import_module_names(
    tree: ast.Module,
    current_module_name: str,
    *,
    include_search_space: bool,
) -> list[tuple[int, str]]:
    module_names: list[tuple[int, str]] = []
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
    imports: list[tuple[int, str, list[str]]] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        imported_names: list[str] = []
        for alias in node.names:
            if alias.name == "*" or not alias.name.isupper():
                continue
            if alias.asname is not None:
                raise ValueError(
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
    module_names: list[tuple[int, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.asname is None or not alias.name.endswith(".config"):
                continue
            module_names.append((node.lineno, alias.name))
    return module_names


def _module_path(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin in {"built-in", "frozen"}:
        return None
    return Path(origin)


def configuration_field_metadata(
    config_module: ModuleType,
    *,
    include_search_space: bool = False,
) -> dict[str, dict[str, Any]]:
    metadata = _configuration_field_metadata_for_module(
        config_module.__name__,
        include_search_space=include_search_space,
        visited=set(),
    )
    aliases = getattr(config_module, "_CONFIG_FIELD_METADATA_ALIASES", {})
    if not isinstance(aliases, dict):
        return metadata
    alias_mapping = cast(dict[str, object], aliases)

    def resolve_alias(key: str, active: set[str]) -> dict[str, Any] | None:
        entry = metadata.get(key)
        if entry is not None:
            return entry
        if key in active:
            return None
        source = alias_mapping.get(key)
        if not isinstance(source, str):
            return None
        return resolve_alias(source, {*active, key})

    for index, target in enumerate(alias_mapping):
        if target in metadata:
            continue
        source_entry = resolve_alias(target, set())
        if source_entry is None:
            continue
        metadata[target] = {
            **source_entry,
            "sortKey": [*source_entry.get("sortKey", [10**9]), index],
        }
    return metadata


class _ModuleMetadataParser:
    def __init__(
        self,
        module_name: str,
        include_search_space: bool,
        visited: set[str],
        source: str,
        tree: ast.Module,
    ) -> None:
        self._module_name = module_name
        self._include_search_space = include_search_space
        self._visited = visited
        self._source = source
        self._tree = tree
        self._metadata: dict[str, dict[str, Any]] = {}

    def parse(self) -> dict[str, dict[str, Any]]:
        self._import_star_metadata()
        self._import_explicit_metadata()
        self._import_config_alias_metadata()
        self._add_local_metadata(self._assignments_by_line())
        return self._metadata

    def _metadata_for(self, imported_module: str) -> dict[str, dict[str, Any]]:
        return _configuration_field_metadata_for_module(
            imported_module,
            include_search_space=self._include_search_space,
            visited=set(self._visited),
        )

    def _merge_imported_metadata(
        self,
        line_number: int,
        source_metadata: dict[str, dict[str, Any]],
        imported_names: set[str] | None = None,
        *,
        overwrite: bool,
    ) -> None:
        for key, entry in source_metadata.items():
            if imported_names is not None and key not in imported_names:
                continue
            if not overwrite and key in self._metadata:
                continue
            self._metadata[key] = {
                **entry,
                "sortKey": [line_number, *entry.get("sortKey", [entry.get("line", 0)])],
            }

    def _import_star_metadata(self) -> None:
        imports = _star_import_module_names(
            self._tree,
            self._module_name,
            include_search_space=self._include_search_space,
        )
        for line_number, imported_module in imports:
            self._merge_imported_metadata(
                line_number,
                self._metadata_for(imported_module),
                overwrite=True,
            )

    def _import_explicit_metadata(self) -> None:
        imports = _explicit_uppercase_imports(self._tree, self._module_name)
        for line_number, imported_module, imported_names in imports:
            self._merge_imported_metadata(
                line_number,
                self._metadata_for(imported_module),
                set(imported_names),
                overwrite=True,
            )

    def _import_config_alias_metadata(self) -> None:
        imports = _config_module_alias_imports(self._tree, self._module_name)
        for line_number, imported_module in imports:
            self._merge_imported_metadata(
                line_number,
                self._metadata_for(imported_module),
                overwrite=False,
            )

    def _assignments_by_line(self) -> dict[int, list[str]]:
        assignments: dict[int, list[str]] = {}
        for node in self._tree.body:
            key = _assignment_key(node)
            if key is None or not key.isupper():
                continue
            if key.startswith("SEARCH_SPACE_") and not self._include_search_space:
                continue
            assignments.setdefault(node.lineno, []).append(key)
        return assignments

    def _add_local_metadata(self, assignments_by_line: dict[int, list[str]]) -> None:
        current_path: list[str] = []
        for line_number, line in enumerate(self._source.splitlines(), start=1):
            heading = _markdown_heading(line.strip())
            if heading is not None:
                level, title = heading
                current_path = [*current_path[: level - 1], title]
            for key in assignments_by_line.get(line_number, []):
                if not current_path:
                    if self._include_search_space:
                        self._metadata[key] = {
                            "line": line_number,
                            "sortKey": [line_number],
                        }
                    continue
                section_path = list(current_path)
                self._metadata[key] = {
                    "line": line_number,
                    "sortKey": [line_number],
                    "section": section_path[-1],
                    "sectionPath": section_path,
                }


def _configuration_field_metadata_for_module(
    module_name: str,
    *,
    include_search_space: bool,
    visited: set[str],
) -> dict[str, dict[str, Any]]:
    if module_name in visited:
        return {}
    visited.add(module_name)

    config_path = _module_path(module_name)
    if config_path is None:
        return {}
    try:
        source = config_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}
    return _ModuleMetadataParser(
        module_name,
        include_search_space,
        visited,
        source,
        tree,
    ).parse()


__all__ = ["RuntimeDefaultsSection", "configuration_field_metadata"]

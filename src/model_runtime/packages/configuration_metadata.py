from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any


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
    module_names = []
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
    return _configuration_field_metadata_for_module(
        config_module.__name__,
        include_search_space=include_search_space,
        visited=set(),
    )


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

    for line_number, imported_module in _star_import_module_names(
        tree,
        module_name,
        include_search_space=include_search_space,
    ):
        import_metadata(
            line_number,
            _configuration_field_metadata_for_module(
                imported_module,
                include_search_space=include_search_space,
                visited=set(visited),
            ),
            overwrite=True,
        )

    for line_number, imported_module, imported_names in _explicit_uppercase_imports(
        tree,
        module_name,
    ):
        import_metadata(
            line_number,
            _configuration_field_metadata_for_module(
                imported_module,
                include_search_space=include_search_space,
                visited=set(visited),
            ),
            set(imported_names),
            overwrite=True,
        )

    for line_number, imported_module in _config_module_alias_imports(
        tree,
        module_name,
    ):
        import_metadata(
            line_number,
            _configuration_field_metadata_for_module(
                imported_module,
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
        heading = _markdown_heading(line.strip())
        if heading is not None:
            level, title = heading
            current_path = [*current_path[: level - 1], title]
        for key in assignments_by_line.get(line_number, []):
            if not current_path:
                if include_search_space:
                    metadata[key] = {
                        "line": line_number,
                        "sortKey": [line_number],
                    }
                continue
            section_path = list(current_path)
            metadata[key] = {
                "line": line_number,
                "sortKey": [line_number],
                "section": section_path[-1],
                "sectionPath": section_path,
            }
    return metadata


__all__ = ["configuration_field_metadata"]

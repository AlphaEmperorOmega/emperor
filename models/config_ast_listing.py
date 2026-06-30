import ast
from pathlib import Path

from models.dataset_naming import dataset_class_name_to_cli_name


def _source_value(source: str, node: ast.AST) -> str:
    value = ast.get_source_segment(source, node)
    if value is None:
        return "..."
    return " ".join(value.split())


def _iter_assignment_nodes(tree: ast.Module):
    for node in tree.body:
        key = None
        value_node = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            key = node.target.id
            value_node = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                key = target.id
                value_node = node.value
        if key is not None and value_node is not None:
            yield key, value_node


def _config_assignment_rows_from_source(
    source: str,
    base_skip_keys: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    tree = ast.parse(source)
    config_options: dict[str, str] = {}
    search_options: dict[str, str] = {}
    module_skip_keys = set()

    for key, value_node in _iter_assignment_nodes(tree):
        if key != "CONFIG_OVERRIDE_SKIP_KEYS" or value_node is None:
            continue
        try:
            raw_skip_keys = ast.literal_eval(value_node)
        except (SyntaxError, ValueError):
            continue
        if isinstance(raw_skip_keys, (list, tuple, set)):
            module_skip_keys.update(
                item for item in raw_skip_keys if isinstance(item, str)
            )

    skip_keys = base_skip_keys | module_skip_keys

    for key, value_node in _iter_assignment_nodes(tree):
        if not key.isupper() or key in skip_keys:
            continue

        default = _source_value(source, value_node)
        if key.startswith("SEARCH_SPACE_"):
            search_options[key[len("SEARCH_SPACE_") :]] = default
        else:
            config_options[key] = default

    return config_options, search_options


def _imports_shared_trainer_config(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "models.trainer_config":
            continue
        if any(alias.name == "*" for alias in node.names):
            return True
    return False


def iter_config_assignments(
    config_path: Path,
    *,
    shared_config_path: Path,
    base_skip_keys: set[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    source = config_path.read_text()
    tree = ast.parse(source)
    config_options: dict[str, str] = {}
    search_options: dict[str, str] = {}

    if _imports_shared_trainer_config(tree):
        shared_source = shared_config_path.read_text()
        shared_config_options, shared_search_options = (
            _config_assignment_rows_from_source(shared_source, base_skip_keys)
        )
        config_options.update(shared_config_options)
        search_options.update(shared_search_options)

    local_config_options, local_search_options = _config_assignment_rows_from_source(
        source,
        base_skip_keys,
    )
    config_options.update(local_config_options)
    search_options.update(local_search_options)

    return list(config_options.items()), list(search_options.items())


def _string_literal(node: ast.AST) -> str:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return ""
    return value if isinstance(value, str) else ""


def _preset_key_from_node(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Attribute):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != "ExperimentPreset":
        return None
    return node.attr


def _preset_definition_description(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    function_name = node.func.id if isinstance(node.func, ast.Name) else ""
    if function_name != "PresetDefinition":
        return ""
    for keyword in node.keywords:
        if keyword.arg == "description":
            return _string_literal(keyword.value)
    return ""


def _preset_definition_descriptions(tree: ast.Module) -> dict[str, str]:
    descriptions = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "_PRESET_DEFINITIONS"
            for target in targets
        ):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        for key_node, value_node in zip(value.keys, value.values):
            if key_node is None:
                continue
            key = _preset_key_from_node(key_node)
            if key is None:
                continue
            description = _preset_definition_description(value_node)
            if description:
                descriptions[key] = description
    return descriptions


def preset_option_rows_from_source(source: str) -> list[tuple[str, str | None]] | None:
    tree = ast.parse(source)
    preset_descriptions = _preset_definition_descriptions(tree)

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "ExperimentPreset":
            continue

        rows = []
        for item in node.body:
            key = None
            value_node = None
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                target = item.targets[0]
                if isinstance(target, ast.Name):
                    key = target.id
                    value_node = item.value
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                key = item.target.id
                value_node = item.value

            if key is None or value_node is None:
                continue
            value = ast.literal_eval(value_node)
            description = preset_descriptions.get(key)
            if description is None and isinstance(value, str):
                description = value
            rows.append((key, description))
        return rows

    return None


def _dataset_option_name(item: ast.AST) -> str | None:
    if isinstance(item, ast.Name):
        return item.id
    if isinstance(item, ast.Attribute):
        return item.attr
    return None


def dataset_option_names_from_source(source: str) -> list[str] | None:
    tree = ast.parse(source)

    for key, value_node in _iter_assignment_nodes(tree):
        if key != "DATASET_OPTIONS" or not isinstance(value_node, (ast.List, ast.Tuple)):
            continue

        dataset_names = [
            name
            for item in value_node.elts
            if (name := _dataset_option_name(item)) is not None
        ]
        if not dataset_names:
            return []
        return [
            dataset_class_name_to_cli_name(dataset_name)
            for dataset_name in dataset_names
        ]

    return None


def _monitor_option_name(item: ast.AST) -> str | None:
    if not isinstance(item, ast.Call):
        return None
    for keyword in item.keywords:
        if keyword.arg != "name":
            continue
        try:
            value = ast.literal_eval(keyword.value)
        except (SyntaxError, ValueError):
            return None
        return value if isinstance(value, str) else None
    return None


def monitor_option_names_from_source(source: str) -> list[str] | None:
    tree = ast.parse(source)

    for key, value_node in _iter_assignment_nodes(tree):
        if key != "MONITOR_OPTIONS":
            continue
        if not isinstance(value_node, (ast.List, ast.Tuple)):
            return []

        return [
            name
            for item in value_node.elts
            if (name := _monitor_option_name(item)) is not None
        ]

    return None

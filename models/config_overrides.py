import argparse
import ast
import inspect
from enum import Enum
from pathlib import Path
from types import ModuleType, UnionType
from typing import Any, Union, get_args, get_origin

from models.catalog import model_identity_payload_from_id, module_path_for_model_id
from models.dataset_naming import dataset_class_name_to_cli_name

SKIP_CONFIG_KEYS = {
    "CONFIG_OVERRIDE_SKIP_KEYS",
    "DATASET_OPTIONS",
    "MONITOR_OPTIONS",
}

MODEL_PARAM_ALIASES = {
    "adaptive_submodule_stack_activation": "adaptive_generator_stack_activation",
    "adaptive_submodule_stack_apply_output_pipeline_flag": (
        "adaptive_generator_stack_apply_output_pipeline_flag"
    ),
    "adaptive_submodule_stack_dropout_probability": (
        "adaptive_generator_stack_dropout_probability"
    ),
    "adaptive_submodule_stack_hidden_dim": "adaptive_generator_stack_hidden_dim",
    "adaptive_submodule_stack_last_layer_bias_option": (
        "adaptive_generator_stack_last_layer_bias_option"
    ),
    "adaptive_submodule_stack_layer_norm_position": (
        "adaptive_generator_stack_layer_norm_position"
    ),
    "adaptive_submodule_stack_num_layers": "adaptive_generator_stack_num_layers",
    "adaptive_submodule_stack_residual_connection_option": (
        "adaptive_generator_stack_residual_connection_option"
    ),
    "expert_capacity_factor": "capacity_factor",
    "expert_compute_expert_mixture_flag": "compute_expert_mixture_flag",
    "expert_dropped_token_behavior": "dropped_token_behavior",
    "expert_num_experts": "num_experts",
    "expert_routing_initialization_mode": "routing_initialization_mode",
    "expert_top_k": "top_k",
    "expert_weighted_parameters_flag": "weighted_parameters_flag",
    "expert_weighting_position_option": "weighting_position_option",
    "gate_flag": "stack_gate_flag",
    "halting_flag": "stack_halting_flag",
    "stack_layer_norm_position": "layer_norm_position",
    "weight_generator_depth": "generator_depth",
}

LEGACY_CONFIG_KEY_ALIASES = {
    "adaptive_stack_activation": "adaptive_submodule_stack_activation",
    "adaptive_stack_apply_output_pipeline_flag": (
        "adaptive_submodule_stack_apply_output_pipeline_flag"
    ),
    "adaptive_stack_dropout_probability": (
        "adaptive_submodule_stack_dropout_probability"
    ),
    "adaptive_stack_hidden_dim": "adaptive_submodule_stack_hidden_dim",
    "adaptive_stack_last_layer_bias_option": (
        "adaptive_submodule_stack_last_layer_bias_option"
    ),
    "adaptive_stack_layer_norm_position": (
        "adaptive_submodule_stack_layer_norm_position"
    ),
    "adaptive_stack_num_layers": "adaptive_submodule_stack_num_layers",
    "adaptive_stack_residual_connection_option": (
        "adaptive_submodule_stack_residual_connection_option"
    ),
    "layer_norm_position": "stack_layer_norm_position",
    "gate_bias_flag": "gate_stack_bias_flag",
    "gate_hidden_dim": "gate_stack_hidden_dim",
    "gate_layer_norm_position": "gate_stack_layer_norm_position",
    "halting_bias_flag": "halting_stack_bias_flag",
    "halting_hidden_dim": "halting_stack_hidden_dim",
    "halting_layer_norm_position": "halting_stack_layer_norm_position",
    "memory_bias_flag": "memory_stack_bias_flag",
    "memory_hidden_dim": "memory_stack_hidden_dim",
    "memory_layer_norm_position": "memory_stack_layer_norm_position",
    "recurrent_gate_bias_flag": "recurrent_gate_stack_bias_flag",
    "recurrent_gate_hidden_dim": "recurrent_gate_stack_hidden_dim",
    "recurrent_gate_layer_norm_position": "recurrent_gate_stack_layer_norm_position",
    "recurrent_halting_bias_flag": "recurrent_halting_stack_bias_flag",
    "recurrent_halting_hidden_dim": "recurrent_halting_stack_hidden_dim",
    "recurrent_halting_layer_norm_position": (
        "recurrent_halting_stack_layer_norm_position"
    ),
    "submodule_bias_flag": "submodule_stack_bias_flag",
    "submodule_hidden_dim": "submodule_stack_hidden_dim",
    "submodule_layer_norm_position": "submodule_stack_layer_norm_position",
}


def normalize_key(key: str) -> str:
    return key.strip().replace("-", "_").lower()


def config_key_to_flag(key: str) -> str:
    return "--" + key.lower().replace("_", "-")


def config_key_to_param(key: str) -> str:
    return key.lower()


def config_key_to_model_param(key: str) -> str:
    param = config_key_to_param(key)
    return MODEL_PARAM_ALIASES.get(param, param)


def search_key_to_config_key(key: str) -> str:
    return "SEARCH_SPACE_" + normalize_key(key).upper()


def canonical_config_key(key: str) -> str:
    stripped_key = key.strip().replace("-", "_")
    if stripped_key.isupper():
        return stripped_key
    normalized_key = normalize_key(key)
    return LEGACY_CONFIG_KEY_ALIASES.get(normalized_key, normalized_key).upper()


def _is_supported_constant(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float, str, bool, Enum)):
        return True
    if inspect.isclass(value):
        return True
    return False


def _module_skip_keys(config_module: ModuleType) -> set[str]:
    return {
        key
        for key in getattr(config_module, "CONFIG_OVERRIDE_SKIP_KEYS", ())
        if isinstance(key, str)
    }


def iter_supported_config_keys(config_module: ModuleType) -> list[str]:
    keys = []
    skip_keys = SKIP_CONFIG_KEYS | _module_skip_keys(config_module)
    for key, value in vars(config_module).items():
        if not key.isupper():
            continue
        if key.startswith("SEARCH_SPACE_"):
            continue
        if key in skip_keys:
            continue
        if _is_supported_constant(value):
            keys.append(key)
    return sorted(keys)


def _bool_value(raw_value: str) -> bool:
    value = raw_value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got '{raw_value}'")


def _none_value(raw_value: str) -> None:
    if raw_value.lower() in {"none", "null"}:
        return None
    raise argparse.ArgumentTypeError(f"expected none/null, got '{raw_value}'")


def _class_lookup(config_module: ModuleType, raw_value: str) -> type:
    value = raw_value.split(".")[-1]
    candidate = getattr(config_module, value, None)
    if inspect.isclass(candidate):
        return candidate
    raise argparse.ArgumentTypeError(f"unknown config class '{raw_value}'")


def _enum_lookup(enum_type: type[Enum], raw_value: str) -> Enum:
    value = raw_value.split(".")[-1]
    try:
        return enum_type[value]
    except KeyError as exc:
        choices = ", ".join(enum_type.__members__)
        raise argparse.ArgumentTypeError(
            f"unknown {enum_type.__name__} value '{raw_value}'. Choices: {choices}"
        ) from exc


def _annotation_classes(annotation: Any) -> list[type]:
    if annotation is None:
        return []
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {UnionType, Union}:
        return [item for arg in args for item in _annotation_classes(arg)]
    if origin is type and args and isinstance(args[0], type):
        return [args[0]]
    if inspect.isclass(annotation):
        return [annotation]
    return []


def _parse_from_annotation(
    config_module: ModuleType,
    annotation: Any,
    raw_value: str,
) -> Any:
    lowered = raw_value.lower()
    if lowered in {"none", "null"}:
        return None

    classes = _annotation_classes(annotation)
    enum_classes = [cls for cls in classes if issubclass(cls, Enum)]
    if enum_classes:
        return _enum_lookup(enum_classes[0], raw_value)
    if bool in classes:
        return _bool_value(raw_value)
    if int in classes:
        return int(raw_value)
    if float in classes:
        return float(raw_value)
    if str in classes:
        return raw_value
    if classes:
        return _class_lookup(config_module, raw_value)
    return raw_value


def parse_config_value(
    config_module: ModuleType,
    key: str,
    raw_value: str,
) -> Any:
    current_value = getattr(config_module, key, None)
    if raw_value.lower() in {"none", "null"}:
        return None
    if isinstance(current_value, list):
        sample_value = next(
            (value for value in current_value if value is not None), None
        )
        if sample_value is None:
            if raw_value.lower() in {"none", "null"}:
                return None
            return raw_value
        temp_key = f"__{key}_ITEM"
        setattr(config_module, temp_key, sample_value)
        try:
            return parse_config_value(config_module, temp_key, raw_value)
        finally:
            delattr(config_module, temp_key)
    if isinstance(current_value, bool):
        return _bool_value(raw_value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, str):
        return raw_value
    if isinstance(current_value, Enum):
        return _enum_lookup(type(current_value), raw_value)
    if inspect.isclass(current_value):
        return _class_lookup(config_module, raw_value)
    if current_value is None:
        annotation = getattr(config_module, "__annotations__", {}).get(key)
        return _parse_from_annotation(config_module, annotation, raw_value)
    return raw_value


def parse_search_set(
    config_module: ModuleType,
    raw_value: str,
) -> tuple[str, list[Any]]:
    if "=" not in raw_value:
        raise argparse.ArgumentTypeError(
            "--search-set values must use KEY=v1,v2 syntax"
        )
    raw_key, raw_values = raw_value.split("=", 1)
    raw_config_key = normalize_key(raw_key).upper()
    if not raw_config_key:
        raise argparse.ArgumentTypeError("--search-set key cannot be empty")
    values = [value.strip() for value in raw_values.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--search-set requires at least one value")

    candidate_keys = list(
        dict.fromkeys([canonical_config_key(raw_key), raw_config_key])
    )
    supported_keys = set(iter_supported_config_keys(config_module))
    for value_config_key in candidate_keys:
        search_config_key = search_key_to_config_key(value_config_key)
        if (
            not hasattr(config_module, search_config_key)
            and value_config_key not in supported_keys
        ):
            continue
        parse_key = (
            search_config_key
            if hasattr(config_module, search_config_key)
            else value_config_key
        )
        return config_key_to_model_param(value_config_key), [
            parse_config_value(config_module, parse_key, value) for value in values
        ]
    raise argparse.ArgumentTypeError(f"unknown config key '{raw_key}'")


def add_config_override_arguments(
    parser: argparse.ArgumentParser,
    config_module: ModuleType,
) -> dict[str, str]:
    dest_to_key = {}
    supported_keys = iter_supported_config_keys(config_module)
    registered_flags = set()
    for key in supported_keys:
        dest = f"override_{key.lower()}"
        flag = config_key_to_flag(key)
        parser.add_argument(
            flag,
            dest=dest,
            default=None,
            metavar="VALUE",
            help=argparse.SUPPRESS,
        )
        registered_flags.add(flag)
        dest_to_key[dest] = key
    for legacy_key, canonical_key in LEGACY_CONFIG_KEY_ALIASES.items():
        key = canonical_key.upper()
        if key not in supported_keys:
            continue
        flag = config_key_to_flag(legacy_key)
        if flag in registered_flags:
            continue
        dest = f"override_{key.lower()}"
        parser.add_argument(
            flag,
            dest=dest,
            default=None,
            metavar="VALUE",
            help=argparse.SUPPRESS,
        )
        registered_flags.add(flag)
    parser.add_argument(
        "--search-set",
        action="append",
        default=[],
        metavar="KEY=v1,v2",
        help="Replace/add a search-space axis for --grid-search or --random-search.",
    )
    return dest_to_key


def extract_config_overrides(
    args: argparse.Namespace,
    config_module: ModuleType,
    dest_to_key: dict[str, str],
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    overrides = {}
    for dest, key in dest_to_key.items():
        value = getattr(args, dest, None)
        if value is not None:
            overrides[config_key_to_model_param(key)] = parse_config_value(
                config_module, key, value
            )

    search_overrides = {}
    for raw_search_set in getattr(args, "search_set", []) or []:
        key, values = parse_search_set(config_module, raw_search_set)
        search_overrides[key] = values

    duplicates = set(overrides) & set(search_overrides)
    if duplicates:
        raise ValueError(
            "Parameters cannot be set both as fixed overrides and search axes: "
            f"{sorted(duplicates)}"
        )

    return overrides, search_overrides


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

    skip_keys = SKIP_CONFIG_KEYS | module_skip_keys

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


def _iter_config_assignments(
    config_path: Path,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    source = config_path.read_text()
    tree = ast.parse(source)
    config_options: dict[str, str] = {}
    search_options: dict[str, str] = {}

    if _imports_shared_trainer_config(tree):
        shared_source = Path(__file__).with_name("trainer_config.py").read_text()
        shared_config_options, shared_search_options = (
            _config_assignment_rows_from_source(shared_source)
        )
        config_options.update(shared_config_options)
        search_options.update(shared_search_options)

    local_config_options, local_search_options = _config_assignment_rows_from_source(
        source
    )
    config_options.update(local_config_options)
    search_options.update(local_search_options)

    return list(config_options.items()), list(search_options.items())


def _catalog_source_path(
    experiment: str,
    filename: str,
    models_dir: str = "models",
) -> Path:
    module_path = module_path_for_model_id(experiment)
    if module_path is None:
        raise SystemExit(f"Unknown model: {experiment}")
    relative_package = module_path.removeprefix("models.").replace(".", "/")
    return Path(models_dir) / relative_package / filename


def _display_model_selector(experiment: str) -> str:
    try:
        identity = model_identity_payload_from_id(experiment)
    except ValueError:
        return experiment
    return f"--model-type {identity['modelType']} --model {identity['model']}"


def print_config_options(experiment: str, models_dir: str = "models") -> None:
    config_path = _catalog_source_path(experiment, "config.py", models_dir)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    config_options, search_options = _iter_config_assignments(config_path)

    print(f"Config options for {_display_model_selector(experiment)}:")
    for key, default in config_options:
        print(f"  {config_key_to_flag(key):45} {default}")

    if search_options:
        print("")
        print("Search axes:")
        for key, default in search_options:
            print(f"  {key.lower():45} {default}")


def print_preset_options(experiment: str, models_dir: str = "models") -> None:
    presets_path = _catalog_source_path(experiment, "presets.py", models_dir)
    if not presets_path.exists():
        raise SystemExit(f"Presets file not found: {presets_path}")

    source = presets_path.read_text()
    tree = ast.parse(source)
    print(f"Available presets for {_display_model_selector(experiment)}:")

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "ExperimentPreset":
            continue
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
            description = f"  --  {value}" if isinstance(value, str) else ""
            print(f"  {config_key_to_flag(key)[2:]}{description}")
        return

    raise SystemExit(f"ExperimentPreset not found in {presets_path}")


def _dataset_option_name(item: ast.AST) -> str | None:
    if isinstance(item, ast.Name):
        return item.id
    if isinstance(item, ast.Attribute):
        return item.attr
    return None


def print_dataset_options(experiment: str, models_dir: str = "models") -> None:
    config_path = _catalog_source_path(experiment, "config.py", models_dir)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    source = config_path.read_text()
    tree = ast.parse(source)

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

        if key != "DATASET_OPTIONS" or not isinstance(
            value_node,
            ast.List | ast.Tuple,
        ):
            continue

        dataset_names = [
            name
            for item in value_node.elts
            if (name := _dataset_option_name(item)) is not None
        ]
        if not dataset_names:
            raise SystemExit(f"No DATASET_OPTIONS found for {experiment}")
        for dataset_name in dataset_names:
            print(dataset_class_name_to_cli_name(dataset_name))
        return

    raise SystemExit(f"No DATASET_OPTIONS found for {experiment}")


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


def print_monitor_options(experiment: str, models_dir: str = "models") -> None:
    config_path = _catalog_source_path(experiment, "config.py", models_dir)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    source = config_path.read_text()
    tree = ast.parse(source)

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

        if key != "MONITOR_OPTIONS":
            continue
        if not isinstance(value_node, ast.List | ast.Tuple):
            return

        for item in value_node.elts:
            name = _monitor_option_name(item)
            if name is not None:
                print(name)
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print experiment config options without importing the training stack."
        )
    )
    parser.add_argument("experiment", help="Experiment package under models/.")
    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument(
        "--presets",
        action="store_true",
        help="Print experiment presets instead of config options.",
    )
    list_group.add_argument(
        "--datasets",
        action="store_true",
        help="Print experiment datasets instead of config options.",
    )
    list_group.add_argument(
        "--monitors",
        action="store_true",
        help="Print experiment monitor option names instead of config options.",
    )
    args = parser.parse_args()
    if args.presets:
        print_preset_options(args.experiment)
    elif args.datasets:
        print_dataset_options(args.experiment)
    elif args.monitors:
        print_monitor_options(args.experiment)
    else:
        print_config_options(args.experiment)


if __name__ == "__main__":
    main()

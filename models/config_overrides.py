import argparse
import inspect
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from models.catalog import model_identity_payload_from_id, module_path_for_model_id
from models.config_ast_listing import (
    dataset_option_names_from_source,
    iter_config_assignments,
    monitor_option_names_from_source,
    preset_option_rows_from_source,
)
from models.config_value_parser import parse_config_value

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
    "adaptive_submodule_stack_bias_flag": "adaptive_generator_stack_bias_flag",
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

    config_options, search_options = iter_config_assignments(
        config_path,
        shared_config_path=Path(__file__).with_name("trainer_config.py"),
        base_skip_keys=SKIP_CONFIG_KEYS,
    )

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
    preset_rows = preset_option_rows_from_source(source)

    print(f"Available presets for {_display_model_selector(experiment)}:")

    if preset_rows is not None:
        for key, description in preset_rows:
            description_suffix = f"  --  {description}" if description else ""
            print(f"  {config_key_to_flag(key)[2:]}{description_suffix}")
        return

    raise SystemExit(f"ExperimentPreset not found in {presets_path}")


def print_dataset_options(experiment: str, models_dir: str = "models") -> None:
    config_path = _catalog_source_path(experiment, "config.py", models_dir)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    source = config_path.read_text()
    dataset_names = dataset_option_names_from_source(source)
    if dataset_names:
        for dataset_name in dataset_names:
            print(dataset_name)
        return

    raise SystemExit(f"No DATASET_OPTIONS found for {experiment}")


def print_monitor_options(experiment: str, models_dir: str = "models") -> None:
    config_path = _catalog_source_path(experiment, "config.py", models_dir)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    source = config_path.read_text()
    monitor_names = monitor_option_names_from_source(source)
    if monitor_names is None:
        return
    for name in monitor_names:
        print(name)


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

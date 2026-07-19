import argparse
import importlib
import inspect
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from model_runtime.packages.configuration import (
    MODEL_PARAM_ALIASES as MODEL_PARAM_ALIASES,
)
from model_runtime.packages.configuration import (
    canonical_config_key as canonical_config_key,
)
from model_runtime.packages.configuration import (
    canonical_config_key_for_module,
    config_key_to_flag,
    config_key_to_model_param,
    normalize_key,
    search_key_to_config_key,
)
from model_runtime.packages.configuration import (
    config_key_to_param as config_key_to_param,
)
from models.catalog import model_identity_payload_from_id, module_path_for_model_id
from models.config_ast_listing import (
    dataset_option_names_from_path,
    iter_config_assignments,
    monitor_option_names_from_path,
    preset_option_rows_from_source,
)
from models.config_value_parser import parse_config_value
from models.model_metadata import load_model_metadata_for_config_module

SKIP_CONFIG_KEYS = {
    "CONFIG_SCHEMA_SKIP_KEYS",
    "CONFIG_OVERRIDE_SKIP_KEYS",
    "DEFAULT_EXPERIMENT_TASK",
    "DATASET_OPTIONS_BY_TASK",
    "MONITOR_OPTIONS",
}


def _display_config_default(value: Any) -> str:
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if inspect.isclass(value):
        return value.__name__
    if isinstance(value, str):
        return repr(value)
    return str(value)


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
        if key.startswith("_") or not key.isupper():
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
    search_space_module: ModuleType | None = None,
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

    value_config_key = canonical_config_key_for_module(config_module, raw_key)
    supported_keys = set(iter_supported_config_keys(config_module))
    search_config_key = search_key_to_config_key(value_config_key)
    if search_space_module is None:
        try:
            search_space_module = load_model_metadata_for_config_module(
                config_module
            ).search_space_module
        except Exception:
            search_space_module = config_module
    if (
        not hasattr(search_space_module, search_config_key)
        and value_config_key not in supported_keys
    ):
        raise argparse.ArgumentTypeError(f"unknown config key '{raw_key}'")
    parse_key = (
        search_config_key
        if hasattr(search_space_module, search_config_key)
        else value_config_key
    )
    parse_module = (
        search_space_module if parse_key == search_config_key else config_module
    )
    return config_key_to_model_param(value_config_key), [
        parse_config_value(parse_module, parse_key, value) for value in values
    ]


def add_config_override_arguments(
    parser: argparse.ArgumentParser,
    config_module: ModuleType,
    search_space_module: ModuleType | None = None,
) -> dict[str, str]:
    dest_to_key = {}
    supported_keys = iter_supported_config_keys(config_module)
    supported_key_set = set(supported_keys)
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
        dest_to_key[dest] = key
    if search_space_module is None:
        try:
            search_space_module = load_model_metadata_for_config_module(
                config_module
            ).search_space_module
        except Exception:
            search_space_module = None
    if search_space_module is not None:
        supported_by_model_param = {
            config_key_to_model_param(key): key for key in supported_keys
        }
        for search_key, value in vars(search_space_module).items():
            if not search_key.startswith("SEARCH_SPACE_") or not isinstance(
                value,
                list,
            ):
                continue
            alias_key = search_key[len("SEARCH_SPACE_") :]
            if alias_key in supported_key_set:
                continue
            target_key = supported_by_model_param.get(
                config_key_to_model_param(alias_key)
            )
            if target_key is None:
                continue
            dest = f"override_{alias_key.lower()}"
            parser.add_argument(
                config_key_to_flag(alias_key),
                dest=dest,
                default=None,
                metavar="VALUE",
                help=argparse.SUPPRESS,
            )
            dest_to_key[dest] = target_key
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
    search_space_module: ModuleType | None = None,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    overrides = {}
    for dest, key in dest_to_key.items():
        value = getattr(args, dest, None)
        if value is not None:
            overrides[config_key_to_model_param(key)] = parse_config_value(
                config_module, key, value
            )

    search_overrides = {}
    if search_space_module is None:
        search_space_module = getattr(args, "_search_space_module", None)
    for raw_search_set in getattr(args, "search_set", []) or []:
        key, values = parse_search_set(
            config_module,
            raw_search_set,
            search_space_module,
        )
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
    models_dir: str | Path | None = None,
) -> Path:
    module_path = module_path_for_model_id(experiment)
    if module_path is None:
        raise SystemExit(f"Unknown model: {experiment}")
    relative_package = module_path.removeprefix("models.").replace(".", "/")
    source_root = Path(models_dir) if models_dir is not None else Path(__file__).parent
    return source_root / relative_package / filename


def _display_model_selector(experiment: str) -> str:
    try:
        identity = model_identity_payload_from_id(experiment)
    except ValueError:
        return experiment
    return f"--model-type {identity['modelType']} --model {identity['model']}"


def print_config_options(
    experiment: str,
    models_dir: str | Path | None = None,
) -> None:
    source_root = Path(models_dir) if models_dir is not None else Path(__file__).parent
    config_path = _catalog_source_path(experiment, "config.py", models_dir)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    search_space_path = _catalog_source_path(experiment, "search_space.py", models_dir)
    if not search_space_path.exists():
        raise SystemExit(f"Search space file not found: {search_space_path}")

    config_options, search_options = iter_config_assignments(
        config_path,
        search_space_path=search_space_path,
        base_skip_keys=SKIP_CONFIG_KEYS,
        models_dir=source_root,
    )
    module_path = module_path_for_model_id(experiment)
    if module_path is not None:
        config_module = importlib.import_module(f"{module_path}.config")
        supported_keys = set(iter_supported_config_keys(config_module))
        config_options = [
            (key, _display_config_default(getattr(config_module, key)))
            for key, _default in config_options
            if key in supported_keys
        ]

    print(f"Config options for {_display_model_selector(experiment)}:")
    for key, default in config_options:
        print(f"  {config_key_to_flag(key):45} {default}")

    if search_options:
        print("")
        print("Search axes:")
        for key, default in search_options:
            print(f"  {key.lower():45} {default}")


def print_preset_options(
    experiment: str,
    models_dir: str | Path | None = None,
) -> None:
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


def print_dataset_options(
    experiment: str,
    models_dir: str | Path | None = None,
) -> None:
    source_root = Path(models_dir) if models_dir is not None else Path(__file__).parent
    dataset_options_path = _catalog_source_path(
        experiment,
        "dataset_options.py",
        models_dir,
    )
    if not dataset_options_path.exists():
        raise SystemExit(f"Dataset options file not found: {dataset_options_path}")

    dataset_names = dataset_option_names_from_path(
        dataset_options_path,
        models_dir=source_root,
    )
    if dataset_names:
        for dataset_name in dataset_names:
            print(dataset_name)
        return

    raise SystemExit(f"No DATASET_OPTIONS_BY_TASK found for {experiment}")


def print_monitor_options(
    experiment: str,
    models_dir: str | Path | None = None,
) -> None:
    source_root = Path(models_dir) if models_dir is not None else Path(__file__).parent
    monitor_options_path = _catalog_source_path(
        experiment,
        "monitor_options.py",
        models_dir,
    )
    if not monitor_options_path.exists():
        raise SystemExit(f"Monitor options file not found: {monitor_options_path}")

    monitor_names = monitor_option_names_from_path(
        monitor_options_path,
        models_dir=source_root,
    )
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
    parser.add_argument("experiment", help="Experiment in the models import package.")
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

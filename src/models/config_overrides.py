from __future__ import annotations

import argparse
import inspect
from enum import Enum
from typing import Any

from model_runtime.packages import (
    ModelPackage,
)
from model_runtime.packages import (
    canonical_config_key as _canonical_config_key,
)
from model_runtime.packages import (
    config_key_to_flag as _config_key_to_flag,
)
from model_runtime.packages import (
    config_key_to_model_param as _config_key_to_model_param,
)
from model_runtime.packages import (
    dataset_cli_name as _dataset_cli_name,
)
from model_runtime.packages import (
    iter_supported_config_keys as _iter_supported_config_keys,
)
from model_runtime.packages import (
    normalize_key as _normalize_key,
)
from model_runtime.packages import (
    parse_config_value as _parse_config_value,
)
from model_runtime.packages import (
    search_key_to_config_key as _search_key_to_config_key,
)
from models.catalog import model_package


def _display_config_default(value: Any) -> str:
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if inspect.isclass(value):
        return value.__name__
    if isinstance(value, str):
        return repr(value)
    return str(value)


def parse_search_set(
    package: ModelPackage,
    raw_value: str,
) -> tuple[str, list[Any]]:
    if not isinstance(package, ModelPackage):
        raise TypeError("Search parsing requires a selected ModelPackage.")
    if "=" not in raw_value:
        raise argparse.ArgumentTypeError(
            "--search-set values must use KEY=v1,v2 syntax"
        )
    raw_key, raw_values = raw_value.split("=", 1)
    raw_config_key = _normalize_key(raw_key).upper()
    if not raw_config_key:
        raise argparse.ArgumentTypeError("--search-set key cannot be empty")
    values = [value.strip() for value in raw_values.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--search-set requires at least one value")

    config_module = package.runtime_defaults
    search_space = package.metadata.search_space
    config_key = _canonical_config_key(raw_key)
    supported_keys = set(_iter_supported_config_keys(config_module))
    search_key = _search_key_to_config_key(config_key)
    if not hasattr(search_space, search_key) and config_key not in supported_keys:
        raise argparse.ArgumentTypeError(f"unknown Runtime Defaults key '{raw_key}'")
    parse_key = search_key if hasattr(search_space, search_key) else config_key
    parse_module = search_space if parse_key == search_key else config_module
    return _config_key_to_model_param(config_key), [
        _parse_config_value(parse_module, parse_key, value) for value in values
    ]


def add_config_override_arguments(
    parser: argparse.ArgumentParser,
    package: ModelPackage,
) -> dict[str, str]:
    if not isinstance(package, ModelPackage):
        raise TypeError("CLI overrides require a selected ModelPackage.")
    dest_to_key: dict[str, str] = {}
    for key in _iter_supported_config_keys(package.runtime_defaults):
        dest = f"override_{key.lower()}"
        parser.add_argument(
            _config_key_to_flag(key),
            dest=dest,
            default=None,
            metavar="VALUE",
            help=argparse.SUPPRESS,
        )
        dest_to_key[dest] = key
    parser.add_argument(
        "--search-set",
        action="append",
        default=[],
        metavar="KEY=v1,v2",
        help="Replace one canonical search axis for grid or random search.",
    )
    return dest_to_key


def extract_config_overrides(
    args: argparse.Namespace,
    package: ModelPackage,
    dest_to_key: dict[str, str],
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    if not isinstance(package, ModelPackage):
        raise TypeError("CLI overrides require a selected ModelPackage.")
    overrides: dict[str, Any] = {}
    for dest, key in dest_to_key.items():
        value = getattr(args, dest, None)
        if value is not None:
            overrides[_config_key_to_model_param(key)] = _parse_config_value(
                package.runtime_defaults,
                key,
                value,
            )

    search_overrides: dict[str, list[Any]] = {}
    for raw_search_set in getattr(args, "search_set", []) or []:
        key, values = parse_search_set(package, raw_search_set)
        search_overrides[key] = values

    duplicates = set(overrides) & set(search_overrides)
    if duplicates:
        raise ValueError(
            "Parameters cannot be set both as fixed overrides and search axes: "
            f"{sorted(duplicates)}"
        )
    return overrides, search_overrides


def _selected_package(catalog_key: str) -> ModelPackage:
    package = model_package(catalog_key)
    if package is None:
        raise SystemExit(f"Unknown Model Package: {catalog_key}")
    return package


def _display_model_selector(package: ModelPackage) -> str:
    identity = package.identity
    return f"--model-type {identity.model_type} --model {identity.model}"


def print_config_options(catalog_key: str) -> None:
    package = _selected_package(catalog_key)
    config_module = package.runtime_defaults
    print(f"Runtime Defaults for {_display_model_selector(package)}:")
    for key in _iter_supported_config_keys(config_module):
        print(
            f"  {_config_key_to_flag(key):45} "
            f"{_display_config_default(getattr(config_module, key))}"
        )

    search_items = package.search_metadata
    if search_items:
        print("")
        print("Search axes:")
        for key in sorted(search_items):
            print(f"  {key.lower():45} {_display_config_default(search_items[key])}")


def print_preset_options(catalog_key: str) -> None:
    package = _selected_package(catalog_key)
    print(f"Available presets for {_display_model_selector(package)}:")
    for preset in package.preset_type:
        description = package.preset_description(preset)
        suffix = f"  --  {description}" if description else ""
        print(f"  {package.preset_name(preset)}{suffix}")


def print_dataset_options(catalog_key: str) -> None:
    package = _selected_package(catalog_key)
    seen: set[str] = set()
    for datasets in package.dataset_metadata.values():
        for dataset in datasets:
            name = _dataset_cli_name(dataset)
            if name not in seen:
                seen.add(name)
                print(name)


def print_monitor_options(catalog_key: str) -> None:
    package = _selected_package(catalog_key)
    for option in package.monitor_options():
        print(option.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print canonical Model Package metadata."
    )
    parser.add_argument("catalog_key", help="Canonical <modelType>/<model> identity.")
    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument("--presets", action="store_true")
    list_group.add_argument("--datasets", action="store_true")
    list_group.add_argument("--monitors", action="store_true")
    args = parser.parse_args()
    if args.presets:
        print_preset_options(args.catalog_key)
    elif args.datasets:
        print_dataset_options(args.catalog_key)
    elif args.monitors:
        print_monitor_options(args.catalog_key)
    else:
        print_config_options(args.catalog_key)


if __name__ == "__main__":
    main()


__all__ = [
    "add_config_override_arguments",
    "extract_config_overrides",
    "parse_search_set",
    "print_config_options",
    "print_dataset_options",
    "print_monitor_options",
    "print_preset_options",
]

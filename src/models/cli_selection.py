from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Literal

from emperor.config import BaseOptions
from emperor.experiments import ExperimentTask
from emperor.monitoring import MonitorSettings
from model_runtime.packages import (
    ModelPackage,
    normalize_key,
)
from models.config_overrides import extract_config_overrides
from models.experiment_cli_parser import preset_name_to_cli


@dataclass(frozen=True)
class CliSelection:
    """Validated state parsed from one Model Package CLI invocation."""

    experiment_task: ExperimentTask = field(
        metadata={"help": "Experiment Task selected by the CLI."}
    )
    preset: BaseOptions | None = field(
        metadata={"help": "Primary selected preset, or None for all presets."}
    )
    selected_presets: list[BaseOptions] | None = field(
        metadata={"help": "Ordered multi-preset selection, when requested."}
    )
    search_mode: Literal["grid", "random"] | None = field(
        metadata={"help": "Selected grid/random search mode, if any."}
    )
    random_samples: int | None = field(
        metadata={"help": "Requested random-search sample count, if any."}
    )
    search_keys: list[str] | None = field(
        metadata={"help": "Canonical search axes selected by --search-keys."}
    )
    config_overrides: dict = field(
        metadata={"help": "Parsed canonical Runtime Defaults overrides."}
    )
    search_overrides: dict = field(
        metadata={"help": "Parsed canonical search-axis overrides."}
    )
    monitor_names: list[str] = field(
        metadata={"help": "Deduplicated selected monitor names."}
    )
    monitor_callbacks: list = field(
        metadata={"help": "Monitor callbacks built for a direct terminal run."}
    )


def model_monitor_options(package: ModelPackage):
    if not isinstance(package, ModelPackage):
        raise TypeError("CLI selection requires a selected ModelPackage.")
    return package.monitor_options()


def resolve_monitor_options(
    package: ModelPackage,
    monitor_names: list[str] | None,
):
    if not isinstance(package, ModelPackage):
        raise TypeError("CLI selection requires a selected ModelPackage.")
    try:
        return package.resolve_monitors(monitor_names)
    except ValueError as exc:
        message = str(exc).replace(
            "Unknown monitor option(s) for model", "Unknown --monitors for model"
        )
        raise ValueError(message) from exc


def monitor_settings_from_package(
    package: ModelPackage,
    config_overrides: dict | None = None,
) -> MonitorSettings:
    default_log_every_n_steps = getattr(
        package.runtime_defaults,
        "MONITOR_LOG_EVERY_N_STEPS",
        100,
    )
    log_every_n_steps = (config_overrides or {}).get(
        "monitor_log_every_n_steps",
        default_log_every_n_steps,
    )
    return MonitorSettings(log_every_n_steps=int(log_every_n_steps))


def resolve_monitor_callbacks(
    package: ModelPackage,
    monitor_names: list[str] | None,
    config_overrides: dict | None = None,
) -> list:
    settings = monitor_settings_from_package(package, config_overrides)
    return [
        option.build_callback(settings)
        for option in resolve_monitor_options(package, monitor_names)
    ]


def resolve_cli_selection(
    args: argparse.Namespace,
    package: ModelPackage,
    preset_enum: type[BaseOptions],
    no_search_presets: list[str] | None = None,
    *,
    build_monitor_callbacks: bool = True,
) -> CliSelection:
    if not isinstance(package, ModelPackage):
        raise TypeError("CLI selection requires a selected ModelPackage.")
    if getattr(args, "list_config", False):
        from models.config_overrides import print_config_options

        print_config_options(package.identity.catalog_key)
        raise SystemExit(0)

    experiment_task = package.resolve_experiment_task(
        getattr(args, "experiment_task", None)
    )

    selected_presets = None
    if args.all_presets:
        preset = None
    elif getattr(args, "presets", None):
        selected_presets = [
            preset_enum.get_member(preset_name) for preset_name in args.presets
        ]
        preset = selected_presets[0]
    else:
        preset = preset_enum.get_member(args.preset)

    if args.random_search is not None:
        search_mode: Literal["grid", "random"] | None = "random"
        random_samples = args.random_search
    elif args.grid_search:
        search_mode = "grid"
        random_samples = None
    else:
        search_mode = None
        random_samples = None
    no_search_preset_names = set(no_search_presets or [])
    search_checked_presets = selected_presets or (
        [preset] if preset is not None else []
    )
    if (
        not args.all_presets
        and search_mode is not None
        and any(
            selected.name in no_search_preset_names
            for selected in search_checked_presets
        )
    ):
        blocked_presets = [
            selected.name
            for selected in search_checked_presets
            if selected.name in no_search_preset_names
        ]
        blocked_cli_presets = ", ".join(
            preset_name_to_cli(name) for name in blocked_presets
        )
        raise ValueError(
            f"'{blocked_cli_presets}' does not support --grid-search or "
            "--random-search. Use CONFIG instead."
        )
    if args.search_keys is not None and search_mode is None:
        raise ValueError("--search-keys requires --grid-search or --random-search.")
    if getattr(args, "search_set", None) and search_mode is None:
        raise ValueError("--search-set requires --grid-search or --random-search.")
    search_keys = (
        [normalize_key(key) for key in args.search_keys]
        if args.search_keys is not None
        else None
    )

    config_overrides, search_overrides = extract_config_overrides(
        args,
        package,
        getattr(args, "_config_override_dests", {}),
    )
    monitor_options = resolve_monitor_options(
        package,
        getattr(args, "monitors", None),
    )
    monitor_settings = monitor_settings_from_package(package, config_overrides)
    return CliSelection(
        experiment_task=experiment_task,
        preset=preset,
        selected_presets=selected_presets,
        search_mode=search_mode,
        random_samples=random_samples,
        search_keys=search_keys,
        config_overrides=config_overrides,
        search_overrides=search_overrides,
        monitor_names=[option.name for option in monitor_options],
        monitor_callbacks=(
            [option.build_callback(monitor_settings) for option in monitor_options]
            if build_monitor_callbacks
            else []
        ),
    )


__all__ = [
    "CliSelection",
    "model_monitor_options",
    "monitor_settings_from_package",
    "resolve_cli_selection",
    "resolve_monitor_callbacks",
    "resolve_monitor_options",
]

import argparse
from dataclasses import dataclass, field
from types import ModuleType

from emperor.base.options import BaseOptions
from emperor.experiments.base import GridSearch, RandomSearch, SearchMode
from emperor.experiments.monitors import MonitorOption, MonitorSettings
from emperor.experiments.tasks import ExperimentTask, resolve_experiment_task

from models.config_overrides import (
    extract_config_overrides,
    normalize_key,
    print_config_options,
)
from models.experiment_cli_parser import preset_name_to_cli
from models.model_metadata import load_model_metadata_for_config_module


@dataclass(frozen=True)
class ExperimentMode:
    experiment_task: ExperimentTask | None = field(
        metadata={
            "help": (
                "Experiment task selected by --experiment-task, or the model "
                "package default task when omitted."
            )
        }
    )
    preset: BaseOptions | None = field(
        metadata={
            "help": (
                "Primary experiment preset selected from --preset/--presets, "
                "or None when --all-presets is used."
            )
        }
    )
    selected_presets: list[BaseOptions] | None = field(
        metadata={
            "help": (
                "Ordered preset list selected by --presets, or None when a "
                "single preset/all presets mode is used."
            )
        }
    )
    search_mode: SearchMode = field(
        metadata={
            "help": "GridSearch, RandomSearch, or None depending on CLI search flags."
        }
    )
    search_keys: list[str] | None = field(
        metadata={
            "help": (
                "Normalized SEARCH_SPACE_* axis names requested by "
                "--search-keys, or None for the full search space."
            )
        }
    )
    config_overrides: dict = field(
        metadata={
            "help": "Parsed fixed config override values from model config flags."
        }
    )
    search_overrides: dict = field(
        metadata={"help": "Parsed search-axis override values from --search-set."}
    )
    monitor_names: list[str] = field(
        metadata={"help": "Deduplicated monitor names selected by --monitors."}
    )
    monitor_callbacks: list = field(
        metadata={"help": "Monitor callback instances selected for this terminal run."}
    )


def _monitor_options_module_for_config_module(
    config_module: ModuleType,
) -> ModuleType:
    return load_model_metadata_for_config_module(config_module).monitor_options_module


def model_monitor_options(
    config_module: ModuleType | None,
    monitor_options_module: ModuleType | None = None,
) -> list[MonitorOption]:
    if config_module is None:
        return []
    if monitor_options_module is None:
        try:
            monitor_options_module = _monitor_options_module_for_config_module(
                config_module
            )
        except Exception:
            monitor_options_module = config_module
    raw_options = getattr(monitor_options_module, "MONITOR_OPTIONS", [])
    if raw_options is None:
        return []
    options = list(raw_options)
    invalid_options = [
        type(option).__name__
        for option in options
        if not isinstance(option, MonitorOption)
    ]
    if invalid_options:
        raise ValueError(
            f"Model monitor options '{monitor_options_module.__name__}' has invalid "
            f"MONITOR_OPTIONS entries: {', '.join(invalid_options)}."
        )
    option_names = [option.name for option in options]
    duplicate_names = sorted(
        name for name in set(option_names) if option_names.count(name) > 1
    )
    if duplicate_names:
        raise ValueError(
            f"Model monitor options '{monitor_options_module.__name__}' has "
            f"duplicate monitor options: {', '.join(duplicate_names)}."
        )
    return options


def resolve_monitor_options(
    config_module: ModuleType | None,
    monitor_names: list[str] | None,
    monitor_options_module: ModuleType | None = None,
) -> list[MonitorOption]:
    if not monitor_names:
        return []
    options_by_name = {
        option.name: option
        for option in model_monitor_options(config_module, monitor_options_module)
    }
    selected = []
    unknown = []
    seen = set()
    for name in monitor_names:
        if name in seen:
            continue
        seen.add(name)
        option = options_by_name.get(name)
        if option is None:
            unknown.append(name)
            continue
        selected.append(option)
    if unknown:
        valid = ", ".join(sorted(options_by_name)) or "none"
        raise ValueError(f"Unknown --monitors: {unknown}. Valid monitors: {valid}")
    return selected


def monitor_settings_from_config(
    config_module: ModuleType | None,
    config_overrides: dict | None = None,
) -> MonitorSettings:
    default_log_every_n_steps = (
        getattr(config_module, "MONITOR_LOG_EVERY_N_STEPS", 100)
        if config_module is not None
        else 100
    )
    log_every_n_steps = (config_overrides or {}).get(
        "monitor_log_every_n_steps",
        default_log_every_n_steps,
    )
    return MonitorSettings(log_every_n_steps=int(log_every_n_steps))


def resolve_monitor_callbacks(
    config_module: ModuleType | None,
    monitor_names: list[str] | None,
    monitor_options_module: ModuleType | None = None,
    config_overrides: dict | None = None,
) -> list:
    settings = monitor_settings_from_config(config_module, config_overrides)
    return [
        option.build_callback(settings)
        for option in resolve_monitor_options(
            config_module,
            monitor_names,
            monitor_options_module,
        )
    ]


def resolve_experiment_mode(
    args: argparse.Namespace,
    preset_enum: type[BaseOptions],
    no_search_presets: list[str] | None = None,
) -> ExperimentMode:
    if getattr(args, "list_config", False):
        print_config_options(getattr(args, "_config_experiment", ""))
        raise SystemExit(0)

    metadata = getattr(args, "_model_metadata", None)
    raw_experiment_task = getattr(args, "experiment_task", None)
    if metadata is not None:
        experiment_task = (
            metadata.default_experiment_task
            if raw_experiment_task is None
            else resolve_experiment_task(raw_experiment_task)
        )
        metadata.dataset_options_for_task(experiment_task)
    else:
        experiment_task = resolve_experiment_task(raw_experiment_task)

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
        search_mode: SearchMode = RandomSearch(args.random_search)
    elif args.grid_search:
        search_mode = GridSearch()
    else:
        search_mode = None
    no_search_preset_names = set(no_search_presets or [])
    search_checked_presets = selected_presets or (
        [preset] if preset is not None else []
    )
    if (
        not args.all_presets
        and search_mode is not None
        and any(
            preset.name in no_search_preset_names for preset in search_checked_presets
        )
    ):
        blocked_presets = [
            preset.name
            for preset in search_checked_presets
            if preset.name in no_search_preset_names
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
    config_module = getattr(args, "_config_module", None)
    if config_module is None:
        config_overrides, search_overrides = {}, {}
    else:
        config_overrides, search_overrides = extract_config_overrides(
            args,
            config_module,
            getattr(args, "_config_override_dests", {}),
            getattr(metadata, "search_space_module", None),
        )
    monitor_options = resolve_monitor_options(
        config_module,
        getattr(args, "monitors", None),
        getattr(metadata, "monitor_options_module", None),
    )
    monitor_settings = monitor_settings_from_config(config_module, config_overrides)
    return ExperimentMode(
        experiment_task=experiment_task,
        preset=preset,
        selected_presets=selected_presets,
        search_mode=search_mode,
        search_keys=search_keys,
        config_overrides=config_overrides,
        search_overrides=search_overrides,
        monitor_names=[option.name for option in monitor_options],
        monitor_callbacks=[
            option.build_callback(monitor_settings) for option in monitor_options
        ],
    )

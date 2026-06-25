import argparse
import importlib
from dataclasses import dataclass, field
from types import ModuleType

from emperor.base.options import BaseOptions
from emperor.experiments.base import GridSearch, RandomSearch, SearchMode
from emperor.experiments.monitors import MonitorOption

from models.catalog import public_id_for_module
from models.config_overrides import (
    add_config_override_arguments,
    extract_config_overrides,
    normalize_key,
    print_config_options,
)
from models.dataset_naming import dataset_cli_name, dataset_name


class _ExperimentParser(argparse.ArgumentParser):
    pass


@dataclass(frozen=True)
class ExperimentMode:
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


def get_experiment_parser(
    config_choices: list | None = None,
    experiment_package: str | None = None,
) -> _ExperimentParser:
    parser = _ExperimentParser(
        description="Run an experiment with a named configuration.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    choices_text = ""
    cli_config_choices = None
    if config_choices:
        cli_config_choices = [preset_name_to_cli(choice) for choice in config_choices]
        choices_text = "\nAvailable presets:\n" + "\n".join(
            f"  {choice}" for choice in cli_config_choices
        )

    preset_group = parser.add_mutually_exclusive_group(required=True)

    preset_group.add_argument(
        "--preset",
        dest="preset",
        type=str,
        help="Name of the experiment preset to run." + choices_text,
        choices=cli_config_choices,
        metavar="PRESET_NAME",
    )

    preset_group.add_argument(
        "--presets",
        dest="presets",
        nargs="+",
        type=str,
        help="Names of experiment presets to run sequentially." + choices_text,
        choices=cli_config_choices,
        metavar="PRESET_NAME",
    )

    preset_group.add_argument(
        "--all-presets",
        dest="all_presets",
        action="store_true",
        help="Run all experiment presets sequentially.",
    )

    preset_group.add_argument(
        "--list-config",
        action="store_true",
        help="Print overridable config flags and defaults, then exit.",
    )

    search_group = parser.add_mutually_exclusive_group()

    search_group.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search over all combinations in the search space.",
    )

    search_group.add_argument(
        "--random-search",
        type=int,
        metavar="N",
        help="Run random search with N sampled combinations from the search space.",
    )

    parser.add_argument(
        "--search-keys",
        nargs="+",
        type=str,
        default=None,
        metavar="KEY",
        help=(
            "Restrict sweep to named SEARCH_SPACE_* axes "
            "(e.g. STACK_HIDDEN_DIM STACK_NUM_LAYERS).\n"
            "Requires --grid-search or --random-search."
        ),
    )

    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help=(
            "Custom folder name for storing experiment logs. If not provided, "
            "the model file name is used.\n"
            "Use the same folder across models to compare them in TensorBoard."
        ),
    )

    parser.add_argument(
        "--datasets",
        "--dataset",
        nargs="+",
        default=None,
        metavar="DATASET",
        help=(
            "Restrict training to one or more lowercase dataset names, "
            "e.g. mnist cifar10."
        ),
    )

    parser.add_argument(
        "--monitors",
        nargs="+",
        default=None,
        metavar="MONITOR",
        help="Enable one or more monitor callbacks for this training run.",
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="Group config override flags, e.g. --config --num-epochs 30.",
    )

    parser.set_defaults(_config_override_dests={})
    if experiment_package is not None:
        config_module = importlib.import_module(f"{experiment_package}.config")
        experiment_id = public_id_for_module(experiment_package)
        if experiment_id is None:
            experiment_id = experiment_package.removeprefix("models.").replace(".", "/")
        parser.set_defaults(
            _config_module=config_module,
            _config_experiment=experiment_id,
            _config_override_dests=add_config_override_arguments(parser, config_module),
        )

    return parser


def preset_name_to_cli(name: str) -> str:
    return name.lower().replace("_", "-")


def resolve_dataset_names(
    dataset_options: list[type],
    dataset_names: list[str] | None,
) -> list[type]:
    if not dataset_names:
        return list(dataset_options)
    by_name = {}
    for dataset in dataset_options:
        by_name[dataset_name(dataset)] = dataset
        by_name[dataset_name(dataset).lower()] = dataset
        by_name[dataset_cli_name(dataset)] = dataset
    resolved = []
    unknown = []
    seen = set()
    for name in dataset_names:
        dataset = by_name.get(name) or by_name.get(name.lower())
        if dataset is None:
            unknown.append(name)
            continue
        if dataset_name(dataset) in seen:
            continue
        seen.add(dataset_name(dataset))
        resolved.append(dataset)
    if unknown:
        valid = ", ".join(dataset_cli_name(dataset) for dataset in dataset_options)
        raise ValueError(f"Unknown --datasets: {unknown}. Valid datasets: {valid}")
    return resolved


def model_monitor_options(config_module: ModuleType | None) -> list[MonitorOption]:
    if config_module is None:
        return []
    raw_options = getattr(config_module, "MONITOR_OPTIONS", [])
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
            f"Model config '{config_module.__name__}' has invalid MONITOR_OPTIONS "
            f"entries: {', '.join(invalid_options)}."
        )
    option_names = [option.name for option in options]
    duplicate_names = sorted(
        name for name in set(option_names) if option_names.count(name) > 1
    )
    if duplicate_names:
        raise ValueError(
            f"Model config '{config_module.__name__}' has duplicate monitor "
            f"options: {', '.join(duplicate_names)}."
        )
    return options


def resolve_monitor_options(
    config_module: ModuleType | None,
    monitor_names: list[str] | None,
) -> list[MonitorOption]:
    if not monitor_names:
        return []
    options_by_name = {
        option.name: option for option in model_monitor_options(config_module)
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


def resolve_monitor_callbacks(
    config_module: ModuleType | None,
    monitor_names: list[str] | None,
) -> list:
    return [
        option.build_callback()
        for option in resolve_monitor_options(config_module, monitor_names)
    ]


def resolve_experiment_mode(
    args: argparse.Namespace,
    preset_enum: type[BaseOptions],
    no_search_presets: list[str] | None = None,
) -> ExperimentMode:
    if getattr(args, "list_config", False):
        print_config_options(getattr(args, "_config_experiment", ""))
        raise SystemExit(0)

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
        )
    monitor_options = resolve_monitor_options(
        config_module,
        getattr(args, "monitors", None),
    )
    return ExperimentMode(
        preset=preset,
        selected_presets=selected_presets,
        search_mode=search_mode,
        search_keys=search_keys,
        config_overrides=config_overrides,
        search_overrides=search_overrides,
        monitor_names=[option.name for option in monitor_options],
        monitor_callbacks=[option.build_callback() for option in monitor_options],
    )

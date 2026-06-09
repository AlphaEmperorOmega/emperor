import argparse
import importlib
from emperor.base.options import BaseOptions
from emperor.experiments.base import GridSearch, RandomSearch, SearchMode
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

    option_group = parser.add_mutually_exclusive_group(required=True)

    option_group.add_argument(
        "--preset",
        dest="option",
        type=str,
        help="Name of the experiment preset to run." + choices_text,
        choices=cli_config_choices,
        metavar="PRESET_NAME",
    )

    option_group.add_argument(
        "--presets",
        dest="options",
        nargs="+",
        type=str,
        help="Names of experiment presets to run sequentially." + choices_text,
        choices=cli_config_choices,
        metavar="PRESET_NAME",
    )

    option_group.add_argument(
        "--all-presets",
        dest="all_options",
        action="store_true",
        help="Run all experiment presets sequentially.",
    )

    option_group.add_argument(
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
        help="Restrict sweep to named SEARCH_SPACE_* axes (e.g. HIDDEN_DIM STACK_NUM_LAYERS).\nRequires --grid-search or --random-search.",
    )

    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Custom folder name for storing experiment logs. If not provided, the model file name is used.\nUse the same folder across models to compare them in TensorBoard.",
    )

    parser.add_argument(
        "--datasets",
        "--dataset",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Restrict training to one or more dataset class names, e.g. Mnist Cifar10.",
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
        valid = ", ".join(dataset_name(dataset) for dataset in dataset_options)
        raise ValueError(f"Unknown --datasets: {unknown}. Valid datasets: {valid}")
    return resolved


def resolve_experiment_mode(
    args: argparse.Namespace,
    options_enum: type[BaseOptions],
    no_search_options: list[str] | None = None,
) -> tuple[
    BaseOptions | None,
    list[BaseOptions] | None,
    SearchMode,
    list[str] | None,
    dict,
    dict,
]:
    if getattr(args, "list_config", False):
        print_config_options(getattr(args, "_config_experiment", ""))
        raise SystemExit(0)

    selected_options = None
    if args.all_options:
        config_option = None
    elif getattr(args, "options", None):
        selected_options = [options_enum.get_option(option) for option in args.options]
        config_option = selected_options[0]
    else:
        config_option = options_enum.get_option(args.option)

    if args.random_search is not None:
        search_mode: SearchMode = RandomSearch(args.random_search)
    elif args.grid_search:
        search_mode = GridSearch()
    else:
        search_mode = None
    no_search_option_names = set(no_search_options or ["PRESET"])
    search_checked_options = selected_options or (
        [config_option] if config_option is not None else []
    )
    if (
        not args.all_options
        and search_mode is not None
        and any(option.name in no_search_option_names for option in search_checked_options)
    ):
        blocked_options = [
            option.name
            for option in search_checked_options
            if option.name in no_search_option_names
        ]
        blocked_cli_options = ", ".join(preset_name_to_cli(name) for name in blocked_options)
        raise ValueError(
            f"'{blocked_cli_options}' does not support --grid-search or --random-search. Use CONFIG instead."
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
    return (
        config_option,
        selected_options,
        search_mode,
        search_keys,
        config_overrides,
        search_overrides,
    )

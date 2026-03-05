import argparse
from emperor.base.enums import BaseOptions
from emperor.experiments.base import GridSearch, RandomSearch, SearchMode


class _ExperimentParser(argparse.ArgumentParser):
    pass


def get_experiment_parser(
    config_choices: list | None = None,
) -> _ExperimentParser:
    parser = _ExperimentParser(
        description="Run an experiment with a named configuration.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    choices_text = ""
    if config_choices:
        choices_text = "\nAvailable configurations:\n" + "\n".join(
            f"  {c}" for c in config_choices
        )

    name_group = parser.add_mutually_exclusive_group(required=True)

    name_group.add_argument(
        "--name",
        type=str,
        help="Name of the experiment configuration to run." + choices_text,
        choices=config_choices,
        metavar="CONFIG_NAME",
    )

    name_group.add_argument(
        "--all-options",
        action="store_true",
        help="Run all experiment configurations sequentially.",
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
        "--logdir",
        type=str,
        default=None,
        help="Custom folder name for storing experiment logs. If not provided, the model file name is used.\nUse the same folder across models to compare them in TensorBoard.",
    )

    return parser


def resolve_experiment_mode(
    args: argparse.Namespace,
    options_enum: type[BaseOptions],
    no_search_options: list[str] = ["PRESET"],
) -> tuple[BaseOptions | None, SearchMode]:
    config_option = None if args.all_options else options_enum.get_option(args.name)
    if args.random_search is not None:
        search_mode: SearchMode = RandomSearch(args.random_search)
    elif args.grid_search:
        search_mode = GridSearch()
    else:
        search_mode = None
    if not args.all_options and search_mode is not None and args.name in no_search_options:
        raise ValueError(
            f"'{args.name}' does not support --grid-search or --random-search. Use CONFIG instead."
        )
    return config_option, search_mode

from __future__ import annotations

import argparse

from emperor.config import BaseOptions
from emperor.experiments import experiment_task_name
from model_runtime.packages import ModelPackage
from models.config_overrides import add_config_override_arguments


class _ExperimentParser(argparse.ArgumentParser):
    pass


def preset_name_to_cli(name: str) -> str:
    return BaseOptions.cli_name(name)


def get_experiment_parser(
    package: ModelPackage,
    config_choices: list | None = None,
) -> _ExperimentParser:
    if not isinstance(package, ModelPackage):
        raise TypeError("Experiment parsing requires a selected ModelPackage.")
    parser = _ExperimentParser(
        description="Run an experiment with a named configuration.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    choices = config_choices or package.preset_type.names()
    cli_config_choices = [preset_name_to_cli(choice) for choice in choices]
    choices_text = "\nAvailable presets:\n" + "\n".join(
        f"  {choice}" for choice in cli_config_choices
    )

    preset_group = parser.add_mutually_exclusive_group(required=True)
    preset_group.add_argument(
        "--preset",
        type=str,
        help="Name of the experiment preset to run." + choices_text,
        choices=cli_config_choices,
        metavar="PRESET_NAME",
    )
    preset_group.add_argument(
        "--presets",
        nargs="+",
        type=str,
        help="Names of experiment presets to run sequentially." + choices_text,
        choices=cli_config_choices,
        metavar="PRESET_NAME",
    )
    preset_group.add_argument(
        "--all-presets",
        action="store_true",
        help="Run all experiment presets sequentially.",
    )
    preset_group.add_argument(
        "--list-config",
        action="store_true",
        help="Print overridable Runtime Defaults flags, then exit.",
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
            "Restrict sweep to named SEARCH_SPACE_* axes.\n"
            "Requires --grid-search or --random-search."
        ),
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Optional experiment namespace inside logs/.",
    )
    parser.add_argument(
        "--datasets",
        "--dataset",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Restrict training to one or more catalogued dataset names.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume one Run from a trusted current-format checkpoint.",
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
        help="Group Runtime Defaults override flags.",
    )

    metadata = package.metadata
    task_choices = [experiment_task_name(task) for task in metadata.experiment_tasks]
    parser.add_argument(
        "--experiment-task",
        type=str,
        default=None,
        choices=task_choices,
        metavar="TASK",
        help=(
            "Experiment Task to run. Defaults to the Model Package default. "
            f"Available tasks: {', '.join(task_choices)}."
        ),
    )
    parser.set_defaults(
        _model_package=package,
        _config_override_dests=add_config_override_arguments(parser, package),
    )
    return parser


__all__ = ["_ExperimentParser", "get_experiment_parser", "preset_name_to_cli"]

import argparse
import importlib

from models.catalog import public_id_for_module
from models.config_overrides import add_config_override_arguments


class _ExperimentParser(argparse.ArgumentParser):
    pass


def preset_name_to_cli(name: str) -> str:
    return name.lower().replace("_", "-")


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

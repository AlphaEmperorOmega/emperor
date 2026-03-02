import argparse
from argparse import Namespace


class _ExperimentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        self.__validate_num_samples(parsed)
        return parsed

    def __validate_num_samples(self, parsed: Namespace) -> None:
        if parsed.num_samples is not None and parsed.run_all:
            self.error("--num-samples can only be used with --name, not --run-all")


def get_experiment_parser(
    config_choices: list | None = None,
) -> _ExperimentParser:
    parser = _ExperimentParser(
        description="Run an experiment with a specific configuration or all available configurations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)

    choices_text = ""
    if config_choices:
        choices_text = "Available configurations:\n" + "\n".join(
            f"  {c}" for c in config_choices
        )

    group.add_argument(
        "--name",
        type=str,
        help="Name of the experiment configuration to run.\n" + choices_text,
        choices=config_choices,
        metavar="CONFIG_NAME",
    )

    group.add_argument(
        "--run-all",
        action="store_true",
        help="Run grid search over all available experiment configurations.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of random search samples. If not provided, grid search is performed over all combinations defined in __base_search_space.",
    )

    parser.add_argument(
        "--log-folder",
        type=str,
        default=None,
        help="Custom folder name for storing experiment logs. If not provided, the model file name is used. Use the same folder across models to compare them in TensorBoard.",
    )

    return parser

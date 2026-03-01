import argparse


def get_experiment_parser(
    config_choices: list | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
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
        help="Run all available experiment configurations.",
    )

    return parser

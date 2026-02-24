import argparse


def get_parser(config_choices: list | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Used to get the config type used for model building."
    )

    parser.add_argument(
        "--config-name",
        type=str,
        help="Name of the config options used to build the model.",
        choices=config_choices,
    )

    return parser

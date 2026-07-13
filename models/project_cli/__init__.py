from __future__ import annotations

from collections.abc import Sequence

from models.project_cli.entrypoint import main


def run_experiment(argv: Sequence[str] | None = None) -> int:
    from models.project_cli.main import run_experiment as implementation

    return implementation(argv)


def run_model_command(
    model_type: str,
    model: str,
    arguments: Sequence[str],
) -> int:
    from models.project_cli.main import run_model_command as implementation

    return implementation(model_type, model, arguments)

__all__ = ["main", "run_experiment", "run_model_command"]

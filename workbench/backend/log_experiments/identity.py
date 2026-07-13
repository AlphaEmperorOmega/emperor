from __future__ import annotations

import re

from workbench.backend.log_experiments.errors import LogExperimentFailure

LOG_EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$")


def is_valid_log_experiment_name(name: str) -> bool:
    return bool(LOG_EXPERIMENT_NAME_RE.fullmatch(name))


def validate_log_experiment_name(name: str) -> str:
    if not name:
        raise LogExperimentFailure("Log experiment folder is required")
    if not is_valid_log_experiment_name(name):
        raise LogExperimentFailure(
            "Log experiment folder must use letters and numbers separated by "
            "single underscores."
        )
    return name


__all__ = [
    "LOG_EXPERIMENT_NAME_RE",
    "is_valid_log_experiment_name",
    "validate_log_experiment_name",
]

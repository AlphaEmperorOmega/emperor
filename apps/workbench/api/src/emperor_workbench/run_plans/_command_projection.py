from __future__ import annotations

import shlex
from typing import Any

from model_runtime.inspection import ConfigurationField
from model_runtime.packages import normalize_key, serialize_config_value
from model_runtime.runs import RunSpec, SearchSpec

from emperor_workbench.model_packages import (
    ModelPackageCatalog,
    ModelPackageFailure,
    ModelPackageIdentity,
    SelectedModelPackage,
)
from emperor_workbench.project_adapter import ModelPackageReference
from emperor_workbench.run_plans._errors import RunPlanFailure
from emperor_workbench.run_plans._records import (
    TrainingCommandsView,
    TrainingRunChangeView,
    TrainingRunView,
)
from emperor_workbench.run_plans._selection import configuration_fields


def render_posix_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in argv)


def render_powershell_command(argv: list[str]) -> str:
    safe_characters = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_@%+=:,./-"
    )

    def quote(value: str) -> str:
        if value and all(character in safe_characters for character in value):
            return value
        return "'" + value.replace("'", "''") + "'"

    return " ".join(quote(value) for value in argv)


def _historical_shell_quote(value: str) -> str:
    if value == "":
        return "''"
    safe = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_@%+=:,./-"
    )
    if all(character in safe for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _command_value(value: Any) -> str:
    serialized = serialize_config_value(value)
    if serialized is None:
        return "None"
    if isinstance(serialized, bool):
        return str(serialized).lower()
    return str(serialized)


def field_label(field: ConfigurationField) -> str:
    return field.key.lower().replace("_", " ")


def project_training_command(
    model_packages: ModelPackageCatalog,
    *,
    model: str,
    preset: str,
    experiment_task: str,
    dataset: str,
    overrides: dict[str, Any],
    log_folder: str,
    monitors: list[str],
) -> tuple[str, list[str], TrainingCommandsView]:
    fields, by_key = configuration_fields(
        model_packages,
        model=model,
        preset=preset,
    )
    values_by_field_key: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        field = by_key.get(normalize_key(str(raw_key)))
        if field is not None:
            values_by_field_key[field.key] = raw_value

    identity = ModelPackageIdentity.from_id(model)
    arguments = [
        "--model-type",
        identity.model_type,
        "--model",
        identity.model,
        "--preset",
        preset,
        "--experiment-task",
        experiment_task,
        "--datasets",
        dataset,
    ]
    if log_folder:
        arguments.extend(["--logdir", log_folder])
    if monitors:
        arguments.append("--monitors")
        arguments.extend(monitors)

    config_parts: list[str] = []
    for field in fields:
        if field.key not in values_by_field_key:
            continue
        config_parts.extend(
            [
                field.flag,
                _command_value(values_by_field_key[field.key]),
            ]
        )
    if config_parts:
        arguments.append("--config")
        arguments.extend(config_parts)
    canonical_argv = ["mise", "run", "experiment", "--", *arguments]
    historical_command = " ".join(
        [
            "source",
            "experiment.sh",
            *(_historical_shell_quote(value) for value in arguments),
        ]
    )
    return (
        historical_command,
        canonical_argv,
        TrainingCommandsView(
            posix=render_posix_command(canonical_argv),
            powershell=render_powershell_command(canonical_argv),
        ),
    )


def run_total_epochs(
    package: ModelPackageReference,
    run: RunSpec,
) -> int:
    try:
        parsed_overrides = (
            SelectedModelPackage(package).parse_overrides(dict(run.overrides)).values
        )
    except ModelPackageFailure as exc:
        raise RunPlanFailure(exc.detail) from exc
    raw_epochs = parsed_overrides.get(
        "num_epochs",
        getattr(package.runtime_defaults, "NUM_EPOCHS", 10),
    )
    try:
        return max(0, int(raw_epochs))
    except (TypeError, ValueError):
        return 0


def project_pending_run(
    model_packages: ModelPackageCatalog,
    *,
    model: str,
    package: ModelPackageReference,
    run: RunSpec,
    index: int,
    log_folder: str,
    monitors: list[str],
    search: SearchSpec | None,
) -> TrainingRunView:
    _fields, by_key = configuration_fields(
        model_packages,
        model=model,
        preset=run.preset,
    )
    search_keys = (
        {normalize_key(axis.key) for axis in (search.axes or ())}
        if search is not None
        else set()
    )
    parameters = list(run.parameters)
    if search is not None:
        search_positions = {
            normalize_key(axis.key): position
            for position, axis in enumerate(search.axes or ())
        }
        fixed_parameters = [
            parameter
            for parameter in parameters
            if normalize_key(parameter.key) not in search_keys
        ]
        searched_parameters = sorted(
            (
                parameter
                for parameter in parameters
                if normalize_key(parameter.key) in search_keys
            ),
            key=lambda parameter: search_positions[normalize_key(parameter.key)],
        )
        parameters = fixed_parameters + searched_parameters
    changes: list[TrainingRunChangeView] = []
    overrides: dict[str, Any] = {}
    for parameter in parameters:
        field = by_key.get(normalize_key(parameter.key))
        field_key = field.key if field is not None else parameter.key
        overrides[field_key] = parameter.value
        changes.append(
            TrainingRunChangeView(
                key=field_key,
                label=field_label(field) if field is not None else field_key,
                value=parameter.value,
                source=(
                    "search"
                    if parameter.source == "search"
                    or normalize_key(parameter.key) in search_keys
                    else "override"
                ),
            )
        )
    command, command_argv, commands = project_training_command(
        model_packages,
        model=model,
        preset=run.preset,
        experiment_task=run.experiment_task,
        dataset=run.dataset,
        overrides=overrides,
        log_folder=log_folder,
        monitors=monitors,
    )
    return TrainingRunView(
        id=run.id or f"run-{index:04d}",
        index=index,
        status="Pending",
        preset=run.preset,
        experiment_task=run.experiment_task,
        dataset=run.dataset,
        changes=changes,
        overrides=overrides,
        command=command,
        command_argv=command_argv,
        commands=commands,
        total_epochs=run_total_epochs(package, run),
    )


__all__ = [
    "field_label",
    "project_pending_run",
    "project_training_command",
    "render_posix_command",
    "render_powershell_command",
    "run_total_epochs",
]

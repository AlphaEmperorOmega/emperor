import random
from collections.abc import Sequence
from pathlib import Path

from model_runtime.packages import (
    GridSearch,
    RandomSearch,
    dataset_name,
    serialize_config_value,
)
from model_runtime.runs import (
    CheckpointContinuation,
    FilesystemRunArtifacts,
    InvalidCheckpointContinuation,
    RunRequest,
    SearchAxisSelection,
    SearchSpec,
    execute_runs,
    plan_runs,
)
from models.catalog import model_package
from models.cli_selection import resolve_cli_selection
from models.experiment_cli_parser import get_experiment_parser


def _selected_preset_names(package, mode) -> tuple[str, ...]:
    if mode.selected_presets is not None:
        presets = mode.selected_presets
    elif mode.preset is not None:
        presets = [mode.preset]
    else:
        presets = list(package.preset_type)
    return tuple(package.preset_name(preset) for preset in presets)


def _search_spec(mode) -> SearchSpec | None:
    if mode.search_mode is None:
        return None
    selections: list[SearchAxisSelection] = []
    positions: dict[str, int] = {}
    if mode.search_keys is not None:
        for key in mode.search_keys:
            positions[key] = len(selections)
            selections.append(SearchAxisSelection(key=key))
    for key, values in mode.search_overrides.items():
        selection = SearchAxisSelection(
            key=key,
            values=tuple(serialize_config_value(value) for value in values),
            allow_custom_values=True,
        )
        position = positions.get(key)
        if position is None:
            positions[key] = len(selections)
            selections.append(selection)
        else:
            selections[position] = selection
    axes = tuple(selections) if selections else None
    if isinstance(mode.search_mode, RandomSearch):
        return SearchSpec(
            mode="random",
            axes=axes,
            random_samples=mode.search_mode.num_samples,
        )
    if isinstance(mode.search_mode, GridSearch):
        return SearchSpec(mode="grid", axes=axes)
    raise ValueError(f"Unsupported search mode: {mode.search_mode!r}")


def _checkpoint_continuation(args) -> CheckpointContinuation | None:
    checkpoint_path = getattr(args, "resume_checkpoint", None)
    if checkpoint_path is None:
        return None
    if (
        getattr(args, "preset", None) is None
        or getattr(args, "presets", None) is not None
        or getattr(args, "all_presets", False)
    ):
        raise InvalidCheckpointContinuation(
            "--resume-checkpoint requires exactly one --preset; "
            "--presets and --all-presets are not supported."
        )
    if len(getattr(args, "datasets", None) or ()) != 1:
        raise InvalidCheckpointContinuation(
            "--resume-checkpoint requires exactly one explicitly selected --dataset."
        )
    if (
        getattr(args, "grid_search", False)
        or getattr(args, "random_search", None) is not None
        or getattr(args, "search_keys", None) is not None
        or bool(getattr(args, "search_set", None))
    ):
        raise InvalidCheckpointContinuation(
            "--resume-checkpoint does not support search flags."
        )
    return CheckpointContinuation(Path(checkpoint_path))


def run_model_package_cli(
    catalog_key: str,
    argv: Sequence[str] | None = None,
) -> int:
    package = model_package(catalog_key)
    if package is None:
        raise ValueError(f"Unknown Model Package: {catalog_key!r}")
    parser = get_experiment_parser(package)
    args = parser.parse_args(argv)
    continuation = _checkpoint_continuation(args)
    selection = resolve_cli_selection(
        args,
        package,
        package.preset_type,
        build_monitor_callbacks=False,
    )
    artifacts = FilesystemRunArtifacts(
        root=Path("logs"),
        namespace=args.logdir,
    )
    dataset_types = package.resolve_datasets(
        args.datasets,
        selection.experiment_task,
    )
    plan = plan_runs(
        package,
        RunRequest(
            presets=_selected_preset_names(package, selection),
            datasets=tuple(dataset_name(dataset) for dataset in dataset_types),
            experiment_task=(package.task_name(selection.experiment_task)),
            overrides={
                key: serialize_config_value(value)
                for key, value in selection.config_overrides.items()
            },
            search=_search_spec(selection),
        ),
        random_source=random if selection.search_mode is not None else None,
    )
    execute_runs(
        package,
        plan,
        artifacts=artifacts,
        monitors=tuple(selection.monitor_names),
        continuation=continuation,
    )
    return 0


__all__ = ["run_model_package_cli"]

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lightning.pytorch.callbacks import Callback

from emperor.monitoring import MonitorSettings
from model_runtime.inspection import InspectionError, parse_overrides
from model_runtime.packages import ModelPackage, dataset_name
from model_runtime.runs.artifacts import FilesystemRunArtifacts
from model_runtime.runs.checkpoints import (
    CheckpointContinuation,
    load_checkpoint_continuation,
    resumed_from_payload,
    validate_model_state,
    validate_target_epochs,
)
from model_runtime.runs.errors import (
    InvalidCheckpointContinuation,
    InvalidRunPlan,
    InvalidRunRequest,
)
from model_runtime.runs.planning import _reject_conflicting_locks
from model_runtime.runs.progress import NeuronClusterGrowthCallback
from model_runtime.runs.records import RunPlan, RunResult


def _invalid_plan(message: str) -> InvalidRunPlan:
    return InvalidRunPlan(message)


def _validated_materialized_runs(
    package: ModelPackage,
    plan: RunPlan,
) -> tuple[Any, list[Any], list[dict[str, Any]]]:
    if not isinstance(package, ModelPackage):
        raise TypeError("Runs require a selected ModelPackage.")
    if plan.identity != package.identity:
        raise _invalid_plan(
            f"Run plan model '{plan.identity.catalog_key}' does not match "
            f"selected model '{package.catalog_key}'."
        )
    if not plan.presets:
        raise _invalid_plan("Run plan requires at least one selected preset.")
    if not plan.datasets:
        raise _invalid_plan("Run plan requires at least one selected dataset.")
    if not plan.runs:
        raise _invalid_plan("Run plan requires at least one training run.")

    try:
        experiment_task = package.resolve_experiment_task(plan.experiment_task)
        selected_presets = [
            package.resolve_preset(preset_name) for preset_name in plan.presets
        ]
        selected_datasets = [
            package.resolve_dataset(dataset, experiment_task)
            for dataset in plan.datasets
        ]
    except ValueError as exc:
        raise _invalid_plan(str(exc)) from exc
    canonical_task = package.task_name(experiment_task)
    canonical_presets = {package.preset_name(preset) for preset in selected_presets}
    canonical_datasets = {dataset_name(dataset) for dataset in selected_datasets}

    materialized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, run in enumerate(plan.runs, start=1):
        if not run.id or run.id in seen_ids:
            detail = "empty" if not run.id else f"duplicate '{run.id}'"
            raise _invalid_plan(f"Run plan contains {detail} run id.")
        seen_ids.add(run.id)
        if run.experiment_task != canonical_task:
            raise _invalid_plan(
                f"Run '{run.id}' experiment task '{run.experiment_task}' does "
                f"not match plan task '{canonical_task}'."
            )
        if run.preset not in canonical_presets:
            raise _invalid_plan(f"Run plan contains unknown preset '{run.preset}'.")
        if run.dataset not in canonical_datasets:
            raise _invalid_plan(f"Run plan contains unknown dataset '{run.dataset}'.")
        if len(run.parameters) != len(run.overrides):
            raise _invalid_plan(
                f"Run '{run.id}' contains duplicate Runtime Defaults assignments."
            )
        try:
            parsed_overrides = parse_overrides(package, run.overrides).values
            _reject_conflicting_locks(package, run.preset, parsed_overrides)
            preset = package.resolve_preset(run.preset)
            dataset = package.resolve_dataset(run.dataset, experiment_task)
        except (InspectionError, InvalidRunRequest, ValueError) as exc:
            raise _invalid_plan(str(exc)) from exc
        materialized.append(
            {
                "id": run.id,
                "index": index,
                "run_total": len(plan.runs),
                "preset": preset,
                "dataset_type": dataset,
                "parameters": dict(run.overrides),
                "config_overrides": dict(parsed_overrides),
            }
        )
    return experiment_task, selected_presets, materialized


def _monitor_callbacks(
    package: ModelPackage,
    plan: RunPlan,
    monitor_names: Sequence[str],
) -> list[Callback]:
    if not monitor_names:
        return []
    try:
        parsed_overrides = parse_overrides(package, plan.overrides).values
        default_interval = getattr(
            package.runtime_defaults,
            "MONITOR_LOG_EVERY_N_STEPS",
            100,
        )
        interval = int(
            parsed_overrides.get(
                "monitor_log_every_n_steps",
                default_interval,
            )
        )
        settings = MonitorSettings(log_every_n_steps=interval)
        return [
            option.build_callback(settings)
            for option in package.resolve_monitors(list(monitor_names))
        ]
    except (InspectionError, ValueError) as exc:
        raise _invalid_plan(str(exc)) from exc


def execute_runs(
    package: ModelPackage,
    plan: RunPlan,
    *,
    artifacts: FilesystemRunArtifacts,
    progress: Callback | None = None,
    monitors: Sequence[str] = (),
    continuation: CheckpointContinuation | None = None,
) -> tuple[RunResult, ...]:
    (
        experiment_task,
        selected_presets,
        materialized_runs,
    ) = _validated_materialized_runs(package, plan)
    if continuation is not None and len(plan.runs) != 1:
        raise InvalidCheckpointContinuation(
            "Checkpoint continuation requires a Run Plan containing exactly one Run."
        )
    loaded_continuation = (
        load_checkpoint_continuation(continuation) if continuation is not None else None
    )

    callbacks: list[Callback] = []
    if progress is not None:
        write_event = getattr(progress, "write_event", None)
        if not callable(write_event):
            raise TypeError("Runs progress callbacks must define write_event(event).")
        callbacks.extend(
            [
                progress,
                NeuronClusterGrowthCallback(write_event),
            ]
        )
    callbacks.extend(_monitor_callbacks(package, plan, monitors))

    experiment = package.build_experiment(
        selected_presets[0],
        experiment_task=experiment_task,
        run_artifacts=artifacts,
    )
    best_results = experiment.load_best_results(artifacts.namespace)

    training_runs = experiment.materialize_training_runs(
        materialized_runs,
        artifacts.namespace,
    )
    if len(training_runs) != len(plan.runs):
        raise _invalid_plan(
            "Run plan materialization produced a different number of Runs: "
            f"expected {len(plan.runs)}, got {len(training_runs)}."
        )
    if loaded_continuation is not None:
        validate_target_epochs(
            loaded_continuation,
            training_runs[0].num_epochs,
        )
        checkpoint_path = loaded_continuation.request.checkpoint_path
        provenance = resumed_from_payload(loaded_continuation)

        def model_validator(model: Any) -> None:
            validate_model_state(loaded_continuation, model)

    else:
        checkpoint_path = None
        provenance = None
        model_validator = None

    results: list[RunResult] = []
    for semantic_run, training_run in zip(plan.runs, training_runs, strict=True):
        payload, log_dir = experiment.execute_training_run(
            training_run,
            log_folder=artifacts.namespace,
            callbacks=callbacks,
            best_results=best_results,
            ckpt_path=checkpoint_path,
            model_validator=model_validator,
            resumed_from=provenance,
        )
        results.append(
            RunResult(
                run_id=semantic_run.id,
                experiment_task=semantic_run.experiment_task,
                preset=semantic_run.preset,
                dataset=semantic_run.dataset,
                log_dir=log_dir,
                payload=payload,
            )
        )
    return tuple(results)


__all__ = ["execute_runs"]

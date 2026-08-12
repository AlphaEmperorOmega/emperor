from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from lightning.pytorch.callbacks import Callback

from emperor.monitoring import MonitorOption, MonitorSettings
from model_runtime.packages import ModelPackage, RuntimeDefaultsError, dataset_name
from model_runtime.runs._handoff import (
    RunExperiment,
    TrainingExecutionRequest,
    TrainingRun,
    TrainingRunRequest,
    require_run_experiment,
)
from model_runtime.runs.artifacts import RunArtifacts
from model_runtime.runs.checkpoints import (
    CheckpointContinuation,
    CheckpointContinuationLifecycle,
    CheckpointExecution,
)
from model_runtime.runs.errors import InvalidRunPlan, InvalidRunRequest
from model_runtime.runs.progress import RunProgress, require_run_progress
from model_runtime.runs.records import PlanningBudget, RunPlan, RunResult, RunSpec


def _invalid_plan(message: str) -> InvalidRunPlan:
    return InvalidRunPlan(message)


def _require_model_package(value: object) -> ModelPackage:
    if not isinstance(value, ModelPackage):
        raise TypeError("Runs require a selected ModelPackage.")
    return value


def _selected_execution_budget(value: object) -> PlanningBudget:
    if value is None:
        return PlanningBudget()
    if not isinstance(value, PlanningBudget):
        raise TypeError("Run execution budget must be a PlanningBudget.")
    return value


@dataclass(frozen=True, slots=True)
class _ResolvedRunPlan:
    experiment_task: Any
    selected_presets: list[Any]
    canonical_task: str
    canonical_presets: set[str]
    canonical_datasets: set[str]


class _RunPlanValidator:
    __slots__ = ("budget", "package", "plan")

    def __init__(
        self,
        package: ModelPackage,
        plan: RunPlan,
        budget: PlanningBudget,
    ) -> None:
        self.package = package
        self.plan = plan
        self.budget = budget

    def validate(self) -> tuple[Any, list[Any], list[TrainingRunRequest]]:
        self._validate_plan_shape()
        resolved = self._resolve_plan()
        materialized = self._materialize_runs(resolved)
        return resolved.experiment_task, resolved.selected_presets, materialized

    def _validate_plan_shape(self) -> None:
        if self.plan.identity != self.package.identity:
            raise _invalid_plan(
                f"Run plan model '{self.plan.identity.catalog_key}' does not match "
                f"selected model '{self.package.catalog_key}'."
            )
        maximum_runs = self.budget.max_materialized_runs
        if maximum_runs is not None and len(self.plan.runs) > maximum_runs:
            raise _invalid_plan(
                f"Run plan contains {len(self.plan.runs)} Runs, exceeding the "
                f"execution maximum of {maximum_runs}."
            )
        if not self.plan.presets:
            raise _invalid_plan("Run plan requires at least one selected preset.")
        if not self.plan.datasets:
            raise _invalid_plan("Run plan requires at least one selected dataset.")
        if not self.plan.runs:
            raise _invalid_plan("Run plan requires at least one training run.")

    def _resolve_plan(self) -> _ResolvedRunPlan:
        try:
            experiment_task = self.package.resolve_experiment_task(
                self.plan.experiment_task
            )
            selected_presets = [
                self.package.resolve_preset(preset_name)
                for preset_name in self.plan.presets
            ]
            selected_datasets = [
                self.package.resolve_dataset(dataset, experiment_task)
                for dataset in self.plan.datasets
            ]
        except ValueError as exc:
            raise _invalid_plan(str(exc)) from exc
        canonical_task = self.package.task_name(experiment_task)
        canonical_presets = {
            self.package.preset_name(preset) for preset in selected_presets
        }
        canonical_datasets = {dataset_name(dataset) for dataset in selected_datasets}
        return _ResolvedRunPlan(
            experiment_task=experiment_task,
            selected_presets=selected_presets,
            canonical_task=canonical_task,
            canonical_presets=canonical_presets,
            canonical_datasets=canonical_datasets,
        )

    def _materialize_runs(
        self,
        resolved: _ResolvedRunPlan,
    ) -> list[TrainingRunRequest]:
        materialized: list[TrainingRunRequest] = []
        seen_ids: set[str] = set()
        for index, run in enumerate(self.plan.runs, start=1):
            self._validate_run_references(run, seen_ids, resolved)
            materialized.append(self._materialized_run(run, index, resolved))
        return materialized

    @staticmethod
    def _validate_run_references(
        run: RunSpec,
        seen_ids: set[str],
        resolved: _ResolvedRunPlan,
    ) -> None:
        if not run.id or run.id in seen_ids:
            detail = "empty" if not run.id else f"duplicate '{run.id}'"
            raise _invalid_plan(f"Run plan contains {detail} run id.")
        seen_ids.add(run.id)
        if run.experiment_task != resolved.canonical_task:
            raise _invalid_plan(
                f"Run '{run.id}' experiment task '{run.experiment_task}' does "
                f"not match plan task '{resolved.canonical_task}'."
            )
        if run.preset not in resolved.canonical_presets:
            raise _invalid_plan(f"Run plan contains unknown preset '{run.preset}'.")
        if run.dataset not in resolved.canonical_datasets:
            raise _invalid_plan(f"Run plan contains unknown dataset '{run.dataset}'.")
        if len(run.parameters) != len(run.overrides):
            raise _invalid_plan(
                f"Run '{run.id}' contains duplicate Runtime Defaults assignments."
            )

    def _materialized_run(
        self,
        run: RunSpec,
        index: int,
        resolved: _ResolvedRunPlan,
    ) -> TrainingRunRequest:
        try:
            runtime_defaults = self.package.runtime_defaults_spec
            parsed_overrides = runtime_defaults.parse_overrides(run.overrides)
            runtime_defaults.reject_conflicting_locked_overrides(
                run.preset,
                parsed_overrides,
            )
            preset = self.package.resolve_preset(run.preset)
            dataset = self.package.resolve_dataset(
                run.dataset,
                resolved.experiment_task,
            )
        except (RuntimeDefaultsError, InvalidRunRequest, ValueError) as exc:
            raise _invalid_plan(str(exc)) from exc
        return TrainingRunRequest(
            run_id=run.id,
            run_index=index,
            run_total=len(self.plan.runs),
            preset=preset,
            dataset_type=dataset,
            parameters=dict(run.overrides),
            config_overrides=dict(parsed_overrides),
        )


def _validated_materialized_runs(
    package: ModelPackage,
    plan: RunPlan,
    budget: PlanningBudget | None = None,
) -> tuple[Any, list[Any], list[TrainingRunRequest]]:
    selected_budget = _selected_execution_budget(budget)
    return _RunPlanValidator(package, plan, selected_budget).validate()


def _resolve_monitor_options(
    package: ModelPackage,
    monitor_names: Sequence[str],
) -> tuple[MonitorOption, ...]:
    if not monitor_names:
        return ()
    try:
        return tuple(package.resolve_monitors(list(monitor_names)))
    except ValueError as exc:
        raise _invalid_plan(str(exc)) from exc


def _monitor_callbacks(
    package: ModelPackage,
    monitor_options: Sequence[MonitorOption],
    run_overrides: Mapping[str, Any],
) -> list[Callback]:
    if not monitor_options:
        return []
    try:
        runtime_defaults = package.runtime_defaults_spec
        default_interval = (
            runtime_defaults.current_value("MONITOR_LOG_EVERY_N_STEPS")
            if "MONITOR_LOG_EVERY_N_STEPS" in runtime_defaults.supported_keys
            else 100
        )
        interval = int(
            run_overrides.get(
                "monitor_log_every_n_steps",
                default_interval,
            )
        )
        settings = MonitorSettings(log_every_n_steps=interval)
        return [option.build_callback(settings) for option in monitor_options]
    except (RuntimeDefaultsError, ValueError) as exc:
        raise _invalid_plan(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class _RunExecutionOptions:
    artifacts: RunArtifacts
    progress: RunProgress | None
    progress_step_interval: int
    monitors: Sequence[str]
    continuation: CheckpointContinuation | None
    budget: PlanningBudget | None


@dataclass(frozen=True, slots=True)
class _PreparedRunExecution:
    experiment: RunExperiment
    training_runs: list[TrainingRun]
    callback_groups: tuple[list[Callback], ...]
    progress: RunProgress | None
    continuation: CheckpointExecution


def _handoff_preset_name(package: ModelPackage, preset: object) -> str:
    try:
        return package.preset_name(preset)
    except (AttributeError, TypeError, ValueError):
        return type(preset).__name__


def _handoff_dataset_name(dataset_type: object) -> str:
    name = getattr(dataset_type, "__name__", None)
    return name if isinstance(name, str) else type(dataset_type).__name__


def _validate_training_run_handoff(
    package: ModelPackage,
    requests: Sequence[TrainingRunRequest],
    training_runs: Sequence[TrainingRun],
) -> None:
    for position, (request, training_run) in enumerate(
        zip(requests, training_runs, strict=True),
        start=1,
    ):
        if not isinstance(cast(object, training_run), TrainingRun):
            raise _invalid_plan(
                f"Run materialization at position {position} expected a "
                f"TrainingRun, got {type(training_run).__name__}."
            )
        if training_run.run_id != request.run_id:
            raise _invalid_plan(
                f"Run materialization at position {position} expected run id "
                f"{request.run_id!r}, got {training_run.run_id!r}."
            )
        if training_run.run_index != request.run_index:
            raise _invalid_plan(
                f"Run materialization at position {position} expected run index "
                f"{request.run_index}, got {training_run.run_index}."
            )
        if training_run.run_total != request.run_total:
            raise _invalid_plan(
                f"Run materialization at position {position} expected run total "
                f"{request.run_total}, got {training_run.run_total}."
            )
        expected_preset_name = _handoff_preset_name(package, request.preset)
        actual_preset_name = _handoff_preset_name(package, training_run.preset)
        if (
            training_run.preset is not request.preset
            or actual_preset_name != expected_preset_name
        ):
            raise _invalid_plan(
                f"Run materialization at position {position} expected preset "
                f"'{expected_preset_name}', got '{actual_preset_name}'."
            )
        if training_run.dataset_type is not request.dataset_type:
            raise _invalid_plan(
                f"Run materialization at position {position} expected Dataset "
                f"'{_handoff_dataset_name(request.dataset_type)}', got "
                f"'{_handoff_dataset_name(training_run.dataset_type)}'."
            )


class _RunExecutor:
    __slots__ = ("options", "package_value", "plan")

    def __init__(
        self,
        package_value: object,
        plan: RunPlan,
        options: _RunExecutionOptions,
    ) -> None:
        self.package_value = package_value
        self.plan = plan
        self.options = options

    def execute(self) -> tuple[RunResult, ...]:
        package = _require_model_package(self.package_value)
        selected_budget = _selected_execution_budget(self.options.budget)
        selected_progress = (
            require_run_progress(self.options.progress)
            if self.options.progress is not None
            else None
        )
        experiment_task, selected_presets, materialized_runs = (
            _validated_materialized_runs(package, self.plan, selected_budget)
        )
        callback_groups = self._callback_groups(package, materialized_runs)
        continuation_lifecycle = CheckpointContinuationLifecycle.admit(
            self.options.continuation,
            self.plan,
        )
        experiment, training_runs = self._materialize_training_runs(
            package,
            experiment_task,
            selected_presets,
            materialized_runs,
        )
        continuation = continuation_lifecycle.bind_training_runs(training_runs)
        prepared = _PreparedRunExecution(
            experiment=experiment,
            training_runs=training_runs,
            callback_groups=callback_groups,
            progress=selected_progress,
            continuation=continuation,
        )
        return self._execute_training_runs(prepared)

    def _callback_groups(
        self,
        package: ModelPackage,
        materialized_runs: list[TrainingRunRequest],
    ) -> tuple[list[Callback], ...]:
        monitor_options = _resolve_monitor_options(package, self.options.monitors)
        if monitor_options:
            return tuple(
                _monitor_callbacks(
                    package,
                    monitor_options,
                    materialized_run.config_overrides,
                )
                for materialized_run in materialized_runs
            )
        return tuple([] for _ in materialized_runs)

    def _materialize_training_runs(
        self,
        package: ModelPackage,
        experiment_task: Any,
        selected_presets: list[Any],
        materialized_runs: list[TrainingRunRequest],
    ) -> tuple[RunExperiment, list[TrainingRun]]:
        experiment = require_run_experiment(
            package.build_experiment(
                selected_presets[0],
                experiment_task=experiment_task,
                run_artifacts=self.options.artifacts,
            ),
            package.catalog_key,
        )
        training_runs = experiment.materialize_training_runs(materialized_runs)
        if len(training_runs) != len(self.plan.runs):
            raise _invalid_plan(
                "Run plan materialization produced a different number of Runs: "
                f"expected {len(self.plan.runs)}, got {len(training_runs)}."
            )
        _validate_training_run_handoff(package, materialized_runs, training_runs)
        return experiment, training_runs

    def _execute_training_runs(
        self,
        prepared: _PreparedRunExecution,
    ) -> tuple[RunResult, ...]:
        results: list[RunResult] = []
        for semantic_run, training_run, callbacks in zip(
            self.plan.runs,
            prepared.training_runs,
            prepared.callback_groups,
            strict=True,
        ):
            payload, log_dir = prepared.experiment.execute_training(
                TrainingExecutionRequest(
                    training_run=training_run,
                    callbacks=callbacks,
                    progress=prepared.progress,
                    progress_step_interval=self.options.progress_step_interval,
                    ckpt_path=prepared.continuation.checkpoint_path,
                    model_validator=prepared.continuation.model_validator,
                    resumed_from=prepared.continuation.provenance,
                )
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


def execute_runs(
    package: ModelPackage,
    plan: RunPlan,
    *,
    artifacts: RunArtifacts,
    progress: RunProgress | None = None,
    progress_step_interval: int = 1,
    monitors: Sequence[str] = (),
    continuation: CheckpointContinuation | None = None,
    budget: PlanningBudget | None = None,
) -> tuple[RunResult, ...]:
    options = _RunExecutionOptions(
        artifacts=artifacts,
        progress=progress,
        progress_step_interval=progress_step_interval,
        monitors=monitors,
        continuation=continuation,
        budget=budget,
    )
    return _RunExecutor(package, plan, options).execute()


__all__ = ["execute_runs"]

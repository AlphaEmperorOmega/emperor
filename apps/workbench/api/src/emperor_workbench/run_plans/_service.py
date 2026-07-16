from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from model_runtime.runs import (
    PlanningBudget,
    RunPlan,
    SubmittedRun,
)

from emperor_workbench.config_snapshots import ConfigSnapshotService
from emperor_workbench.failures import FailureKind
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.project_adapter import (
    ProjectAdapterFailure,
)
from emperor_workbench.run_plans._command_projection import (
    project_pending_run,
)
from emperor_workbench.run_plans._errors import RunPlanFailure
from emperor_workbench.run_plans._limits import (
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_SEARCH_AXES,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)
from emperor_workbench.run_plans._progress_projection import (
    RunPlanProgressProjector,
)
from emperor_workbench.run_plans._records import (
    CreateTrainingRunPlanCommand,
    MaterializedTrainingRunPlan,
    MaterializeTrainingRunPlanCommand,
    SubmittedTrainingRun,
    SubmittedTrainingRunPlan,
    TrainingRunPlanView,
    TrainingRunView,
)
from emperor_workbench.run_plans._search import search_from_spec
from emperor_workbench.run_plans._selection import (
    SelectedTrainingInputs,
    require_package,
    resolve_inputs,
    resolve_monitor_names,
    resolve_presets,
)
from emperor_workbench.run_plans._snapshot_resolution import SnapshotResolver


class RunPlanService:
    """Create, accept, and materialize exact typed Run Plans."""

    def __init__(
        self,
        *,
        model_packages: ModelPackageCatalog,
        config_snapshots: ConfigSnapshotService | None = None,
        random_source: random.Random | None = None,
    ) -> None:
        self._model_packages = model_packages
        self._snapshots = SnapshotResolver(config_snapshots)
        self._random = random_source or random

    def preview(
        self,
        command: CreateTrainingRunPlanCommand,
    ) -> TrainingRunPlanView:
        if command.snapshot_ids:
            return self._create_snapshot_plan(
                model=command.model,
                preset=command.preset,
                presets=command.presets,
                experiment_task=command.experiment_task,
                datasets=command.datasets,
                overrides=command.overrides,
                log_folder=command.log_folder,
                monitors=command.monitors,
                search=command.search,
                snapshot_ids=command.snapshot_ids,
            )
        selected = resolve_inputs(
            self._model_packages,
            model=command.model,
            preset=command.preset,
            presets=command.presets,
            experiment_task=command.experiment_task,
            datasets=command.datasets,
            overrides=command.overrides,
            search=command.search,
        )
        monitor_names = resolve_monitor_names(selected.package, command.monitors)
        return self._create(
            model=command.model,
            selected=selected,
            log_folder=command.log_folder,
            monitors=monitor_names,
        )

    def materialize(
        self,
        command: MaterializeTrainingRunPlanCommand,
        *,
        validated_log_folder: str,
    ) -> MaterializedTrainingRunPlan:
        if command.snapshot_ids:
            if command.submitted_plan is not None:
                raise RunPlanFailure(
                    "Snapshot Training Jobs do not accept a submitted exact Run Plan.",
                    kind=FailureKind.CONFLICT,
                )
            self._snapshots.require_current_revisions(
                model=command.model,
                snapshot_ids=command.snapshot_ids,
                submitted_revisions=command.snapshot_revisions,
            )
            plan = self._create_snapshot_plan(
                model=command.model,
                preset=command.preset,
                presets=command.presets,
                experiment_task=command.experiment_task,
                datasets=command.datasets,
                overrides=command.overrides,
                log_folder=validated_log_folder,
                monitors=command.monitors,
                search=command.search,
                snapshot_ids=command.snapshot_ids,
            )
            package = require_package(self._model_packages, command.model)
            monitor_names = resolve_monitor_names(package, command.monitors)
            return MaterializedTrainingRunPlan(
                plan=plan,
                monitors=tuple(monitor_names),
            )

        submitted_plan = command.submitted_plan
        snapshot_records = ()
        if submitted_plan is not None:
            submitted_plan, snapshot_records = self._snapshots.reconcile_submitted(
                model=command.model,
                plan=submitted_plan,
                envelope_snapshot_ids=command.snapshot_ids,
                envelope_overrides=command.overrides,
            )
        selected_presets = command.presets
        if snapshot_records:
            regular_presets = (
                list(command.presets)
                if command.presets is not None
                else [command.preset]
            )
            selected_presets = [
                *regular_presets,
                *(record.preset for record in snapshot_records),
            ]
        selected = resolve_inputs(
            self._model_packages,
            model=command.model,
            preset=command.preset,
            presets=selected_presets,
            experiment_task=command.experiment_task,
            datasets=command.datasets,
            overrides=command.overrides,
            search=command.search,
        )
        monitor_names = resolve_monitor_names(
            selected.package,
            command.monitors,
        )
        plan = (
            self._from_submitted(
                model=command.model,
                selected=selected,
                run_plan=submitted_plan,
                log_folder=validated_log_folder,
                monitors=monitor_names,
                envelope_snapshot_ids=command.snapshot_ids,
            )
            if submitted_plan is not None
            else self._create(
                model=command.model,
                selected=selected,
                log_folder=validated_log_folder,
                monitors=monitor_names,
            )
        )
        return MaterializedTrainingRunPlan(
            plan=plan,
            monitors=tuple(monitor_names),
        )

    def _create_snapshot_plan(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None,
        experiment_task: str | None,
        datasets: list[str],
        overrides: dict[str, Any],
        log_folder: str,
        monitors: list[str] | None,
        search: object,
        snapshot_ids: list[str],
    ) -> TrainingRunPlanView:
        if search is not None:
            raise RunPlanFailure(
                "Config Snapshot Run Plans cannot be combined with a search."
            )
        records = self._snapshots.resolve_records(snapshot_ids, model=model)
        regular_presets = list(presets) if presets is not None else [preset]
        combined_presets = [
            *regular_presets,
            *(record.preset for record in records),
        ]
        selected = resolve_inputs(
            self._model_packages,
            model=model,
            preset=preset,
            presets=combined_presets,
            experiment_task=experiment_task,
            datasets=datasets,
            overrides={},
            search=None,
        )
        canonical_regular_presets = (
            resolve_presets(
                selected.package,
                model=model,
                preset=preset,
                presets=regular_presets,
            )
            if regular_presets
            else []
        )
        submitted_runs: list[SubmittedTrainingRun] = []
        for regular_preset in canonical_regular_presets:
            for dataset in selected.request.datasets:
                index = len(submitted_runs) + 1
                submitted_runs.append(
                    SubmittedTrainingRun(
                        id=f"preset-{regular_preset}-{dataset}-{index}",
                        preset=regular_preset,
                        dataset=dataset,
                        overrides=dict(overrides),
                    )
                )
        for record in records:
            for dataset in selected.request.datasets:
                index = len(submitted_runs) + 1
                submitted_runs.append(
                    SubmittedTrainingRun(
                        id=f"snapshot-{record.id}-{dataset}-{index}",
                        preset=record.preset,
                        dataset=dataset,
                        overrides={**dict(record.overrides), **overrides},
                        snapshot_id=record.id,
                        snapshot_name=record.name,
                    )
                )
        monitor_names = resolve_monitor_names(selected.package, monitors)
        plan = SubmittedTrainingRunPlan(
            runs=submitted_runs,
            snapshot_revisions=self._snapshots.revisions(records),
        )
        return self._from_submitted(
            model=model,
            selected=selected,
            run_plan=plan,
            log_folder=log_folder,
            monitors=monitor_names,
            snapshot_overlay_overrides=overrides,
        )

    def _create(
        self,
        *,
        model: str,
        selected: SelectedTrainingInputs,
        log_folder: str,
        monitors: list[str] | None = None,
    ) -> TrainingRunPlanView:
        monitor_names = monitors or []
        try:
            semantic_plan = selected.package.client.plan_runs(
                selected.package.catalog_key,
                selected.request,
                random_source=(
                    self._random
                    if selected.request.search is not None
                    and selected.request.search.mode == "random"
                    else None
                ),
                budget=_planning_budget(),
            )
        except ProjectAdapterFailure as exc:
            raise RunPlanFailure(exc.detail, kind=exc.kind) from exc
        runs = [
            project_pending_run(
                self._model_packages,
                model=model,
                package=selected.package,
                run=run,
                index=index,
                log_folder=log_folder,
                monitors=monitor_names,
                search=semantic_plan.search,
            )
            for index, run in enumerate(semantic_plan.runs, start=1)
        ]
        return self._plan_view(
            model=model,
            semantic_plan=semantic_plan,
            log_folder=log_folder,
            runs=runs,
        )

    def _from_submitted(
        self,
        *,
        model: str,
        selected: SelectedTrainingInputs,
        run_plan: SubmittedTrainingRunPlan,
        log_folder: str,
        monitors: list[str] | None = None,
        envelope_snapshot_ids: list[str] | None = None,
        snapshot_overlay_overrides: Mapping[str, Any] | None = None,
    ) -> TrainingRunPlanView:
        monitor_names = monitors or []
        submitted_plan, _snapshot_records = self._snapshots.reconcile_submitted(
            model=model,
            plan=run_plan,
            envelope_snapshot_ids=envelope_snapshot_ids,
            envelope_overrides=(
                snapshot_overlay_overrides
                if snapshot_overlay_overrides is not None
                else selected.request.overrides
            ),
        )
        submitted_runs = submitted_plan.runs
        if len(submitted_runs) > MAX_TRAINING_PLANNED_RUNS:
            raise RunPlanFailure(
                "Submitted run plan is too large: "
                f"{len(submitted_runs)} submitted runs exceeds "
                f"{MAX_TRAINING_PLANNED_RUNS}."
            )
        try:
            semantic_plan = selected.package.client.accept_run_plan(
                selected.package.catalog_key,
                selected.request,
                tuple(
                    SubmittedRun(
                        id=row.id or None,
                        preset=row.preset,
                        dataset=row.dataset,
                        overrides=dict(row.overrides),
                    )
                    for row in submitted_runs
                ),
                budget=_planning_budget(),
            )
        except ProjectAdapterFailure as exc:
            raise RunPlanFailure(exc.detail, kind=exc.kind) from exc

        runs = []
        for index, (row, semantic_run) in enumerate(
            zip(submitted_runs, semantic_plan.runs, strict=True),
            start=1,
        ):
            projected_row = project_pending_run(
                self._model_packages,
                model=model,
                package=selected.package,
                run=semantic_run,
                index=index,
                log_folder=log_folder,
                monitors=monitor_names,
                search=semantic_plan.search,
            )
            runs.append(
                replace(
                    projected_row,
                    snapshot_id=row.snapshot_id,
                    snapshot_name=row.snapshot_name,
                    snapshot_id_present=True,
                    snapshot_name_present=True,
                )
            )

        return self._plan_view(
            model=model,
            semantic_plan=semantic_plan,
            log_folder=log_folder,
            runs=runs,
            snapshot_revisions=submitted_plan.snapshot_revisions,
        )

    @staticmethod
    def _plan_view(
        *,
        model: str,
        semantic_plan: RunPlan,
        log_folder: str,
        runs: list[TrainingRunView],
        snapshot_revisions=(),
    ) -> TrainingRunPlanView:
        return TrainingRunPlanView(
            model=model,
            preset=semantic_plan.presets[0],
            presets=list(semantic_plan.presets),
            experiment_task=semantic_plan.experiment_task,
            datasets=list(semantic_plan.datasets),
            overrides=dict(semantic_plan.overrides),
            search=search_from_spec(semantic_plan.search),
            log_folder=log_folder,
            is_random_search=bool(
                semantic_plan.search and semantic_plan.search.mode == "random"
            ),
            runs=runs,
            summary=RunPlanProgressProjector.summarize(runs),
            snapshot_revisions=snapshot_revisions,
        )


def _planning_budget() -> PlanningBudget:
    return PlanningBudget(
        max_axes=MAX_TRAINING_SEARCH_AXES,
        max_values_per_axis=MAX_TRAINING_SEARCH_AXIS_VALUES,
        max_materialized_runs=MAX_TRAINING_PLANNED_RUNS,
    )


__all__ = ["RunPlanService"]

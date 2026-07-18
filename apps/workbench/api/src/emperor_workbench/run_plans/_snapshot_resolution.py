from __future__ import annotations

from collections.abc import Mapping

from emperor_workbench.config_snapshots import (
    ConfigSnapshotRecord,
    ConfigSnapshotService,
)
from emperor_workbench.failures import FailureKind
from emperor_workbench.run_plans._errors import RunPlanFailure
from emperor_workbench.run_plans._records import (
    ConfigSnapshotRevision,
    SubmittedTrainingRun,
    SubmittedTrainingRunPlan,
)


class SnapshotResolver:
    def __init__(self, snapshots: ConfigSnapshotService | None) -> None:
        self._snapshots = snapshots

    def resolve_records(
        self,
        snapshot_ids: list[str] | tuple[str, ...],
        *,
        model: str,
        failure_kind: FailureKind = FailureKind.INVALID,
    ) -> tuple[ConfigSnapshotRecord, ...]:
        if not snapshot_ids:
            return ()
        if self._snapshots is None:
            raise RunPlanFailure(
                "Config Snapshot Run Plans are not configured.",
                kind=failure_kind,
            )
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise RunPlanFailure(
                "Config Snapshot Run Plan contains duplicate snapshot ids.",
                kind=failure_kind,
            )
        records: list[ConfigSnapshotRecord] = []
        for snapshot_id in snapshot_ids:
            snapshot = self._snapshots.get_snapshot(snapshot_id)
            if snapshot is None:
                raise RunPlanFailure(
                    f"Config snapshot '{snapshot_id}' no longer exists.",
                    kind=failure_kind,
                )
            if snapshot.model != model:
                raise RunPlanFailure(
                    f"Config snapshot '{snapshot_id}' belongs to Model Package "
                    f"'{snapshot.model}', not '{model}'.",
                    kind=failure_kind,
                )
            records.append(snapshot)
        return tuple(records)

    def revisions(
        self,
        records: tuple[ConfigSnapshotRecord, ...],
    ) -> tuple[ConfigSnapshotRevision, ...]:
        if not records:
            return ()
        assert self._snapshots is not None
        return tuple(
            ConfigSnapshotRevision(
                id=record.id,
                semantic_revision=self._snapshots.semantic_revision(record),
            )
            for record in records
        )

    def reconcile_submitted(
        self,
        *,
        model: str,
        plan: SubmittedTrainingRunPlan,
        envelope_snapshot_ids: list[str] | None = None,
        envelope_overrides: Mapping[str, object] | None = None,
    ) -> tuple[SubmittedTrainingRunPlan, tuple[ConfigSnapshotRecord, ...]]:
        run_snapshot_ids = [
            run.snapshot_id for run in plan.runs if run.snapshot_id is not None
        ]
        revision_ids = [revision.id for revision in plan.snapshot_revisions]
        if not run_snapshot_ids and not revision_ids:
            return plan, ()
        if not plan.snapshot_revisions:
            raise RunPlanFailure(
                "Snapshot provenance requires a backend-issued semantic revision.",
                kind=FailureKind.CONFLICT,
            )
        if set(run_snapshot_ids) != set(revision_ids):
            raise RunPlanFailure(
                "Submitted Snapshot provenance does not match its revision set.",
                kind=FailureKind.CONFLICT,
            )
        if envelope_snapshot_ids and set(envelope_snapshot_ids) != set(revision_ids):
            raise RunPlanFailure(
                "Submitted snapshotIds do not match the Run Plan revision set.",
                kind=FailureKind.CONFLICT,
            )
        records = self.resolve_records(
            revision_ids,
            model=model,
            failure_kind=FailureKind.CONFLICT,
        )
        current_revisions = self.revisions(records)
        submitted_by_id = {
            revision.id: revision.semantic_revision
            for revision in plan.snapshot_revisions
        }
        for revision in current_revisions:
            if submitted_by_id.get(revision.id) != revision.semantic_revision:
                raise RunPlanFailure(
                    f"Config snapshot '{revision.id}' changed semantically. "
                    "Refresh the Run Plan before starting Training.",
                    kind=FailureKind.CONFLICT,
                )
        records_by_id = {record.id: record for record in records}
        reconciled_runs: list[SubmittedTrainingRun] = []
        for run in plan.runs:
            if run.snapshot_id is None:
                reconciled_runs.append(run)
                continue
            record = records_by_id[run.snapshot_id]
            reconciled_runs.append(
                SubmittedTrainingRun(
                    id=run.id,
                    preset=record.preset,
                    dataset=run.dataset,
                    overrides={
                        **dict(record.overrides),
                        **dict(envelope_overrides or {}),
                    },
                    snapshot_id=record.id,
                    snapshot_name=record.name,
                )
            )
        return (
            SubmittedTrainingRunPlan(
                runs=reconciled_runs,
                snapshot_revisions=current_revisions,
            ),
            records,
        )

    def require_current_revisions(
        self,
        *,
        model: str,
        snapshot_ids: list[str],
        submitted_revisions: tuple[ConfigSnapshotRevision, ...],
    ) -> tuple[ConfigSnapshotRecord, ...]:
        records = self.resolve_records(
            snapshot_ids,
            model=model,
            failure_kind=FailureKind.CONFLICT,
        )
        current_revisions = self.revisions(records)
        submitted_by_id = {
            revision.id: revision.semantic_revision for revision in submitted_revisions
        }
        if set(submitted_by_id) != set(snapshot_ids):
            raise RunPlanFailure(
                "Snapshot Training Jobs require backend-issued semantic "
                "revisions for every snapshotId.",
                kind=FailureKind.CONFLICT,
            )
        for revision in current_revisions:
            if submitted_by_id.get(revision.id) != revision.semantic_revision:
                raise RunPlanFailure(
                    f"Config snapshot '{revision.id}' changed semantically. "
                    "Refresh the Run Plan before starting Training.",
                    kind=FailureKind.CONFLICT,
                )
        return records


__all__ = ["SnapshotResolver"]

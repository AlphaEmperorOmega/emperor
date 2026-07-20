"""Per-group scheduler reconciliation for dynamic Neuron optimizers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer

from emperor.neuron._optimizer_checkpoint import LegacyOptimizerAppendPolicy

_PER_GROUP_SEQUENCE_NAMES = (
    "base_lrs",
    "_last_lr",
    "lr_lambdas",
    "min_lrs",
    "max_lrs",
    "base_momentums",
    "max_momentums",
)
_SUPPORTED_TORCH_SCHEDULERS = {
    "ChainedScheduler",
    "ConstantLR",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "ExponentialLR",
    "LambdaLR",
    "LinearLR",
    "MultiStepLR",
    "MultiplicativeLR",
    "OneCycleLR",
    "PolynomialLR",
    "ReduceLROnPlateau",
    "SequentialLR",
    "StepLR",
}


@dataclass(frozen=True)
class SchedulerGroupLoadBinding:
    scheduler: object
    saved_state: dict[str, Any] | None
    optimizer: Optimizer
    policy: LegacyOptimizerAppendPolicy | None
    target_group_count: int


@dataclass(frozen=True)
class _ListSnapshot:
    original_list: list[Any]
    original_values: tuple[Any, ...]

    def restore(self) -> None:
        self.original_list[:] = self.original_values


@dataclass(frozen=True)
class _NamespaceSnapshot:
    namespace: dict[str, Any]
    original_items: tuple[tuple[str, Any], ...]
    list_snapshots: tuple[_ListSnapshot, ...]

    def restore(self) -> None:
        for list_snapshot in reversed(self.list_snapshots):
            list_snapshot.restore()
        self.namespace.clear()
        self.namespace.update(self.original_items)


@dataclass
class _SchedulerMigration:
    optimizer: Optimizer
    live_snapshots: tuple[_NamespaceSnapshot, ...]
    payload_snapshots: tuple[_NamespaceSnapshot, ...]
    optimizer_loaded: bool = False


class NeuronSchedulerMutationTransaction:
    """Make a set of live scheduler mutations atomic."""

    def __init__(self) -> None:
        self._snapshots: list[_NamespaceSnapshot] = []

    def prepare(self, schedulers: list[object]) -> None:
        self.clear()
        snapshot_keys: set[int] = set()
        for scheduler in schedulers:
            _snapshot_live_scheduler_namespaces(
                scheduler,
                self._snapshots,
                snapshot_keys,
            )

    def commit(self) -> None:
        self._snapshots.clear()

    def clear(self) -> None:
        for namespace_snapshot in reversed(self._snapshots):
            namespace_snapshot.restore()
        self._snapshots.clear()


class NeuronSchedulerCheckpointReconciler:
    """Own temporary scheduler edits until checkpoint restoration succeeds."""

    def __init__(self) -> None:
        self._migrations: list[_SchedulerMigration] = []

    def prepare_for_load(
        self,
        bindings: list[SchedulerGroupLoadBinding],
    ) -> None:
        self.clear()
        for binding in bindings:
            if binding.policy is None:
                continue
            _validate_scheduler_group_count(
                binding.scheduler,
                binding.saved_state,
                binding.policy,
                binding.target_group_count,
            )

        live_snapshots_by_optimizer: dict[int, list[_NamespaceSnapshot]] = {}
        payload_snapshots_by_optimizer: dict[int, list[_NamespaceSnapshot]] = {}
        optimizers_by_id: dict[int, Optimizer] = {}
        live_snapshot_keys_by_optimizer: dict[int, set[int]] = {}
        payload_snapshot_keys_by_optimizer: dict[int, set[int]] = {}
        for binding in bindings:
            optimizer_id = id(binding.optimizer)
            optimizers_by_id[optimizer_id] = binding.optimizer
            live_snapshots = live_snapshots_by_optimizer.setdefault(
                optimizer_id,
                [],
            )
            live_snapshot_keys = live_snapshot_keys_by_optimizer.setdefault(
                optimizer_id,
                set(),
            )
            payload_snapshots = payload_snapshots_by_optimizer.setdefault(
                optimizer_id,
                [],
            )
            payload_snapshot_keys = payload_snapshot_keys_by_optimizer.setdefault(
                optimizer_id,
                set(),
            )
            _snapshot_live_scheduler_namespaces(
                binding.scheduler,
                live_snapshots,
                live_snapshot_keys,
            )
            _snapshot_saved_scheduler_namespaces(
                binding.scheduler,
                binding.saved_state,
                payload_snapshots,
                payload_snapshot_keys,
            )
        self._migrations = [
            _SchedulerMigration(
                optimizer=optimizers_by_id[optimizer_id],
                live_snapshots=tuple(live_snapshots_by_optimizer[optimizer_id]),
                payload_snapshots=tuple(payload_snapshots_by_optimizer[optimizer_id]),
            )
            for optimizer_id in live_snapshots_by_optimizer
        ]
        try:
            for binding in bindings:
                if binding.policy is None:
                    continue
                _apply_scheduler_group_count(
                    binding.scheduler,
                    binding.saved_state,
                    binding.policy,
                    binding.target_group_count,
                )
        except BaseException:
            self.clear()
            raise

    def optimizer_requires_completion(self, optimizer: Optimizer) -> bool:
        return any(migration.optimizer is optimizer for migration in self._migrations)

    def mark_optimizer_loaded(self, optimizer: Optimizer) -> None:
        for migration in self._migrations:
            if migration.optimizer is optimizer:
                migration.optimizer_loaded = True

    def commit_loaded(self) -> None:
        self._migrations = [
            migration
            for migration in self._migrations
            if not migration.optimizer_loaded
        ]

    def clear(self) -> None:
        for migration in reversed(self._migrations):
            for payload_snapshot in reversed(migration.payload_snapshots):
                payload_snapshot.restore()
            for live_snapshot in reversed(migration.live_snapshots):
                live_snapshot.restore()
        self._migrations.clear()


def reconcile_scheduler_group_count(
    scheduler: object,
    saved_state: dict[str, Any] | None,
    policy: LegacyOptimizerAppendPolicy,
    target_group_count: int,
) -> None:
    """Copy the historical reference group's scheduler settings to suffix groups."""

    _validate_scheduler_group_count(
        scheduler,
        saved_state,
        policy,
        target_group_count,
    )
    _apply_scheduler_group_count(
        scheduler,
        saved_state,
        policy,
        target_group_count,
    )


def preflight_scheduler_group_extension(
    scheduler: object,
    *,
    previous_group_count: int,
    reference_group_index: int,
) -> None:
    policy = LegacyOptimizerAppendPolicy(
        base_group_count=previous_group_count,
        reference_group_index=reference_group_index,
    )
    _validate_scheduler_group_count(
        scheduler,
        None,
        policy,
        previous_group_count + 1,
    )


def extend_scheduler_for_new_group(
    scheduler: object,
    *,
    previous_group_count: int,
    reference_group_index: int,
) -> None:
    policy = LegacyOptimizerAppendPolicy(
        base_group_count=previous_group_count,
        reference_group_index=reference_group_index,
    )
    reconcile_scheduler_group_count(
        scheduler,
        None,
        policy,
        previous_group_count + 1,
    )


def preflight_scheduler_group_removal(
    scheduler: object,
    removed_group_indices: tuple[int, ...],
    *,
    previous_group_count: int,
) -> None:
    _validate_scheduler_group_removal(
        scheduler,
        removed_group_indices,
        previous_group_count=previous_group_count,
    )


def remove_scheduler_groups(
    scheduler: object,
    removed_group_indices: tuple[int, ...],
    *,
    previous_group_count: int,
) -> None:
    if not removed_group_indices:
        return
    _validate_scheduler_group_removal(
        scheduler,
        removed_group_indices,
        previous_group_count=previous_group_count,
    )
    _apply_scheduler_group_removal(
        scheduler,
        removed_group_indices,
        previous_group_count=previous_group_count,
    )


def _validate_scheduler_group_count(
    scheduler: object,
    saved_state: dict[str, Any] | None,
    policy: LegacyOptimizerAppendPolicy,
    target_group_count: int,
) -> None:
    _validate_supported_scheduler(scheduler)
    if (
        (
            target_group_count < policy.base_group_count
            and not policy.group_reference_indices
        )
        or not 0 <= policy.reference_group_index < policy.base_group_count
        or (
            policy.group_reference_indices
            and (
                len(policy.group_reference_indices) != target_group_count
                or any(
                    isinstance(reference_index, bool)
                    or not isinstance(reference_index, int)
                    or not 0 <= reference_index < policy.base_group_count
                    for reference_index in policy.group_reference_indices
                )
            )
        )
    ):
        raise RuntimeError("Invalid Neuron scheduler group reconciliation policy.")
    allowed_lengths = {policy.base_group_count, target_group_count}
    _validate_sequence_lengths(
        scheduler.__dict__,
        allowed_lengths,
        operation="reconcile a legacy Neuron optimizer scheduler",
    )
    if saved_state is not None:
        _validate_sequence_lengths(
            saved_state,
            allowed_lengths,
            operation="reconcile a saved legacy Neuron optimizer scheduler",
        )
    for child_scheduler, child_state in _scheduler_children(
        scheduler,
        saved_state,
    ):
        _validate_scheduler_group_count(
            child_scheduler,
            child_state,
            policy,
            target_group_count,
        )


def _apply_scheduler_group_count(
    scheduler: object,
    saved_state: dict[str, Any] | None,
    policy: LegacyOptimizerAppendPolicy,
    target_group_count: int,
) -> None:
    for sequence_name in _PER_GROUP_SEQUENCE_NAMES:
        _reconcile_live_sequence(
            scheduler.__dict__,
            sequence_name,
            policy=policy,
            target_group_count=target_group_count,
            preserve_reference=sequence_name == "lr_lambdas",
            saved_state=saved_state,
        )
        if saved_state is not None:
            _extend_sequence(
                saved_state,
                sequence_name,
                base_group_count=policy.base_group_count,
                reference_group_index=policy.reference_group_index,
                target_group_count=target_group_count,
                preserve_reference=False,
            )
    for child_scheduler, child_state in _scheduler_children(
        scheduler,
        saved_state,
    ):
        _apply_scheduler_group_count(
            child_scheduler,
            child_state,
            policy,
            target_group_count,
        )


def _validate_scheduler_group_removal(
    scheduler: object,
    removed_group_indices: tuple[int, ...],
    *,
    previous_group_count: int,
) -> None:
    _validate_supported_scheduler(scheduler)
    if len(set(removed_group_indices)) != len(removed_group_indices) or any(
        index < 0 or index >= previous_group_count for index in removed_group_indices
    ):
        raise RuntimeError("Invalid Neuron scheduler group-removal indices.")
    remaining_group_count = previous_group_count - len(removed_group_indices)
    _validate_sequence_lengths(
        scheduler.__dict__,
        {previous_group_count, remaining_group_count},
        operation="synchronize a pruned Neuron optimizer scheduler",
    )
    for child_scheduler, _ in _scheduler_children(scheduler, None):
        _validate_scheduler_group_removal(
            child_scheduler,
            removed_group_indices,
            previous_group_count=previous_group_count,
        )


def _apply_scheduler_group_removal(
    scheduler: object,
    removed_group_indices: tuple[int, ...],
    *,
    previous_group_count: int,
) -> None:
    remaining_group_count = previous_group_count - len(removed_group_indices)
    for sequence_name in _PER_GROUP_SEQUENCE_NAMES:
        scheduler_group_values = scheduler.__dict__.get(sequence_name)
        if (
            not isinstance(scheduler_group_values, list)
            or len(scheduler_group_values) == remaining_group_count
        ):
            continue
        for group_index in sorted(removed_group_indices, reverse=True):
            del scheduler_group_values[group_index]
    for child_scheduler, _ in _scheduler_children(scheduler, None):
        _apply_scheduler_group_removal(
            child_scheduler,
            removed_group_indices,
            previous_group_count=previous_group_count,
        )


def _validate_sequence_lengths(
    namespace: dict[str, Any],
    allowed_lengths: set[int],
    *,
    operation: str,
) -> None:
    for sequence_name in _PER_GROUP_SEQUENCE_NAMES:
        sequence_values = namespace.get(sequence_name)
        if (
            isinstance(sequence_values, list)
            and len(sequence_values) not in allowed_lengths
        ):
            expected_lengths = " or ".join(
                str(length) for length in sorted(allowed_lengths)
            )
            raise RuntimeError(
                f"Cannot safely {operation}: {sequence_name} has "
                f"{len(sequence_values)} entries, expected {expected_lengths}."
            )


def _scheduler_children(
    scheduler: object,
    saved_state: dict[str, Any] | None,
) -> list[tuple[object, dict[str, Any] | None]]:
    child_schedulers = getattr(scheduler, "_schedulers", None)
    if not isinstance(child_schedulers, list):
        return []
    saved_child_states = (
        saved_state.get("_schedulers") if saved_state is not None else None
    )
    if saved_child_states is not None and (
        not isinstance(saved_child_states, list)
        or len(saved_child_states) != len(child_schedulers)
    ):
        raise RuntimeError(
            "Cannot safely reconcile nested Neuron optimizer schedulers: "
            "live and saved child counts differ."
        )
    aligned_saved_child_states = (
        saved_child_states
        if isinstance(saved_child_states, list)
        else [None] * len(child_schedulers)
    )
    return list(zip(child_schedulers, aligned_saved_child_states, strict=True))


def _snapshot_live_scheduler_namespaces(
    scheduler: object,
    snapshots: list[_NamespaceSnapshot],
    snapshot_keys: set[int],
) -> None:
    _snapshot_object_namespace(scheduler, snapshots, snapshot_keys)
    for namespace_value in scheduler.__dict__.values():
        callable_values = (
            namespace_value if isinstance(namespace_value, list) else [namespace_value]
        )
        for callable_value in callable_values:
            if callable(callable_value):
                _snapshot_object_namespace(
                    callable_value,
                    snapshots,
                    snapshot_keys,
                )
    for child_scheduler, _ in _scheduler_children(scheduler, None):
        _snapshot_live_scheduler_namespaces(
            child_scheduler,
            snapshots,
            snapshot_keys,
        )


def _snapshot_object_namespace(
    value: object,
    snapshots: list[_NamespaceSnapshot],
    snapshot_keys: set[int],
) -> None:
    namespace = getattr(value, "__dict__", None)
    if not isinstance(namespace, dict) or id(namespace) in snapshot_keys:
        return
    _snapshot_namespace_dict(namespace, snapshots, snapshot_keys)


def _snapshot_namespace_dict(
    namespace: dict[str, Any],
    snapshots: list[_NamespaceSnapshot],
    snapshot_keys: set[int],
) -> None:
    if id(namespace) in snapshot_keys:
        return
    snapshot_keys.add(id(namespace))
    snapshots.append(
        _NamespaceSnapshot(
            namespace=namespace,
            original_items=tuple(namespace.items()),
            list_snapshots=tuple(
                _ListSnapshot(
                    original_list=list_value,
                    original_values=tuple(list_value),
                )
                for list_value in namespace.values()
                if isinstance(list_value, list)
            ),
        )
    )


def _snapshot_saved_scheduler_namespaces(
    scheduler: object,
    saved_state: dict[str, Any] | None,
    snapshots: list[_NamespaceSnapshot],
    snapshot_keys: set[int],
) -> None:
    if saved_state is not None:
        _snapshot_namespace_dict(saved_state, snapshots, snapshot_keys)
    for child_scheduler, child_state in _scheduler_children(
        scheduler,
        saved_state,
    ):
        _snapshot_saved_scheduler_namespaces(
            child_scheduler,
            child_state,
            snapshots,
            snapshot_keys,
        )


def _validate_supported_scheduler(scheduler: object) -> None:
    scheduler_type = type(scheduler)
    if (
        scheduler_type.__module__ != "torch.optim.lr_scheduler"
        or scheduler_type.__name__ not in _SUPPORTED_TORCH_SCHEDULERS
    ):
        raise RuntimeError(
            "Cannot safely change Neuron optimizer group cardinality for "
            f"unrecognized scheduler {scheduler_type.__module__}."
            f"{scheduler_type.__qualname__}."
        )


def _extend_sequence(
    namespace: dict[str, Any],
    sequence_name: str,
    *,
    base_group_count: int,
    reference_group_index: int,
    target_group_count: int,
    preserve_reference: bool,
) -> None:
    sequence_values = namespace.get(sequence_name)
    if (
        not isinstance(sequence_values, list)
        or len(sequence_values) == target_group_count
    ):
        return
    reference_value = sequence_values[reference_group_index]
    sequence_values.extend(
        reference_value if preserve_reference else deepcopy(reference_value)
        for _ in range(target_group_count - base_group_count)
    )


def _reconcile_live_sequence(
    namespace: dict[str, Any],
    sequence_name: str,
    *,
    policy: LegacyOptimizerAppendPolicy,
    target_group_count: int,
    preserve_reference: bool,
    saved_state: dict[str, Any] | None,
) -> None:
    if not policy.group_reference_indices:
        _extend_sequence(
            namespace,
            sequence_name,
            base_group_count=policy.base_group_count,
            reference_group_index=policy.reference_group_index,
            target_group_count=target_group_count,
            preserve_reference=preserve_reference,
        )
        return

    live_sequence_values = namespace.get(sequence_name)
    if not isinstance(live_sequence_values, list):
        return
    base_sequence_values = tuple(live_sequence_values[: policy.base_group_count])
    reconciled_sequence_values = [
        (
            base_sequence_values[reference_index]
            if preserve_reference
            else deepcopy(base_sequence_values[reference_index])
        )
        for reference_index in policy.group_reference_indices
    ]
    saved_sequence_values = (
        saved_state.get(sequence_name) if saved_state is not None else None
    )
    if live_sequence_values is saved_sequence_values:
        namespace[sequence_name] = reconciled_sequence_values
    else:
        live_sequence_values[:] = reconciled_sequence_values

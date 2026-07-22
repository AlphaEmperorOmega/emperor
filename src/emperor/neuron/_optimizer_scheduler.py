"""Atomic scheduler synchronization for dynamic Neuron optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer

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
            "Cannot safely restore nested Neuron optimizer schedulers: "
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

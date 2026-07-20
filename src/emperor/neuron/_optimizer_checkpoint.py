"""Optimizer-layout compatibility for historical dynamic-neuron checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class LegacyOptimizerAppendPolicy:
    base_group_count: int
    reference_group_index: int
    group_reference_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _OptimizerGroupSnapshot:
    group: dict[str, Any]
    original_values: dict[str, Any]
    original_lists: dict[str, tuple[list[Any], tuple[Any, ...]]]

    @classmethod
    def capture(cls, group: dict[str, Any]) -> _OptimizerGroupSnapshot:
        return cls(
            group=group,
            original_values=dict(group),
            original_lists={
                name: (value, tuple(value))
                for name, value in group.items()
                if isinstance(value, list)
            },
        )

    def restore(self) -> None:
        self.group.clear()
        for name, value in self.original_values.items():
            list_snapshot = self.original_lists.get(name)
            if list_snapshot is None:
                self.group[name] = value
                continue
            original_list, original_items = list_snapshot
            original_list[:] = original_items
            self.group[name] = original_list


@dataclass
class _OptimizerLoadSnapshot:
    optimizer: Optimizer
    original_param_groups: list[dict[str, Any]]
    groups: tuple[_OptimizerGroupSnapshot, ...]
    original_state: dict[Any, Any]
    original_state_items: tuple[tuple[Any, Any], ...]
    optimizer_loaded: bool = False

    @classmethod
    def capture(cls, optimizer: Optimizer) -> _OptimizerLoadSnapshot:
        return cls(
            optimizer=optimizer,
            original_param_groups=optimizer.param_groups,
            groups=tuple(
                _OptimizerGroupSnapshot.capture(group)
                for group in optimizer.param_groups
            ),
            original_state=optimizer.state,
            original_state_items=tuple(optimizer.state.items()),
        )

    def restore(self) -> None:
        for group in self.groups:
            group.restore()
        self.original_param_groups[:] = [group.group for group in self.groups]
        self.optimizer.param_groups = self.original_param_groups
        self.original_state.clear()
        self.original_state.update(self.original_state_items)
        self.optimizer.state = self.original_state


class NeuronOptimizerLoadTransaction:
    """Retain the complete pre-load optimizer state until fit can start."""

    def __init__(self) -> None:
        self._snapshots: list[_OptimizerLoadSnapshot] = []

    def prepare_for_load(self, optimizers: list[Optimizer]) -> None:
        self.clear()
        self._snapshots = [
            _OptimizerLoadSnapshot.capture(optimizer) for optimizer in optimizers
        ]

    def optimizer_requires_completion(self, optimizer: Optimizer) -> bool:
        return any(snapshot.optimizer is optimizer for snapshot in self._snapshots)

    def mark_optimizer_loaded(self, optimizer: Optimizer) -> None:
        for snapshot in self._snapshots:
            if snapshot.optimizer is optimizer:
                snapshot.optimizer_loaded = True

    def commit_loaded(self) -> None:
        if self._snapshots and not all(
            snapshot.optimizer_loaded for snapshot in self._snapshots
        ):
            raise RuntimeError(
                "Cannot commit a partial Neuron optimizer checkpoint load."
            )
        self._snapshots.clear()

    def commit(self) -> None:
        """Keep current optimizer mutations and discard rollback snapshots."""

        self._snapshots.clear()

    def clear(self) -> None:
        for snapshot in reversed(self._snapshots):
            snapshot.restore()
        self._snapshots.clear()


@dataclass(frozen=True)
class _LegacyOptimizerMigration:
    optimizer: Optimizer
    base_group_count: int
    original_group_parameters: tuple[
        tuple[list[nn.Parameter], tuple[nn.Parameter, ...]],
        ...,
    ]
    original_group_parameter_names: tuple[
        tuple[list[str], tuple[str, ...]] | None,
        ...,
    ]
    base_alias_parameter_ids: frozenset[int]
    append_policy: LegacyOptimizerAppendPolicy


@dataclass(frozen=True)
class _RootParameterTopology:
    current_parameters: tuple[nn.Parameter, ...]
    base_parameters: tuple[nn.Parameter, ...]
    base_occurrence_ids: frozenset[int]


class NeuronOptimizerCheckpointReconciler:
    """Reconstruct and retain historical groups for exact legacy continuation."""

    def __init__(self) -> None:
        self._migrations: list[_LegacyOptimizerMigration] = []

    def prepare_for_load(
        self,
        optimizers: list[Optimizer],
        clusters: list[nn.Module],
        saved_optimizer_states: list[dict[str, Any]],
        *,
        root_module: nn.Module | None = None,
    ) -> None:
        self.clear()
        if len(optimizers) != len(saved_optimizer_states):
            return
        initial_role_owner_ids = self.__initial_role_owner_ids(
            optimizers,
            clusters,
        )
        root_parameter_topology = self.__root_parameter_topology(
            root_module,
            clusters,
        )
        try:
            for optimizer, saved_state in zip(
                optimizers,
                saved_optimizer_states,
                strict=True,
            ):
                self.__prepare_optimizer(
                    optimizer,
                    clusters,
                    saved_state,
                    initial_role_owner_ids,
                    root_parameter_topology,
                )
            if (
                sum(
                    bool(migration.base_alias_parameter_ids)
                    for migration in self._migrations
                )
                > 1
            ):
                raise RuntimeError(
                    "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                    "unnamed dynamic aliases occur across multiple optimizers."
                )
        except BaseException:
            self.clear()
            raise

    def optimizer_requires_completion(self, optimizer: Optimizer) -> bool:
        return any(migration.optimizer is optimizer for migration in self._migrations)

    def pending_append_policy(
        self,
        optimizer: Optimizer,
    ) -> LegacyOptimizerAppendPolicy | None:
        return next(
            (
                migration.append_policy
                for migration in self._migrations
                if migration.optimizer is optimizer
            ),
            None,
        )

    def complete_optimizer_load(
        self,
        optimizer: Optimizer,
    ) -> LegacyOptimizerAppendPolicy | None:
        append_policy = None
        remaining_migrations = []
        for migration in self._migrations:
            if migration.optimizer is optimizer:
                append_policy = migration.append_policy
            else:
                remaining_migrations.append(migration)
        self._migrations = remaining_migrations
        return append_policy

    def clear(self) -> None:
        for migration in self._migrations:
            self.__restore_original_groups(migration)
        self._migrations.clear()

    def __prepare_optimizer(
        self,
        optimizer: Optimizer,
        clusters: list[nn.Module],
        saved_state: dict[str, Any],
        initial_role_owner_ids: dict[tuple[int, str], set[int]],
        root_parameter_topology: _RootParameterTopology | None,
    ) -> None:
        saved_groups = saved_state.get("param_groups")
        if not isinstance(saved_groups, list):
            return
        base_group_count = len(optimizer.param_groups)
        if len(saved_groups) <= base_group_count:
            return
        saved_extra_groups = saved_groups[base_group_count:]
        saved_extra_parameter_names = tuple(
            self.__validated_group_param_names(saved_group)
            for saved_group in saved_extra_groups
        )
        saved_dynamic_count = sum(len(group["params"]) for group in saved_extra_groups)

        owned_dynamic_clusters = self.__owned_dynamic_clusters(
            optimizer,
            clusters,
            initial_role_owner_ids,
            saved_dynamic_count,
            root_parameter_topology,
        )
        if len(owned_dynamic_clusters) != 1:
            raise RuntimeError(
                "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                "appended parameter groups require exactly one optimizer-owned "
                "NeuronCluster with dynamic neurons."
            )
        (
            cluster,
            owning_group_indices,
            dynamic_parameters,
            base_alias_parameter_ids,
        ) = owned_dynamic_clusters[0]
        reference_group_index = min(owning_group_indices)
        original_group_parameters = tuple(
            (group["params"], tuple(group["params"]))
            for group in optimizer.param_groups
        )
        validated_group_parameter_names = tuple(
            self.__validated_group_param_names(group)
            for group in optimizer.param_groups
        )
        original_group_parameter_names = tuple(
            (
                (group["param_names"], parameter_names)
                if parameter_names is not None
                else None
            )
            for group, parameter_names in zip(
                optimizer.param_groups,
                validated_group_parameter_names,
                strict=True,
            )
        )
        dynamic_parameter_ids = {id(parameter) for parameter in dynamic_parameters}
        stripped_group_parameters = [
            [
                parameter
                for parameter in group["params"]
                if id(parameter) not in dynamic_parameter_ids
            ]
            for group in optimizer.param_groups
        ]
        stripped_group_parameter_names = [
            (
                [
                    name
                    for name, parameter in zip(
                        parameter_names,
                        group["params"],
                        strict=True,
                    )
                    if id(parameter) not in dynamic_parameter_ids
                ]
                if parameter_names is not None
                else None
            )
            for group, parameter_names in zip(
                optimizer.param_groups,
                validated_group_parameter_names,
                strict=True,
            )
        ]
        if root_parameter_topology is not None and base_alias_parameter_ids:
            stripped_group_parameters = self.__restore_base_alias_order(
                optimizer,
                stripped_group_parameters,
                base_alias_parameter_ids,
                root_parameter_topology,
            )
            reordered_parameter_names: list[list[str] | None] = []
            for group, parameter_names, stripped_parameters in zip(
                optimizer.param_groups,
                validated_group_parameter_names,
                stripped_group_parameters,
                strict=True,
            ):
                if parameter_names is None:
                    reordered_parameter_names.append(None)
                    continue
                parameter_names_by_id = {
                    id(parameter): name
                    for name, parameter in zip(
                        parameter_names,
                        group["params"],
                        strict=True,
                    )
                }
                reordered_parameter_names.append(
                    [
                        parameter_names_by_id[id(parameter)]
                        for parameter in stripped_parameters
                    ]
                )
            stripped_group_parameter_names = reordered_parameter_names
        saved_base_group_lengths = [
            len(group["params"]) for group in saved_groups[:base_group_count]
        ]
        live_base_group_lengths = [
            len(parameters) for parameters in stripped_group_parameters
        ]
        if live_base_group_lengths != saved_base_group_lengths:
            raise RuntimeError(
                "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                "the configured base parameter-group sizes do not match the "
                f"checkpoint ({live_base_group_lengths} != "
                f"{saved_base_group_lengths})."
            )

        reference_saved_group = saved_groups[reference_group_index]
        if any(
            not self.__group_options_equal(group, reference_saved_group)
            for group in saved_extra_groups
        ):
            raise RuntimeError(
                "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                "an appended group's optimizer options diverge from its historical "
                "reference group."
            )

        append_policy = LegacyOptimizerAppendPolicy(
            base_group_count=base_group_count,
            reference_group_index=reference_group_index,
            group_reference_indices=(
                tuple(range(base_group_count))
                + (reference_group_index,) * len(saved_extra_groups)
            ),
        )
        self._migrations.append(
            _LegacyOptimizerMigration(
                optimizer=optimizer,
                base_group_count=base_group_count,
                original_group_parameters=original_group_parameters,
                original_group_parameter_names=original_group_parameter_names,
                base_alias_parameter_ids=base_alias_parameter_ids,
                append_policy=append_policy,
            )
        )

        for group, parameters, parameter_names in zip(
            optimizer.param_groups,
            stripped_group_parameters,
            stripped_group_parameter_names,
            strict=True,
        ):
            group["params"] = parameters
            if parameter_names is not None:
                group["param_names"] = parameter_names

        reference_group = optimizer.param_groups[reference_group_index]
        dynamic_offset = 0
        for saved_group, saved_parameter_names in zip(
            saved_extra_groups,
            saved_extra_parameter_names,
            strict=True,
        ):
            group_size = len(saved_group["params"])
            group_parameters = list(
                dynamic_parameters[dynamic_offset : dynamic_offset + group_size]
            )
            dynamic_offset += group_size
            group_options = {
                name: value
                for name, value in reference_group.items()
                if name not in {"params", "param_names"}
            }
            new_group = {**group_options, "params": group_parameters}
            if saved_parameter_names is not None:
                new_group["param_names"] = list(saved_parameter_names)
            optimizer.add_param_group(new_group)

    def __owned_dynamic_clusters(
        self,
        optimizer: Optimizer,
        clusters: list[nn.Module],
        initial_role_owner_ids: dict[tuple[int, str], set[int]],
        saved_dynamic_count: int,
        root_parameter_topology: _RootParameterTopology | None,
    ) -> list[tuple[nn.Module, list[int], tuple[nn.Parameter, ...], frozenset[int]]]:
        all_initial_parameter_ids = {
            parameter_id
            for cluster in clusters
            for parameter_id in self.__initial_parameter_ids(cluster)
        }
        cluster_candidates = []
        claimed_candidate_parameter_ids: set[int] = set()
        retained_base_alias_parameter_ids: set[int] = set()
        for cluster in clusters:
            raw_parameter_candidates, fixed_base_alias_parameter_ids = (
                self.__owned_dynamic_parameter_candidates(
                    optimizer,
                    cluster,
                    initial_role_owner_ids,
                    all_initial_parameter_ids,
                    root_parameter_topology,
                )
            )
            retained_base_alias_parameter_ids.update(fixed_base_alias_parameter_ids)
            cluster_initial_parameter_ids = self.__initial_parameter_ids(cluster)
            owning_group_indices = [
                index
                for index, group in enumerate(optimizer.param_groups)
                if any(
                    id(parameter) in cluster_initial_parameter_ids
                    for parameter in group["params"]
                )
            ]
            if not owning_group_indices:
                continue
            parameter_candidates = tuple(
                (parameter, has_base_occurrence)
                for parameter, has_base_occurrence in raw_parameter_candidates
                if id(parameter) not in claimed_candidate_parameter_ids
            )
            claimed_candidate_parameter_ids.update(
                id(parameter) for parameter, _ in parameter_candidates
            )
            retained_base_alias_parameter_ids.update(
                id(parameter)
                for parameter, has_base_occurrence in parameter_candidates
                if has_base_occurrence
            )
            if parameter_candidates and owning_group_indices:
                cluster_candidates.append(
                    (cluster, owning_group_indices, parameter_candidates)
                )

        eligible_suffix_owners = []
        for candidate_index, cluster_candidate in enumerate(cluster_candidates):
            _, _, parameter_candidates = cluster_candidate
            exclusive_parameter_count = sum(
                not has_base_occurrence
                for _, has_base_occurrence in parameter_candidates
            )
            total_parameter_count = len(parameter_candidates)
            another_cluster_has_exclusive_parameters = any(
                other_index != candidate_index
                and any(
                    not has_base_occurrence
                    for _, has_base_occurrence in other_parameter_candidates
                )
                for other_index, (_, _, other_parameter_candidates) in enumerate(
                    cluster_candidates
                )
            )
            if not another_cluster_has_exclusive_parameters and saved_dynamic_count in {
                exclusive_parameter_count,
                total_parameter_count,
            }:
                eligible_suffix_owners.append(cluster_candidate)

        if len(eligible_suffix_owners) != 1:
            if len(cluster_candidates) == 1:
                _, _, parameter_candidates = cluster_candidates[0]
                exclusive_parameter_count = sum(
                    not has_base_occurrence
                    for _, has_base_occurrence in parameter_candidates
                )
                total_parameter_count = len(parameter_candidates)
                if (
                    exclusive_parameter_count
                    < saved_dynamic_count
                    < total_parameter_count
                ):
                    raise RuntimeError(
                        "Cannot safely reconcile a legacy Neuron optimizer "
                        "checkpoint: an unnamed dynamic parameter alias makes "
                        "historical base/suffix membership ambiguous."
                    )
                raise RuntimeError(
                    "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                    "the appended parameter count does not match the reconstructed "
                    f"dynamic topology ({saved_dynamic_count} not in "
                    f"{{{exclusive_parameter_count}, {total_parameter_count}}})."
                )
            return []
        cluster, owning_group_indices, parameter_candidates = eligible_suffix_owners[0]
        exclusive_parameter_count = sum(
            not has_base_occurrence for _, has_base_occurrence in parameter_candidates
        )
        include_base_aliases = saved_dynamic_count > exclusive_parameter_count
        selected_base_alias_parameter_ids = {
            id(parameter)
            for parameter, has_base_occurrence in parameter_candidates
            if has_base_occurrence
        }
        if include_base_aliases and (
            retained_base_alias_parameter_ids - selected_base_alias_parameter_ids
        ):
            raise RuntimeError(
                "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                "unnamed dynamic alias membership across clusters is ambiguous."
            )
        dynamic_parameters = tuple(
            parameter
            for parameter, has_base_occurrence in parameter_candidates
            if not has_base_occurrence or include_base_aliases
        )
        if include_base_aliases:
            retained_base_alias_parameter_ids.difference_update(
                selected_base_alias_parameter_ids
            )
        return [
            (
                cluster,
                owning_group_indices,
                dynamic_parameters,
                frozenset(retained_base_alias_parameter_ids),
            )
        ]

    def __restore_base_alias_order(
        self,
        optimizer: Optimizer,
        stripped_group_parameters: list[list[nn.Parameter]],
        base_alias_parameter_ids: frozenset[int],
        root_parameter_topology: _RootParameterTopology,
    ) -> list[list[nn.Parameter]]:
        alias_group_count = sum(
            bool(
                base_alias_parameter_ids.intersection(
                    id(parameter) for parameter in group["params"]
                )
            )
            for group in optimizer.param_groups
        )
        if alias_group_count > 1:
            raise RuntimeError(
                "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                "unnamed dynamic aliases occur in multiple base parameter groups."
            )
        current_root_parameters = root_parameter_topology.current_parameters
        base_root_parameters = root_parameter_topology.base_parameters
        restored_group_parameters = []
        for group, stripped_parameters in zip(
            optimizer.param_groups,
            stripped_group_parameters,
            strict=True,
        ):
            original_parameters = group["params"]
            original_parameter_ids = [
                id(parameter) for parameter in original_parameters
            ]
            if not base_alias_parameter_ids.intersection(original_parameter_ids):
                restored_group_parameters.append(stripped_parameters)
                continue
            if len(original_parameter_ids) != len(set(original_parameter_ids)):
                raise RuntimeError(
                    "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                    "a base alias occurs in a group with duplicate parameters."
                )
            stripped_parameter_ids = {
                id(parameter) for parameter in stripped_parameters
            }
            restored_parameters = [
                parameter
                for parameter in base_root_parameters
                if id(parameter) in stripped_parameter_ids
            ]
            if (
                len(restored_parameters) != len(stripped_parameters)
                or {id(parameter) for parameter in restored_parameters}
                != stripped_parameter_ids
            ):
                raise RuntimeError(
                    "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                    "the root module cannot reconstruct an aliased base group."
                )
            if [id(parameter) for parameter in stripped_parameters] == [
                id(parameter) for parameter in restored_parameters
            ]:
                restored_group_parameters.append(stripped_parameters)
                continue
            original_parameter_id_set = set(original_parameter_ids)
            canonical_current_ids = [
                id(parameter)
                for parameter in current_root_parameters
                if id(parameter) in original_parameter_id_set
            ]
            if canonical_current_ids != original_parameter_ids:
                raise RuntimeError(
                    "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                    "an unnamed base alias uses a custom optimizer parameter order."
                )
            restored_group_parameters.append(restored_parameters)
        return restored_group_parameters

    def __root_parameter_topology(
        self,
        root_module: nn.Module | None,
        clusters: list[nn.Module],
    ) -> _RootParameterTopology | None:
        if root_module is None:
            return None
        root_module_paths = list(root_module.named_modules(remove_duplicate=False))
        dynamic_neuron_prefixes = []
        for cluster in clusters:
            initial_neuron_names = self.__initial_neuron_names(cluster)
            cluster_paths = [
                module_name
                for module_name, module in root_module_paths
                if module is cluster
            ]
            for cluster_path in cluster_paths:
                for neuron_name in cluster.cluster:
                    if neuron_name in initial_neuron_names:
                        continue
                    dynamic_neuron_prefixes.append(
                        ".".join(
                            part
                            for part in (cluster_path, "cluster", neuron_name)
                            if part
                        )
                    )

        named_parameter_occurrences = list(
            root_module.named_parameters(remove_duplicate=False)
        )

        def is_dynamic_occurrence(name: str) -> bool:
            return any(
                name.startswith(f"{prefix}.") for prefix in dynamic_neuron_prefixes
            )

        def unique_parameters(parameter_occurrences) -> tuple[nn.Parameter, ...]:
            unique_parameters_in_order = []
            seen_parameter_ids: set[int] = set()
            for _, parameter in parameter_occurrences:
                if id(parameter) in seen_parameter_ids:
                    continue
                unique_parameters_in_order.append(parameter)
                seen_parameter_ids.add(id(parameter))
            return tuple(unique_parameters_in_order)

        base_parameter_occurrences = [
            (name, parameter)
            for name, parameter in named_parameter_occurrences
            if not is_dynamic_occurrence(name)
        ]
        base_parameters = unique_parameters(base_parameter_occurrences)
        return _RootParameterTopology(
            current_parameters=unique_parameters(named_parameter_occurrences),
            base_parameters=base_parameters,
            base_occurrence_ids=frozenset(
                id(parameter) for parameter in base_parameters
            ),
        )

    def __initial_role_owner_ids(
        self,
        optimizers: list[Optimizer],
        clusters: list[nn.Module],
    ) -> dict[tuple[int, str], set[int]]:
        optimizer_ids_by_parameter_id: dict[int, set[int]] = {}
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    optimizer_ids_by_parameter_id.setdefault(id(parameter), set()).add(
                        id(optimizer)
                    )

        role_owner_ids: dict[tuple[int, str], set[int]] = {}
        for cluster in clusters:
            initial_neuron_names = self.__initial_neuron_names(cluster)
            for name, parameter in cluster.named_parameters(remove_duplicate=False):
                neuron_parameter = self.__neuron_parameter(name)
                if (
                    neuron_parameter is None
                    or neuron_parameter[0] not in initial_neuron_names
                ):
                    continue
                role = neuron_parameter[1]
                role_owner_ids.setdefault((id(cluster), role), set()).update(
                    optimizer_ids_by_parameter_id.get(id(parameter), ())
                )
        return role_owner_ids

    def __initial_parameter_ids(self, cluster: nn.Module) -> set[int]:
        return {
            id(parameter)
            for neuron_name in self.__initial_neuron_names(cluster)
            if neuron_name in cluster.cluster
            for parameter in cluster.cluster[neuron_name].parameters()
        }

    def __owned_dynamic_parameter_candidates(
        self,
        optimizer: Optimizer,
        cluster: nn.Module,
        initial_role_owner_ids: dict[tuple[int, str], set[int]],
        initial_parameter_ids: set[int],
        root_parameter_topology: _RootParameterTopology | None,
    ) -> tuple[tuple[tuple[nn.Parameter, bool], ...], frozenset[int]]:
        initial_neuron_names = self.__initial_neuron_names(cluster)
        candidate_parameters: list[tuple[nn.Parameter, bool]] = []
        candidate_parameter_ids: set[int] = set()
        base_alias_parameter_ids: set[int] = set()
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        base_occurrence_ids = (
            root_parameter_topology.base_occurrence_ids
            if root_parameter_topology is not None
            else frozenset()
        )
        expected_owner_ids = {id(optimizer)}
        for name, parameter in cluster.named_parameters(remove_duplicate=False):
            neuron_parameter = self.__neuron_parameter(name)
            if neuron_parameter is None or neuron_parameter[0] in initial_neuron_names:
                continue
            parameter_id = id(parameter)
            role = neuron_parameter[1]
            has_base_occurrence = parameter_id in base_occurrence_ids
            if parameter_id in initial_parameter_ids and (
                root_parameter_topology is None or has_base_occurrence
            ):
                if parameter_id in optimizer_parameter_ids:
                    base_alias_parameter_ids.add(parameter_id)
                continue
            role_is_owned = (
                initial_role_owner_ids.get((id(cluster), role), set())
                == expected_owner_ids
            )
            if not role_is_owned:
                if has_base_occurrence and parameter_id in optimizer_parameter_ids:
                    base_alias_parameter_ids.add(parameter_id)
                continue
            if parameter_id in candidate_parameter_ids:
                continue
            candidate_parameters.append((parameter, has_base_occurrence))
            candidate_parameter_ids.add(parameter_id)
        return tuple(candidate_parameters), frozenset(base_alias_parameter_ids)

    @staticmethod
    def __neuron_parameter(name: str) -> tuple[str, str] | None:
        parameter_name_parts = name.split(".", 2)
        if (
            len(parameter_name_parts) != 3
            or parameter_name_parts[0] != "cluster"
            or not parameter_name_parts[1].startswith("neuron_")
        ):
            return None
        return parameter_name_parts[1], parameter_name_parts[2]

    @staticmethod
    def __initial_neuron_names(cluster: nn.Module) -> set[str]:
        x_coordinates = range(
            cluster.initial_x_axis_start,
            cluster.initial_x_axis_start + cluster.initial_x_axis_total_neurons,
        )
        y_coordinates = range(
            cluster.initial_y_axis_start,
            cluster.initial_y_axis_start + cluster.initial_y_axis_total_neurons,
        )
        z_coordinates = range(
            cluster.initial_z_axis_start,
            cluster.initial_z_axis_start + cluster.initial_z_axis_total_neurons,
        )
        return {
            cluster._neuron_name(x, y, z)
            for x, y, z in product(x_coordinates, y_coordinates, z_coordinates)
        }

    @classmethod
    def __group_options_equal(cls, left: dict, right: dict) -> bool:
        ignored_names = {"params", "param_names"}
        left_options = {
            name: value for name, value in left.items() if name not in ignored_names
        }
        right_options = {
            name: value for name, value in right.items() if name not in ignored_names
        }
        if left_options.keys() != right_options.keys():
            return False
        return all(
            cls.__option_value_equal(left_options[name], right_options[name])
            for name in left_options
        )

    @classmethod
    def __option_value_equal(cls, left: Any, right: Any) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return (
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and torch.equal(left, right)
            )
        if isinstance(left, dict) and isinstance(right, dict):
            return cls.__group_options_equal(left, right)
        if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
            return len(left) == len(right) and all(
                cls.__option_value_equal(left_value, right_value)
                for left_value, right_value in zip(left, right, strict=True)
            )
        return bool(left == right)

    @staticmethod
    def __validated_group_param_names(
        group: dict,
    ) -> tuple[str, ...] | None:
        parameter_names = group.get("param_names")
        if parameter_names is None:
            return None
        if (
            not isinstance(parameter_names, list)
            or len(parameter_names) != len(group["params"])
            or not all(isinstance(name, str) for name in parameter_names)
        ):
            raise RuntimeError(
                "Cannot safely reconcile a legacy Neuron optimizer checkpoint: "
                "param_names are not aligned with params."
            )
        return tuple(parameter_names)

    @staticmethod
    def __restore_original_groups(migration: _LegacyOptimizerMigration) -> None:
        optimizer = migration.optimizer
        if len(optimizer.param_groups) < migration.base_group_count:
            return
        optimizer.param_groups[:] = optimizer.param_groups[: migration.base_group_count]
        for group, parameter_snapshot, parameter_name_snapshot in zip(
            optimizer.param_groups,
            migration.original_group_parameters,
            migration.original_group_parameter_names,
            strict=True,
        ):
            parameter_list, parameters = parameter_snapshot
            parameter_list[:] = parameters
            group["params"] = parameter_list
            if parameter_name_snapshot is not None:
                parameter_name_list, parameter_names = parameter_name_snapshot
                parameter_name_list[:] = parameter_names
                group["param_names"] = parameter_name_list

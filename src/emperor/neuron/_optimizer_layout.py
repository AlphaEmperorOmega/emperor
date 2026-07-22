"""Name-aware optimizer layout metadata for Neuron checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn
from torch.optim import Optimizer

OPTIMIZER_LAYOUT_CHECKPOINT_KEY = "emperor_neuron_optimizer_layout"
_OPTIMIZER_LAYOUT_VERSION = 1
_ROLE_SYNC_POLICY = "role"


@dataclass(frozen=True)
class _NamedOptimizerMigration:
    optimizer: Optimizer
    original_groups: tuple[dict, ...]
    saved_state: dict[str, Any]
    original_saved_parameter_ids: tuple[tuple[Any, ...], ...]
    original_saved_parameter_names: tuple[tuple[Any, ...] | None, ...]


class NeuronOptimizerNamedLayout:
    """Reconstruct saved optimizer groups from stable parameter names."""

    def __init__(self) -> None:
        self._migrations: list[_NamedOptimizerMigration] = []

    @classmethod
    def capture(
        cls,
        module: nn.Module,
        optimizers: list[Optimizer],
        saved_optimizer_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(optimizers) != len(saved_optimizer_states):
            raise RuntimeError(
                "Cannot save Neuron optimizer layout metadata: the live and "
                "serialized optimizer counts differ."
            )
        parameter_names_by_id = cls.__parameter_names_by_identity(module)
        optimizer_layouts = []
        for optimizer, saved_state in zip(
            optimizers,
            saved_optimizer_states,
            strict=True,
        ):
            saved_groups = saved_state.get("param_groups")
            if not isinstance(saved_groups, list) or len(saved_groups) != len(
                optimizer.param_groups
            ):
                raise RuntimeError(
                    "Cannot save Neuron optimizer layout metadata: the live and "
                    "serialized parameter-group counts differ."
                )
            parameter_names_by_group = []
            optimizer_parameter_names = []
            for live_group, saved_group in zip(
                optimizer.param_groups,
                saved_groups,
                strict=True,
            ):
                live_parameters = live_group["params"]
                saved_parameter_ids = saved_group.get("params")
                if not isinstance(saved_parameter_ids, list) or len(
                    live_parameters
                ) != len(saved_parameter_ids):
                    raise RuntimeError(
                        "Cannot save Neuron optimizer layout metadata: a live and "
                        "serialized parameter-group size differs."
                    )
                for parameter_group, layout_source in (
                    (live_group, "live"),
                    (saved_group, "serialized"),
                ):
                    group_parameter_names = parameter_group.get("param_names")
                    if group_parameter_names is not None and (
                        not isinstance(group_parameter_names, list)
                        or len(group_parameter_names) != len(live_parameters)
                    ):
                        raise RuntimeError(
                            "Cannot save Neuron optimizer layout metadata: "
                            f"{layout_source} param_names are not aligned with params."
                        )
                try:
                    live_parameter_names = [
                        parameter_names_by_id[id(parameter)]
                        for parameter in live_parameters
                    ]
                except KeyError as error:
                    raise RuntimeError(
                        "Cannot save Neuron optimizer layout metadata: every "
                        "optimizer parameter must be registered on the Lightning "
                        "module."
                    ) from error
                parameter_names_by_group.append(live_parameter_names)
                optimizer_parameter_names.extend(live_parameter_names)
            if len(optimizer_parameter_names) != len(set(optimizer_parameter_names)):
                raise RuntimeError(
                    "Cannot save Neuron optimizer layout metadata: a parameter "
                    "appears more than once in an optimizer."
                )
            optimizer_layouts.append(
                {
                    "parameter_names": parameter_names_by_group,
                    "sync_policy": _ROLE_SYNC_POLICY,
                }
            )
        return {
            "version": _OPTIMIZER_LAYOUT_VERSION,
            "optimizers": optimizer_layouts,
        }

    def prepare_for_load(
        self,
        module: nn.Module,
        optimizers: list[Optimizer],
        saved_optimizer_states: list[dict[str, Any]],
        layout: Any,
    ) -> None:
        self.clear()
        optimizer_layouts = self.__validated_optimizer_layouts(layout)
        if not (
            len(optimizers) == len(saved_optimizer_states) == len(optimizer_layouts)
        ):
            raise RuntimeError(
                "Cannot load named Neuron optimizer state: optimizer counts differ."
            )
        parameters_by_name = dict(module.named_parameters())
        names_by_parameter_id = self.__parameter_names_by_identity(module)
        prepared_migrations = [
            self.__build_migration(
                optimizer,
                saved_state,
                optimizer_layout,
                parameters_by_name,
                names_by_parameter_id,
            )
            for optimizer, saved_state, optimizer_layout in zip(
                optimizers,
                saved_optimizer_states,
                optimizer_layouts,
                strict=True,
            )
        ]
        self._migrations.extend(prepared_migrations)
        try:
            for migration, saved_state, optimizer_layout in zip(
                prepared_migrations,
                saved_optimizer_states,
                optimizer_layouts,
                strict=True,
            ):
                self.__reorder_saved_parameter_ids(
                    migration.optimizer,
                    saved_state,
                    optimizer_layout,
                    names_by_parameter_id,
                )
        except BaseException:
            self.clear()
            raise

    def optimizer_requires_completion(self, optimizer: Optimizer) -> bool:
        return any(migration.optimizer is optimizer for migration in self._migrations)

    def complete_optimizer_load(self, optimizer: Optimizer) -> None:
        remaining_migrations = []
        for migration in self._migrations:
            if migration.optimizer is optimizer:
                self.__restore_saved_parameter_ids(migration)
            else:
                remaining_migrations.append(migration)
        self._migrations = remaining_migrations

    def clear(self) -> None:
        for migration in self._migrations:
            migration.optimizer.param_groups[:] = list(migration.original_groups)
            self.__restore_saved_parameter_ids(migration)
        self._migrations.clear()

    @staticmethod
    def __parameter_names_by_identity(module: nn.Module) -> dict[int, str]:
        return {id(parameter): name for name, parameter in module.named_parameters()}

    @staticmethod
    def __validated_optimizer_layouts(layout: Any) -> list[dict[str, Any]]:
        if not isinstance(layout, dict):
            raise RuntimeError("Invalid named Neuron optimizer layout metadata.")
        if set(layout) != {"version", "optimizers"}:
            raise RuntimeError("Invalid named Neuron optimizer layout metadata.")
        if layout.get("version") != _OPTIMIZER_LAYOUT_VERSION:
            raise RuntimeError(
                "Unsupported named Neuron optimizer layout metadata version: "
                f"{layout.get('version')!r}."
            )
        optimizer_layouts = layout.get("optimizers")
        if not isinstance(optimizer_layouts, list) or not all(
            isinstance(optimizer_layout, dict) for optimizer_layout in optimizer_layouts
        ):
            raise RuntimeError("Invalid named Neuron optimizer layout metadata.")
        return optimizer_layouts

    @classmethod
    def __build_migration(
        cls,
        optimizer: Optimizer,
        saved_state: dict[str, Any],
        optimizer_layout: dict[str, Any],
        parameters_by_name: dict[str, nn.Parameter],
        names_by_parameter_id: dict[int, str],
    ) -> _NamedOptimizerMigration:
        saved_groups = saved_state.get("param_groups")
        saved_group_names = optimizer_layout.get("parameter_names")
        if set(optimizer_layout) != {"parameter_names", "sync_policy"}:
            raise RuntimeError("Invalid named Neuron optimizer layout metadata.")
        if optimizer_layout.get("sync_policy") != _ROLE_SYNC_POLICY:
            raise RuntimeError("Invalid named Neuron optimizer sync policy.")
        if (
            not isinstance(saved_groups, list)
            or not isinstance(saved_group_names, list)
            or len(saved_groups) != len(saved_group_names)
        ):
            raise RuntimeError("Invalid named Neuron optimizer group metadata.")
        validated_saved_parameter_names = cls.__validate_saved_group_names(
            saved_groups,
            saved_group_names,
            parameters_by_name,
        )
        for saved_group, group_parameter_names in zip(
            saved_groups,
            saved_group_names,
            strict=True,
        ):
            parameter_names = saved_group.get("param_names")
            if parameter_names is not None and (
                not isinstance(parameter_names, list)
                or len(parameter_names) != len(group_parameter_names)
            ):
                raise RuntimeError(
                    "Invalid named Neuron optimizer param_names metadata."
                )
        try:
            live_optimizer_parameter_names = [
                names_by_parameter_id[id(parameter)]
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
        except KeyError as error:
            raise RuntimeError(
                "Cannot load named Neuron optimizer state: every live optimizer "
                "parameter must be registered on the Lightning module."
            ) from error
        if len(live_optimizer_parameter_names) != len(
            set(live_optimizer_parameter_names)
        ) or set(live_optimizer_parameter_names) != set(
            validated_saved_parameter_names
        ):
            raise RuntimeError(
                "Cannot load named Neuron optimizer state: configured parameter "
                "membership differs from the checkpoint."
            )
        if len(optimizer.param_groups) != len(saved_group_names):
            raise RuntimeError(
                "Cannot load named Neuron optimizer state: configured "
                "parameter-group counts differ from the checkpoint."
            )
        for live_group, group_parameter_names in zip(
            optimizer.param_groups,
            saved_group_names,
            strict=True,
        ):
            live_group_parameter_names = {
                names_by_parameter_id[id(parameter)]
                for parameter in live_group["params"]
            }
            if live_group_parameter_names != set(group_parameter_names):
                raise RuntimeError(
                    "Cannot load named Neuron optimizer state: configured "
                    "parameter-group membership differs from the checkpoint."
                )
        return _NamedOptimizerMigration(
            optimizer=optimizer,
            original_groups=tuple(optimizer.param_groups),
            saved_state=saved_state,
            original_saved_parameter_ids=tuple(
                tuple(group["params"]) for group in saved_groups
            ),
            original_saved_parameter_names=tuple(
                tuple(group["param_names"])
                if isinstance(group.get("param_names"), list)
                else None
                for group in saved_groups
            ),
        )

    @staticmethod
    def __restore_saved_parameter_ids(
        migration: _NamedOptimizerMigration,
    ) -> None:
        for group, parameter_ids in zip(
            migration.saved_state["param_groups"],
            migration.original_saved_parameter_ids,
            strict=True,
        ):
            group["params"] = list(parameter_ids)
        for group, parameter_names in zip(
            migration.saved_state["param_groups"],
            migration.original_saved_parameter_names,
            strict=True,
        ):
            if parameter_names is not None:
                group["param_names"] = list(parameter_names)

    @staticmethod
    def __validate_saved_group_names(
        saved_groups: list[dict[str, Any]],
        saved_group_names: list[Any],
        parameters_by_name: dict[str, nn.Parameter],
    ) -> list[str]:
        validated_parameter_names = []
        for saved_group, group_parameter_names in zip(
            saved_groups,
            saved_group_names,
            strict=True,
        ):
            serialized_parameter_ids = saved_group.get("params")
            if (
                not isinstance(group_parameter_names, list)
                or not all(
                    isinstance(parameter_name, str)
                    for parameter_name in group_parameter_names
                )
                or not isinstance(serialized_parameter_ids, list)
                or len(group_parameter_names) != len(serialized_parameter_ids)
            ):
                raise RuntimeError("Invalid named Neuron optimizer group metadata.")
            if any(
                parameter_name not in parameters_by_name
                for parameter_name in group_parameter_names
            ):
                raise RuntimeError(
                    "Cannot load named Neuron optimizer state: a saved parameter "
                    "name is absent from the reconstructed model."
                )
            validated_parameter_names.extend(group_parameter_names)
        if len(validated_parameter_names) != len(set(validated_parameter_names)):
            raise RuntimeError(
                "Invalid named Neuron optimizer layout: duplicate parameter name."
            )
        return validated_parameter_names

    @staticmethod
    def __reorder_saved_parameter_ids(
        optimizer: Optimizer,
        saved_state: dict[str, Any],
        optimizer_layout: dict[str, Any],
        names_by_parameter_id: dict[int, str],
    ) -> None:
        for live_group, saved_group, layout_parameter_names in zip(
            optimizer.param_groups,
            saved_state["param_groups"],
            optimizer_layout["parameter_names"],
            strict=True,
        ):
            saved_parameter_ids_by_name = dict(
                zip(layout_parameter_names, saved_group["params"], strict=True)
            )
            live_parameter_names = [
                names_by_parameter_id[id(parameter)]
                for parameter in live_group["params"]
            ]
            saved_group["params"] = [
                saved_parameter_ids_by_name[name] for name in live_parameter_names
            ]
            serialized_parameter_names = saved_group.get("param_names")
            if serialized_parameter_names is not None:
                serialized_parameter_names_by_layout_name = dict(
                    zip(
                        layout_parameter_names,
                        serialized_parameter_names,
                        strict=True,
                    )
                )
                saved_group["param_names"] = [
                    serialized_parameter_names_by_layout_name[name]
                    for name in live_parameter_names
                ]

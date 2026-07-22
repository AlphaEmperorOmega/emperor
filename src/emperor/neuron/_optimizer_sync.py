from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from emperor.neuron._distributed_gradients import (
    average_post_wrap_gradients,
    configure_conditional_ddp_strategy,
)
from emperor.neuron._optimizer_layout import (
    OPTIMIZER_LAYOUT_CHECKPOINT_KEY,
    NeuronOptimizerNamedLayout,
)
from emperor.neuron._optimizer_scheduler import (
    NeuronSchedulerCheckpointReconciler,
    NeuronSchedulerMutationTransaction,
    SchedulerGroupLoadBinding,
    preflight_scheduler_group_removal,
    remove_scheduler_groups,
)
from emperor.neuron._optimizer_transaction import NeuronOptimizerLoadTransaction

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer


class NeuronClusterOptimizerSyncCallback(Callback):
    """Keep dynamic parameters synchronized and configure conditional DDP."""

    def __init__(self) -> None:
        super().__init__()
        self._clusters: list[nn.Module] = []
        self._synced_neuron_names: dict[int, set[str]] = {}
        self._synced_param_ids: dict[int, set[int]] = {}
        self._synced_parameter_names_by_id: dict[int, str] = {}
        self._post_wrap_param_ids: set[int] = set()
        self._fit_started = False
        self._optimizer_load_transaction = NeuronOptimizerLoadTransaction()
        self._named_layout = NeuronOptimizerNamedLayout()
        self._scheduler_reconciler = NeuronSchedulerCheckpointReconciler()
        self._pending_saved_optimizer_states: list[dict] | None = None
        self._pending_saved_scheduler_states: list[dict] | None = None
        self._pending_named_optimizer_layout: dict | None = None
        self._optimizer_load_hook_handles: dict[int, RemovableHandle] = {}

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        if stage != "fit":
            return
        configure_conditional_ddp_strategy(trainer.strategy)

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict,
    ) -> None:
        if not self.__is_fitting(trainer):
            return
        saved_optimizer_states = checkpoint.get("optimizer_states")
        if not isinstance(saved_optimizer_states, list):
            return
        self._pending_saved_optimizer_states = saved_optimizer_states
        saved_scheduler_states = checkpoint.get("lr_schedulers")
        self._pending_saved_scheduler_states = (
            saved_scheduler_states if isinstance(saved_scheduler_states, list) else None
        )
        named_layout = checkpoint.get(OPTIMIZER_LAYOUT_CHECKPOINT_KEY)
        if saved_optimizer_states and not isinstance(named_layout, dict):
            raise RuntimeError(
                "Neuron optimizer checkpoints require canonical named-layout "
                "metadata; this checkpoint uses a retired optimizer layout."
            )
        self._pending_named_optimizer_layout = named_layout
        optimizers = list(getattr(trainer, "optimizers", []) or [])
        if not optimizers:
            return
        self._clusters = self.__find_neuron_clusters(pl_module)
        self.sync_optimizers(trainer, pl_module)
        self.__prepare_optimizer_checkpoint_load(
            trainer,
            pl_module,
            optimizers,
            saved_optimizer_states,
        )

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict,
    ) -> None:
        saved_optimizer_states = checkpoint.get("optimizer_states")
        optimizers = list(getattr(trainer, "optimizers", []) or [])
        if not isinstance(saved_optimizer_states, list) or not optimizers:
            return
        checkpoint[OPTIMIZER_LAYOUT_CHECKPOINT_KEY] = (
            NeuronOptimizerNamedLayout.capture(
                pl_module,
                optimizers,
                saved_optimizer_states,
            )
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__commit_optimizer_checkpoint_load()
        self._synced_neuron_names.clear()
        self._synced_param_ids.clear()
        self._synced_parameter_names_by_id.clear()
        self._post_wrap_param_ids.clear()
        self._fit_started = False
        self._clusters = self.__find_neuron_clusters(pl_module)
        self.sync_optimizers(trainer, pl_module)
        optimizers = list(getattr(trainer, "optimizers", []) or [])
        if self._pending_saved_optimizer_states is not None:
            self.__prepare_optimizer_checkpoint_load(
                trainer,
                pl_module,
                optimizers,
                self._pending_saved_optimizer_states,
            )
        self._fit_started = True

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__commit_optimizer_checkpoint_load()

    def __prepare_optimizer_checkpoint_load(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizers: list[Optimizer],
        saved_optimizer_states: list[dict],
    ) -> None:
        named_layout = self._pending_named_optimizer_layout
        self._pending_saved_optimizer_states = None
        self._pending_named_optimizer_layout = None
        if named_layout is None and saved_optimizer_states:
            raise RuntimeError(
                "Neuron optimizer checkpoints require canonical named-layout "
                "metadata; this checkpoint uses a retired optimizer layout."
            )
        self._optimizer_load_transaction.prepare_for_load(optimizers)
        try:
            if saved_optimizer_states:
                assert named_layout is not None
                self._named_layout.prepare_for_load(
                    pl_module,
                    optimizers,
                    saved_optimizer_states,
                    named_layout,
                )
            self.__reconcile_pending_schedulers(trainer, optimizers)
            self.__register_optimizer_load_hooks(optimizers)
        except BaseException:
            self.__remove_optimizer_load_hooks()
            self._scheduler_reconciler.clear()
            self._named_layout.clear()
            self.__rollback_optimizer_checkpoint_load()
            raise

    def __reconcile_pending_schedulers(
        self,
        trainer: Trainer,
        optimizers: list[Optimizer],
    ) -> None:
        scheduler_configs = list(getattr(trainer, "lr_scheduler_configs", []) or [])
        saved_scheduler_states = self._pending_saved_scheduler_states
        self._pending_saved_scheduler_states = None
        if not scheduler_configs:
            self._scheduler_reconciler.prepare_for_load([])
            return
        if saved_scheduler_states is not None and len(saved_scheduler_states) != len(
            scheduler_configs
        ):
            raise RuntimeError(
                "Cannot safely restore Neuron optimizer schedulers: "
                "live and saved scheduler counts differ."
            )
        aligned_saved_scheduler_states = (
            saved_scheduler_states
            if saved_scheduler_states is not None
            else [None] * len(scheduler_configs)
        )
        reconciled_optimizer_ids = {id(optimizer) for optimizer in optimizers}
        scheduler_load_bindings = []
        for scheduler_config, saved_state in zip(
            scheduler_configs,
            aligned_saved_scheduler_states,
            strict=True,
        ):
            scheduler = scheduler_config.scheduler
            optimizer = getattr(scheduler, "optimizer", None)
            if id(optimizer) not in reconciled_optimizer_ids:
                continue
            scheduler_load_bindings.append(
                SchedulerGroupLoadBinding(
                    scheduler=scheduler,
                    saved_state=saved_state,
                    optimizer=optimizer,
                )
            )
        self._scheduler_reconciler.prepare_for_load(scheduler_load_bindings)

    @staticmethod
    def __is_fitting(trainer: Trainer) -> bool:
        trainer_function = getattr(getattr(trainer, "state", None), "fn", None)
        return trainer_function is None or trainer_function == TrainerFn.FITTING

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        self.__sync_optimizers_if_clusters_grew(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self.__sync_optimizers_if_clusters_grew(trainer, pl_module)

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
    ) -> None:
        average_post_wrap_gradients(
            pl_module,
            optimizer,
            self._post_wrap_param_ids,
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__clear_fit_state()

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        self.__clear_fit_state()

    def __sync_optimizers_if_clusters_grew(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if not self.__clusters_changed_since_last_sync(pl_module):
            return
        self.sync_optimizers(trainer, pl_module)

    def __clusters_changed_since_last_sync(
        self,
        pl_module: LightningModule,
    ) -> bool:
        clusters = self._clusters or self.__find_neuron_clusters(pl_module)
        return any(
            self._synced_neuron_names.get(id(cluster)) != set(cluster.cluster.keys())
            for cluster in clusters
        )

    def sync_optimizers(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        optimizers = list(getattr(trainer, "optimizers", []) or [])
        optimizer_transaction = NeuronOptimizerLoadTransaction()
        optimizer_transaction.prepare_for_load(optimizers)
        scheduler_transaction = NeuronSchedulerMutationTransaction()
        scheduler_transaction.prepare(
            [
                scheduler_config.scheduler
                for scheduler_config in list(
                    getattr(trainer, "lr_scheduler_configs", []) or []
                )
            ]
        )
        try:
            self.__sync_optimizers(trainer, pl_module)
        except BaseException:
            scheduler_transaction.clear()
            optimizer_transaction.clear()
            raise
        scheduler_transaction.commit()
        optimizer_transaction.commit()

    def __sync_optimizers(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        optimizers = list(getattr(trainer, "optimizers", []) or [])
        if not optimizers:
            return

        clusters = self._clusters or self.__find_neuron_clusters(pl_module)
        if not clusters:
            return
        parameter_names_by_id = {
            id(parameter): name for name, parameter in pl_module.named_parameters()
        }
        live_module_parameter_ids = {
            id(parameter) for parameter in pl_module.parameters()
        }
        new_post_wrap_param_ids = (
            {
                id(parameter)
                for cluster in clusters
                for parameter in cluster.parameters()
                if id(parameter)
                not in self._synced_param_ids.get(
                    id(cluster),
                    {
                        id(current_parameter)
                        for current_parameter in cluster.parameters()
                    },
                )
            }
            if self._fit_started
            else set()
        )

        for optimizer in optimizers:
            self.__remove_pruned_neuron_parameters(
                trainer,
                optimizer,
                clusters,
                live_module_parameter_ids,
                parameter_names_by_id,
            )
        parameter_locations = self.__optimizer_parameter_locations(optimizers)
        for cluster in clusters:
            self.__sync_cluster_parameters(
                cluster,
                parameter_locations,
                parameter_names_by_id,
            )
        self.__warn_about_unoptimized_cluster_parameters(optimizers, clusters)
        self._synced_neuron_names = {
            id(cluster): set(cluster.cluster.keys()) for cluster in clusters
        }
        self._synced_param_ids = {
            id(cluster): {id(parameter) for parameter in cluster.parameters()}
            for cluster in clusters
        }
        self._synced_parameter_names_by_id = dict(parameter_names_by_id)
        current_cluster_param_ids = {
            parameter_id
            for parameter_ids in self._synced_param_ids.values()
            for parameter_id in parameter_ids
        }
        self._post_wrap_param_ids.intersection_update(current_cluster_param_ids)
        self._post_wrap_param_ids.update(new_post_wrap_param_ids)
        for cluster in clusters:
            cluster._checkpoint_removed_parameter_ids.clear()

    def __find_neuron_clusters(self, module: nn.Module):
        from emperor.neuron._cluster.model import NeuronCluster

        clusters: list[nn.Module] = []
        seen_cluster_ids: set[int] = set()
        for _, candidate_module in module.named_modules(remove_duplicate=False):
            if not isinstance(candidate_module, NeuronCluster):
                continue
            if id(candidate_module) in seen_cluster_ids:
                continue
            clusters.append(candidate_module)
            seen_cluster_ids.add(id(candidate_module))
        return clusters

    def __sync_cluster_parameters(
        self,
        cluster: nn.Module,
        parameter_locations: dict[int, list[tuple[Optimizer, dict]]],
        parameter_names_by_id: dict[int, str],
    ) -> None:
        named_parameters = list(cluster.named_parameters(remove_duplicate=False))
        cluster_parameter_order: dict[int, int] = {}
        for index, (_, parameter) in enumerate(named_parameters):
            cluster_parameter_order.setdefault(id(parameter), index)
        parameters_by_role: dict[str, list[nn.Parameter]] = {}
        for name, parameter in named_parameters:
            role = self.__dynamic_neuron_parameter_role(name)
            if role is not None:
                parameters_by_role.setdefault(role, []).append(parameter)

        for name, parameter in named_parameters:
            if id(parameter) in parameter_locations:
                continue
            role = self.__dynamic_neuron_parameter_role(name)
            if role is None:
                continue
            role_parameter_locations = {
                (id(optimizer), id(group)): (optimizer, group)
                for role_parameter in parameters_by_role[role]
                for optimizer, group in parameter_locations.get(id(role_parameter), [])
            }
            if len(role_parameter_locations) != 1:
                continue
            owning_optimizer, owning_group = next(
                iter(role_parameter_locations.values())
            )
            self.__insert_in_cluster_parameter_order(
                owning_group,
                parameter,
                cluster_parameter_order,
                parameter_names_by_id,
            )
            parameter_locations[id(parameter)] = [(owning_optimizer, owning_group)]

    def __insert_in_cluster_parameter_order(
        self,
        group: dict,
        parameter: nn.Parameter,
        cluster_parameter_order: dict[int, int],
        parameter_names_by_id: dict[int, str],
    ) -> None:
        desired_parameter_order = cluster_parameter_order[id(parameter)]
        last_cluster_group_index: int | None = None
        insertion_index = len(group["params"])
        for index, existing_group_parameter in enumerate(group["params"]):
            existing_parameter_order = cluster_parameter_order.get(
                id(existing_group_parameter)
            )
            if existing_parameter_order is None:
                continue
            if existing_parameter_order > desired_parameter_order:
                insertion_index = index
                break
            last_cluster_group_index = index
        else:
            insertion_index = (
                len(group["params"])
                if last_cluster_group_index is None
                else last_cluster_group_index + 1
            )
        if "param_names" in group:
            self.__validate_official_param_names(group, parameter_names_by_id)
            try:
                parameter_name = parameter_names_by_id[id(parameter)]
            except KeyError as error:
                raise RuntimeError(
                    "Cannot safely synchronize optimizer param_names for a "
                    "dynamic Neuron parameter that is not registered on the "
                    "Lightning module."
                ) from error
            group["param_names"].insert(insertion_index, parameter_name)
        group["params"].insert(insertion_index, parameter)

    def __dynamic_neuron_parameter_role(self, name: str) -> str | None:
        parameter_name_parts = name.split(".", 2)
        if (
            len(parameter_name_parts) != 3
            or parameter_name_parts[0] != "cluster"
            or not parameter_name_parts[1].startswith("neuron_")
        ):
            return None
        return parameter_name_parts[2]

    def __remove_pruned_neuron_parameters(
        self,
        trainer: Trainer,
        optimizer: Optimizer,
        clusters: list[nn.Module],
        live_module_parameter_ids: set[int],
        parameter_names_by_id: dict[int, str],
    ) -> None:
        pruned_cluster_param_ids: set[int] = set()
        for cluster in clusters:
            pruned_cluster_param_ids.update(
                self._synced_param_ids.get(id(cluster), set())
                - {id(parameter) for parameter in cluster.parameters()}
            )
            pruned_cluster_param_ids.update(cluster._checkpoint_removed_parameter_ids)
        stale_param_ids = pruned_cluster_param_ids - live_module_parameter_ids
        topology_was_pruned = any(
            bool(
                self._synced_neuron_names.get(id(cluster), set())
                - set(cluster.cluster.keys())
            )
            for cluster in clusters
        )
        if not pruned_cluster_param_ids and not topology_was_pruned:
            return

        previous_group_count = len(optimizer.param_groups)
        projected_parameters = [
            [
                parameter
                for parameter in group["params"]
                if id(parameter) not in stale_param_ids
            ]
            for group in optimizer.param_groups
        ]
        projected_parameter_names: list[list[str] | None] = []
        for group in optimizer.param_groups:
            parameter_names = group.get("param_names")
            if parameter_names is None:
                projected_parameter_names.append(None)
                continue
            if not isinstance(parameter_names, list) or len(parameter_names) != len(
                group["params"]
            ):
                raise RuntimeError(
                    "Cannot safely prune Neuron optimizer parameters because "
                    "param_names are not aligned with params."
                )
            group_uses_official_parameter_names = all(
                self._synced_parameter_names_by_id.get(
                    id(parameter),
                    parameter_names_by_id.get(id(parameter)),
                )
                == name
                for name, parameter in zip(
                    parameter_names,
                    group["params"],
                    strict=True,
                )
            )
            projected_parameter_names.append(
                [
                    (
                        parameter_names_by_id.get(id(parameter), name)
                        if group_uses_official_parameter_names
                        else name
                    )
                    for name, parameter in zip(
                        parameter_names,
                        group["params"],
                        strict=True,
                    )
                    if id(parameter) not in stale_param_ids
                ]
            )
        removed_group_indices = tuple(
            index
            for index, (group, parameters) in enumerate(
                zip(optimizer.param_groups, projected_parameters, strict=True)
            )
            if group["params"] and not parameters
        )
        schedulers = self.__optimizer_schedulers(trainer, optimizer)
        if removed_group_indices:
            for scheduler in schedulers:
                preflight_scheduler_group_removal(
                    scheduler,
                    removed_group_indices,
                    previous_group_count=previous_group_count,
                )

        for group, parameters, parameter_names in zip(
            optimizer.param_groups,
            projected_parameters,
            projected_parameter_names,
            strict=True,
        ):
            group["params"] = parameters
            if parameter_names is not None:
                group["param_names"] = parameter_names
        for parameter in list(optimizer.state.keys()):
            if id(parameter) in stale_param_ids:
                optimizer.state.pop(parameter, None)
        removed_group_index_set = set(removed_group_indices)
        optimizer.param_groups[:] = [
            group
            for index, group in enumerate(optimizer.param_groups)
            if index not in removed_group_index_set
        ]
        for scheduler in schedulers:
            remove_scheduler_groups(
                scheduler,
                removed_group_indices,
                previous_group_count=previous_group_count,
            )

    @staticmethod
    def __optimizer_schedulers(
        trainer: Trainer,
        optimizer: Optimizer,
    ) -> list[object]:
        return [
            scheduler_config.scheduler
            for scheduler_config in list(
                getattr(trainer, "lr_scheduler_configs", []) or []
            )
            if getattr(scheduler_config.scheduler, "optimizer", None) is optimizer
        ]

    @staticmethod
    def __validate_official_param_names(
        group: dict,
        parameter_names_by_id: dict[int, str],
    ) -> None:
        parameter_names = group.get("param_names")
        if not isinstance(parameter_names, list) or len(parameter_names) != len(
            group["params"]
        ):
            raise RuntimeError(
                "Cannot safely synchronize Neuron optimizer param_names because "
                "they are not aligned with params."
            )
        if any(
            not isinstance(parameter_name, str)
            or parameter_names_by_id.get(id(parameter)) != parameter_name
            for parameter, parameter_name in zip(
                group["params"],
                parameter_names,
                strict=True,
            )
        ):
            raise RuntimeError(
                "Cannot derive names for new Neuron optimizer parameters because "
                "the owning group does not use fully-qualified module names."
            )

    def __warn_about_unoptimized_cluster_parameters(
        self,
        optimizers: list[Optimizer],
        clusters: list[nn.Module],
    ) -> None:
        all_optimizer_param_ids: set[int] = set()
        for optimizer in optimizers:
            all_optimizer_param_ids |= self.__optimizer_param_ids(optimizer)

        for cluster in clusters:
            unoptimized_count = sum(
                1
                for parameter in cluster.parameters()
                if parameter.requires_grad
                and id(parameter) not in all_optimizer_param_ids
            )
            if unoptimized_count == 0:
                continue
            warnings.warn(
                f"NeuronClusterOptimizerSyncCallback found {unoptimized_count} "
                "trainable NeuronCluster parameters missing from every optimizer; "
                "their existing same-role parameters do not identify one unique "
                "optimizer parameter group, so these parameters will not be trained.",
                stacklevel=1,
            )

    def __optimizer_param_ids(self, optimizer: Optimizer) -> set[int]:
        return {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    def __optimizer_parameter_locations(
        self,
        optimizers: list[Optimizer],
    ) -> dict[int, list[tuple[Optimizer, dict]]]:
        parameter_locations: dict[int, list[tuple[Optimizer, dict]]] = {}
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter_locations.setdefault(id(parameter), []).append(
                        (optimizer, group)
                    )
        return parameter_locations

    def __register_optimizer_load_hooks(
        self,
        optimizers: list[Optimizer],
    ) -> None:
        for optimizer in optimizers:
            optimizer_id = id(optimizer)
            if optimizer_id in self._optimizer_load_hook_handles or not (
                self._named_layout.optimizer_requires_completion(optimizer)
                or self._scheduler_reconciler.optimizer_requires_completion(optimizer)
                or self._optimizer_load_transaction.optimizer_requires_completion(
                    optimizer
                )
            ):
                continue
            self._optimizer_load_hook_handles[optimizer_id] = (
                optimizer.register_load_state_dict_post_hook(
                    self.__complete_loaded_optimizer
                )
            )

    def __complete_loaded_optimizer(self, optimizer: Optimizer) -> None:
        self._named_layout.complete_optimizer_load(optimizer)
        self._scheduler_reconciler.mark_optimizer_loaded(optimizer)
        self._optimizer_load_transaction.mark_optimizer_loaded(optimizer)
        handle = self._optimizer_load_hook_handles.pop(id(optimizer))
        handle.remove()

    def __remove_optimizer_load_hooks(self) -> None:
        for handle in self._optimizer_load_hook_handles.values():
            handle.remove()
        self._optimizer_load_hook_handles.clear()

    def __clear_fit_state(self) -> None:
        self._clusters.clear()
        self._synced_neuron_names.clear()
        self._synced_param_ids.clear()
        self._synced_parameter_names_by_id.clear()
        self._post_wrap_param_ids.clear()
        self._fit_started = False
        self._pending_saved_optimizer_states = None
        self._pending_saved_scheduler_states = None
        self._pending_named_optimizer_layout = None
        self.__remove_optimizer_load_hooks()
        self._scheduler_reconciler.clear()
        self._named_layout.clear()
        self.__rollback_optimizer_checkpoint_load()

    def __rollback_optimizer_checkpoint_load(self) -> None:
        self._optimizer_load_transaction.clear()

    def __commit_optimizer_checkpoint_load(self) -> None:
        self._optimizer_load_transaction.commit_loaded()
        self._scheduler_reconciler.commit_loaded()

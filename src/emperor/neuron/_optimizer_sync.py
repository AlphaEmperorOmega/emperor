from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from emperor.neuron._distributed_gradients import (
    _ConditionalDDPStrategyAdapter,
    average_post_wrap_gradients,
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
        self._synced_cluster_signatures: dict[
            int, tuple[tuple[str, int, bool], ...]
        ] = {}
        self._synced_parameter_names_by_id: dict[int, str] = {}
        self._synced_parameter_ids_by_name: dict[str, int] = {}
        self._post_wrap_param_ids: set[int] = set()
        self._ddp_registered_param_ids: set[int] = set()
        self._fit_started = False
        self._fit_optimizers: list[Optimizer] | None = None
        self._optimizer_load_transaction = NeuronOptimizerLoadTransaction()
        self._named_layout = NeuronOptimizerNamedLayout()
        self._scheduler_reconciler = NeuronSchedulerCheckpointReconciler()
        self._checkpoint_scheduler_transaction = NeuronSchedulerMutationTransaction()
        self._checkpoint_sync_snapshot: dict | None = None
        self._checkpoint_removed_ids: list[tuple[nn.Module, set[int]]] = []
        self._checkpoint_load_prepared = False
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
        self.__clear_fit_state()
        _ConditionalDDPStrategyAdapter.configure(trainer.strategy)

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
        saved_scheduler_states = checkpoint.get("lr_schedulers")
        named_layout = checkpoint.get(OPTIMIZER_LAYOUT_CHECKPOINT_KEY)
        if saved_optimizer_states and not isinstance(named_layout, dict):
            raise RuntimeError(
                "Neuron optimizer checkpoints require canonical named-layout "
                "metadata; this checkpoint uses a retired optimizer layout."
            )
        if named_layout is not None:
            NeuronOptimizerNamedLayout.validate_checkpoint(
                saved_optimizer_states, named_layout
            )
        if (
            self._checkpoint_load_prepared
            and self._pending_saved_optimizer_states is saved_optimizer_states
        ):
            return
        self.__remove_optimizer_load_hooks()
        self._scheduler_reconciler.clear()
        self._named_layout.clear()
        self.__rollback_optimizer_checkpoint_load()
        self._pending_saved_optimizer_states = saved_optimizer_states
        self._pending_saved_scheduler_states = (
            saved_scheduler_states if isinstance(saved_scheduler_states, list) else None
        )
        self._pending_named_optimizer_layout = named_layout
        optimizers = self._fit_optimizers
        if optimizers is None:
            return
        if optimizers != list(getattr(trainer, "optimizers", []) or []):
            raise RuntimeError(
                "Trainer optimizers changed after Neuron fit binding; "
                "begin a new fit before restoring a checkpoint."
            )
        self._clusters = self.__find_neuron_clusters(pl_module)
        self.__prepare_optimizer_checkpoint_load(
            trainer,
            pl_module,
            optimizers,
            saved_optimizer_states,
        )

    @staticmethod
    def __is_fitting(trainer: Trainer) -> bool:
        trainer_function = getattr(getattr(trainer, "state", None), "fn", None)
        return trainer_function is None or trainer_function == TrainerFn.FITTING

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

    def __prepare_optimizer_checkpoint_load(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizers: list[Optimizer],
        saved_optimizer_states: list[dict],
    ) -> None:
        named_layout = self._pending_named_optimizer_layout
        if named_layout is None and saved_optimizer_states:
            raise RuntimeError(
                "Neuron optimizer checkpoints require canonical named-layout "
                "metadata; this checkpoint uses a retired optimizer layout."
            )
        self._optimizer_load_transaction.prepare_for_load(optimizers)
        self._checkpoint_scheduler_transaction.prepare(
            [
                config.scheduler
                for config in list(getattr(trainer, "lr_scheduler_configs", []) or [])
            ]
        )
        self.__capture_checkpoint_synchronization()
        try:
            self.sync_optimizers(trainer, pl_module)
            if saved_optimizer_states:
                assert named_layout is not None
                self._named_layout.prepare_for_load(
                    pl_module,
                    optimizers,
                    saved_optimizer_states,
                    named_layout,
                )
            self.__reconcile_pending_schedulers(trainer, optimizers)
            if saved_optimizer_states:
                self.__register_optimizer_load_hooks(optimizers)
            else:
                # No serialized optimizer load is expected, but sync mutations
                # still need rollback protection until the fit can start.
                for optimizer in optimizers:
                    self._optimizer_load_transaction.mark_optimizer_loaded(optimizer)
                    self._scheduler_reconciler.mark_optimizer_loaded(optimizer)
            self._checkpoint_load_prepared = True
        except BaseException:
            self.__remove_optimizer_load_hooks()
            self._scheduler_reconciler.clear()
            self._named_layout.clear()
            self.__rollback_optimizer_checkpoint_load()
            raise

    def __capture_checkpoint_synchronization(self) -> None:
        self._checkpoint_sync_snapshot = {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                "_synced_neuron_names",
                "_synced_param_ids",
                "_synced_cluster_signatures",
                "_synced_parameter_names_by_id",
                "_synced_parameter_ids_by_name",
                "_post_wrap_param_ids",
            )
        }
        self._checkpoint_removed_ids = [
            (cluster, set(cluster._checkpoint_removed_parameter_ids))
            for cluster in self._clusters
        ]


    def __reconcile_pending_schedulers(
        self,
        trainer: Trainer,
        optimizers: list[Optimizer],
    ) -> None:
        scheduler_configs = list(getattr(trainer, "lr_scheduler_configs", []) or [])
        saved_scheduler_states = self._pending_saved_scheduler_states
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

    def __rollback_optimizer_checkpoint_load(self) -> None:
        self._optimizer_load_transaction.clear()
        self._checkpoint_scheduler_transaction.clear()
        if self._checkpoint_sync_snapshot is not None:
            for name, value in self._checkpoint_sync_snapshot.items():
                setattr(self, name, value)
        for cluster, removed_ids in self._checkpoint_removed_ids:
            cluster._checkpoint_removed_parameter_ids.update(removed_ids)
        self._checkpoint_sync_snapshot = None
        self._checkpoint_removed_ids.clear()
        self._checkpoint_load_prepared = False

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
        optimizers = list(getattr(trainer, "optimizers", []) or [])
        self._synced_neuron_names.clear()
        self._synced_param_ids.clear()
        self._synced_cluster_signatures.clear()
        self._synced_parameter_names_by_id.clear()
        self._synced_parameter_ids_by_name.clear()
        self._post_wrap_param_ids.clear()
        self._fit_started = False
        self._clusters = self.__find_neuron_clusters(pl_module)
        self._ddp_registered_param_ids = {
            id(parameter)
            for cluster in self._clusters
            for parameter in cluster.parameters()
            if parameter.requires_grad
        }
        self._fit_optimizers = optimizers
        if self._pending_saved_optimizer_states is not None:
            self.__prepare_optimizer_checkpoint_load(
                trainer,
                pl_module,
                optimizers,
                self._pending_saved_optimizer_states,
            )
        else:
            self.sync_optimizers(trainer, pl_module)
        self._fit_started = True

    def __commit_optimizer_checkpoint_load(self) -> None:
        if (
            self._pending_saved_optimizer_states is not None
            and not self._checkpoint_load_prepared
        ):
            raise RuntimeError(
                "Cannot commit a Neuron checkpoint before fit optimizers are ready."
            )
        self._optimizer_load_transaction.commit_loaded()
        self._scheduler_reconciler.commit_loaded()
        self._checkpoint_scheduler_transaction.commit()
        self._checkpoint_sync_snapshot = None
        self._checkpoint_removed_ids.clear()
        self._checkpoint_load_prepared = False
        self._pending_saved_optimizer_states = None
        self._pending_saved_scheduler_states = None
        self._pending_named_optimizer_layout = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__commit_optimizer_checkpoint_load()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        self.__sync_optimizers_if_clusters_grew(trainer, pl_module)

    def on_before_zero_grad(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
    ) -> None:
        self.__sync_optimizers_if_clusters_grew(trainer, pl_module)

    def on_before_backward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        loss,
    ) -> None:
        self.__sync_optimizers_if_clusters_grew(trainer, pl_module)


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
            or self._synced_cluster_signatures.get(id(cluster))
            != self.__cluster_parameter_signature(cluster)
            for cluster in clusters
        )

    @staticmethod
    def __cluster_parameter_signature(
        cluster: nn.Module,
    ) -> tuple[tuple[str, int, bool], ...]:
        return tuple(
            (name, id(parameter), parameter.requires_grad)
            for name, parameter in cluster.named_parameters(remove_duplicate=False)
        )


    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self.__sync_optimizers_if_clusters_grew(trainer, pl_module)

    def on_after_backward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        # Lightning unscales after backward and before on_before_optimizer_step.
        # Reduce each microbatch while scaled so overflow decisions agree on all ranks.
        reduced_parameter_ids: set[int] = set()
        for optimizer in list(getattr(trainer, "optimizers", []) or []):
            average_post_wrap_gradients(
                pl_module,
                optimizer,
                self._post_wrap_param_ids - reduced_parameter_ids,
            )
            reduced_parameter_ids.update(
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__clear_fit_state()

    def __clear_fit_state(self) -> None:
        self.__remove_optimizer_load_hooks()
        self._scheduler_reconciler.clear()
        self._named_layout.clear()
        self.__rollback_optimizer_checkpoint_load()
        self._clusters.clear()
        self._synced_neuron_names.clear()
        self._synced_param_ids.clear()
        self._synced_cluster_signatures.clear()
        self._synced_parameter_names_by_id.clear()
        self._synced_parameter_ids_by_name.clear()
        self._post_wrap_param_ids.clear()
        self._ddp_registered_param_ids.clear()
        self._fit_started = False
        self._fit_optimizers = None
        self._pending_saved_optimizer_states = None
        self._pending_saved_scheduler_states = None
        self._pending_named_optimizer_layout = None

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        self.__clear_fit_state()

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
        parameters_by_name = dict(pl_module.named_parameters(remove_duplicate=False))
        parameter_names_by_id: dict[int, str] = {}
        for name, parameter in parameters_by_name.items():
            parameter_names_by_id.setdefault(id(parameter), name)
        live_module_parameter_ids = set(parameter_names_by_id)
        cluster_parameters = {
            id(cluster): tuple(cluster.parameters()) for cluster in clusters
        }
        current_parameter_ids = {
            cluster_id: {id(parameter) for parameter in parameters}
            for cluster_id, parameters in cluster_parameters.items()
        }
        new_post_wrap_param_ids = self.__new_post_wrap_parameter_ids(
            current_parameter_ids
        )
        replacements = self.__same_name_replacements(
            optimizers, parameters_by_name, current_parameter_ids
        )

        for optimizer in optimizers:
            self.__remove_pruned_neuron_parameters(
                trainer,
                optimizer,
                clusters,
                live_module_parameter_ids,
                parameter_names_by_id,
                current_parameter_ids,
                replacements,
            )
        parameter_locations = self.__optimizer_parameter_locations(optimizers)
        for cluster in clusters:
            self.__sync_cluster_parameters(
                cluster,
                parameter_locations,
                parameter_names_by_id,
            )
        self.__warn_about_unoptimized_cluster_parameters(optimizers, cluster_parameters)
        self.__record_synchronized_parameters(
            clusters,
            parameter_names_by_id,
            new_post_wrap_param_ids,
            current_parameter_ids,
            parameters_by_name,
        )

    def __new_post_wrap_parameter_ids(
        self,
        current_parameter_ids: dict[int, set[int]],
    ) -> set[int]:
        if not self._fit_started:
            return set()
        new_parameter_ids: set[int] = set()
        for parameter_ids in current_parameter_ids.values():
            new_parameter_ids.update(parameter_ids - self._ddp_registered_param_ids)
        return new_parameter_ids


    def __remove_pruned_neuron_parameters(
        self,
        trainer: Trainer,
        optimizer: Optimizer,
        clusters: list[nn.Module],
        live_module_parameter_ids: set[int],
        parameter_names_by_id: dict[int, str],
        current_parameter_ids: dict[int, set[int]],
        replacements: dict[int, list[nn.Parameter]],
    ) -> None:
        pruned_cluster_param_ids = self.__pruned_cluster_parameter_ids(
            clusters, current_parameter_ids
        )
        stale_param_ids = pruned_cluster_param_ids - live_module_parameter_ids
        topology_was_pruned = any(
            bool(
                self._synced_neuron_names.get(id(cluster), set())
                - set(cluster.cluster.keys())
            )
            for cluster in clusters
        )
        if (
            not pruned_cluster_param_ids
            and not topology_was_pruned
            and not replacements
        ):
            return

        previous_group_count = len(optimizer.param_groups)
        projected_parameters, projected_parameter_names = (
            self.__project_retained_optimizer_parameters(
                optimizer,
                stale_param_ids,
                parameter_names_by_id,
                replacements,
            )
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

        self.__apply_pruned_optimizer_parameters(
            optimizer,
            projected_parameters,
            projected_parameter_names,
            stale_param_ids,
            removed_group_indices,
        )
        for scheduler in schedulers:
            remove_scheduler_groups(
                scheduler,
                removed_group_indices,
                previous_group_count=previous_group_count,
            )

    def __same_name_replacements(
        self,
        optimizers: list[Optimizer],
        parameters_by_name: dict[str, nn.Parameter],
        current_parameter_ids: dict[int, set[int]],
    ) -> dict[int, list[nn.Parameter]]:
        live_parameters = {
            id(parameter): parameter for parameter in parameters_by_name.values()
        }
        previous_cluster_ids = set().union(*self._synced_param_ids.values())
        live_cluster_ids = set().union(*current_parameter_ids.values())
        locations = self.__optimizer_parameter_locations(optimizers)
        parameter_order = {
            parameter_id: index for index, parameter_id in enumerate(locations)
        }
        candidates: dict[int, set[int]] = {}
        for name, parameter in parameters_by_name.items():
            old_id = self._synced_parameter_ids_by_name.get(name)
            if (
                old_id not in previous_cluster_ids
                or id(parameter) not in live_cluster_ids
            ):
                continue
            if id(parameter) in locations:
                continue
            candidates.setdefault(id(parameter), set()).add(old_id)

        replacements: dict[int, list[nn.Parameter]] = {}
        for parameter_id, previous_ids in candidates.items():
            owners = {
                (id(optimizer), id(group))
                for old_id in previous_ids
                for optimizer, group in locations.get(old_id, [])
            }
            if len(owners) > 1:
                raise RuntimeError(
                    "Cannot synchronize a Neuron parameter replacement with ambiguous "
                    "optimizer group ownership across its previous aliases."
                )
            if not owners:
                continue
            old_id = min(
                (candidate for candidate in previous_ids if candidate in locations),
                key=parameter_order.__getitem__,
            )
            # A split may produce several parameters from one owned alias set.
            # Keep the old parameter too when another live role still uses it.
            if old_id not in replacements:
                replacements[old_id] = (
                    [live_parameters[old_id]] if old_id in live_parameters else []
                )
            replacements[old_id].append(live_parameters[parameter_id])
        live_parameter_order = {
            parameter_id: index for index, parameter_id in enumerate(live_parameters)
        }
        for replacement_parameters in replacements.values():
            replacement_parameters.sort(
                key=lambda parameter: live_parameter_order[id(parameter)]
            )
        return replacements


    def __pruned_cluster_parameter_ids(
        self,
        clusters: list[nn.Module],
        current_parameter_ids: dict[int, set[int]],
    ) -> set[int]:
        pruned_cluster_param_ids: set[int] = set()
        for cluster in clusters:
            pruned_cluster_param_ids.update(
                self._synced_param_ids.get(id(cluster), set())
                - current_parameter_ids[id(cluster)]
            )
            pruned_cluster_param_ids.update(cluster._checkpoint_removed_parameter_ids)
        return pruned_cluster_param_ids

    def __project_retained_optimizer_parameters(
        self,
        optimizer: Optimizer,
        stale_param_ids: set[int],
        parameter_names_by_id: dict[int, str],
        replacements: dict[int, list[nn.Parameter]],
    ) -> tuple[list[list[nn.Parameter]], list[list[str] | None]]:
        projected_parameters = [
            [
                retained_parameter
                for parameter in group["params"]
                for retained_parameter in replacements.get(
                    id(parameter),
                    [] if id(parameter) in stale_param_ids else [parameter],
                )
            ]
            for group in optimizer.param_groups
        ]
        projected_parameter_names: list[list[str] | None] = []
        for group in optimizer.param_groups:
            projected_parameter_names.append(
                self.__project_retained_group_names(
                    group,
                    stale_param_ids,
                    parameter_names_by_id,
                    replacements,
                )
            )
        return projected_parameters, projected_parameter_names

    def __project_retained_group_names(
        self,
        group: dict,
        stale_param_ids: set[int],
        parameter_names_by_id: dict[int, str],
        replacements: dict[int, list[nn.Parameter]],
    ) -> list[str] | None:
        parameter_names = group.get("param_names")
        if parameter_names is None:
            return None
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
        projected_names = []
        for name, parameter in zip(parameter_names, group["params"], strict=True):
            retained_parameters = replacements.get(
                id(parameter), [] if id(parameter) in stale_param_ids else [parameter]
            )
            if len(retained_parameters) > 1 and not group_uses_official_parameter_names:
                raise RuntimeError(
                    "Cannot derive names for split Neuron optimizer parameters "
                    "because the owning group does not use fully-qualified module names."
                )
            projected_names.extend(
                parameter_names_by_id.get(id(retained), name)
                if group_uses_official_parameter_names
                else name
                for retained in retained_parameters
            )
        return projected_names

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
    def __apply_pruned_optimizer_parameters(
        optimizer: Optimizer,
        projected_parameters: list[list[nn.Parameter]],
        projected_parameter_names: list[list[str] | None],
        stale_param_ids: set[int],
        removed_group_indices: tuple[int, ...],
    ) -> None:
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

    def __dynamic_neuron_parameter_role(self, name: str) -> str | None:
        parameter_name_parts = name.split(".", 2)
        if (
            len(parameter_name_parts) != 3
            or parameter_name_parts[0] != "cluster"
            or not parameter_name_parts[1].startswith("neuron_")
        ):
            return None
        return parameter_name_parts[2]

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
        cluster_parameters: dict[int, tuple[nn.Parameter, ...]],
    ) -> None:
        all_optimizer_param_ids: set[int] = set()
        for optimizer in optimizers:
            all_optimizer_param_ids |= self.__optimizer_param_ids(optimizer)

        for parameters in cluster_parameters.values():
            unoptimized_count = sum(
                1
                for parameter in parameters
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

    def __record_synchronized_parameters(
        self,
        clusters: list[nn.Module],
        parameter_names_by_id: dict[int, str],
        new_post_wrap_param_ids: set[int],
        current_parameter_ids: dict[int, set[int]],
        parameters_by_name: dict[str, nn.Parameter],
    ) -> None:
        self._synced_neuron_names = {
            id(cluster): set(cluster.cluster.keys()) for cluster in clusters
        }
        self._synced_param_ids = current_parameter_ids
        self._synced_cluster_signatures = {
            id(cluster): self.__cluster_parameter_signature(cluster)
            for cluster in clusters
        }
        self._synced_parameter_names_by_id = dict(parameter_names_by_id)
        self._synced_parameter_ids_by_name = {
            name: id(parameter) for name, parameter in parameters_by_name.items()
        }
        current_cluster_param_ids = {
            parameter_id
            for parameter_ids in self._synced_param_ids.values()
            for parameter_id in parameter_ids
        }
        self._post_wrap_param_ids.intersection_update(current_cluster_param_ids)
        self._post_wrap_param_ids.update(new_post_wrap_param_ids)
        for cluster in clusters:
            cluster._checkpoint_removed_parameter_ids.clear()

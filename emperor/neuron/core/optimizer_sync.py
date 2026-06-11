from __future__ import annotations

import warnings

from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback
from torch import nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer


class NeuronClusterOptimizerSyncCallback(Callback):
    """Keeps optimizers in sync with neuron cluster membership: grown
    neurons' trainable parameters are added and pruned neurons' parameters
    are removed together with their optimizer state."""

    def __init__(self) -> None:
        super().__init__()
        self._clusters: list[nn.Module] = []
        self._synced_neuron_names: dict[int, set[str]] = {}
        self._synced_param_ids: dict[int, set[int]] = {}

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self._clusters = self.__find_neuron_clusters(pl_module)
        self.sync_optimizers(trainer, pl_module)

    def on_train_batch_start(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        batch,
        batch_idx: int,
    ) -> None:
        self.__sync_optimizers_if_clusters_grew(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self.__sync_optimizers_if_clusters_grew(trainer, pl_module)

    def __sync_optimizers_if_clusters_grew(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
    ) -> None:
        if not self.__clusters_changed_since_last_sync(pl_module):
            return
        self.sync_optimizers(trainer, pl_module)

    def __clusters_changed_since_last_sync(
        self,
        pl_module: "LightningModule",
    ) -> bool:
        clusters = self._clusters or self.__find_neuron_clusters(pl_module)
        return any(
            self._synced_neuron_names.get(id(cluster))
            != set(cluster.cluster.keys())
            for cluster in clusters
        )

    def sync_optimizers(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
    ) -> None:
        optimizers = list(getattr(trainer, "optimizers", []) or [])
        if not optimizers:
            return

        clusters = self._clusters or self.__find_neuron_clusters(pl_module)
        if not clusters:
            return

        for optimizer in optimizers:
            self.__sync_optimizer(optimizer, clusters)
        self.__warn_about_unoptimized_cluster_parameters(optimizers, clusters)
        self._synced_neuron_names = {
            id(cluster): set(cluster.cluster.keys()) for cluster in clusters
        }
        self._synced_param_ids = {
            id(cluster): {id(parameter) for parameter in cluster.parameters()}
            for cluster in clusters
        }

    def __find_neuron_clusters(self, module: nn.Module):
        from emperor.neuron.core.model import NeuronCluster

        return [
            cluster
            for _, cluster in module.named_modules()
            if isinstance(cluster, NeuronCluster)
        ]

    def __sync_optimizer(
        self,
        optimizer: Optimizer,
        clusters: list[nn.Module],
    ) -> None:
        for cluster in clusters:
            self.__remove_pruned_neuron_parameters(optimizer, cluster)

        optimizer_param_ids = self.__optimizer_param_ids(optimizer)
        for cluster in clusters:
            trainable_params = [
                parameter
                for parameter in cluster.parameters()
                if parameter.requires_grad
            ]
            if not trainable_params:
                continue

            missing_params = [
                parameter
                for parameter in trainable_params
                if id(parameter) not in optimizer_param_ids
            ]
            if not missing_params:
                continue

            reference_group = self.__reference_param_group(optimizer, cluster)
            if reference_group is None:
                continue

            optimizer.add_param_group(
                {
                    **{
                        key: value
                        for key, value in reference_group.items()
                        if key != "params"
                    },
                    "params": missing_params,
                }
            )
            optimizer_param_ids.update(id(parameter) for parameter in missing_params)

    def __remove_pruned_neuron_parameters(
        self,
        optimizer: Optimizer,
        cluster: nn.Module,
    ) -> None:
        current_param_ids = {id(parameter) for parameter in cluster.parameters()}
        stale_param_ids = (
            self._synced_param_ids.get(id(cluster), set()) - current_param_ids
        )
        if not stale_param_ids:
            return

        for group in optimizer.param_groups:
            group["params"] = [
                parameter
                for parameter in group["params"]
                if id(parameter) not in stale_param_ids
            ]
        for parameter in list(optimizer.state.keys()):
            if id(parameter) in stale_param_ids:
                optimizer.state.pop(parameter, None)
        optimizer.param_groups[:] = [
            group for group in optimizer.param_groups if group["params"]
        ]

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
                "no optimizer holds existing parameters of that cluster to copy "
                "a param group from, so these parameters will not be trained."
            )

    def __optimizer_param_ids(self, optimizer: Optimizer) -> set[int]:
        return {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    def __reference_param_group(self, optimizer: Optimizer, cluster: nn.Module):
        cluster_param_ids = {id(parameter) for parameter in cluster.parameters()}
        for group in optimizer.param_groups:
            if any(
                id(parameter) in cluster_param_ids for parameter in group["params"]
            ):
                return group
        return None

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class _ConditionalDDPStrategyAdapter:
    """Own Lightning DDP capability detection, validation, and configuration.

    Neuron supports a Lightning ``DDPStrategy`` when it exposes the effective
    PyTorch DDP options as a mutable mapping. Unknown layouts fail closed before
    any option is changed.
    """

    @classmethod
    def configure(cls, strategy: object) -> None:
        """Require unused-parameter discovery for conditional routes."""

        from lightning.pytorch.strategies import DDPStrategy

        if not isinstance(strategy, DDPStrategy):
            return
        ddp_options = cls.__ddp_options(strategy)
        cls.__validate_options(ddp_options)
        ddp_options["find_unused_parameters"] = True

    @staticmethod
    def __ddp_options(strategy: object) -> MutableMapping[str, Any]:
        try:
            ddp_options = strategy._ddp_kwargs
        except AttributeError as error:
            raise RuntimeError(
                "The installed Lightning DDPStrategy does not expose the DDP "
                "configuration required by NeuronCluster conditional routing."
            ) from error
        if not isinstance(ddp_options, MutableMapping):
            raise RuntimeError(
                "The installed Lightning DDPStrategy does not expose the DDP "
                "configuration as a mutable mapping required by NeuronCluster "
                "conditional routing."
            )
        return ddp_options

    @staticmethod
    def __validate_options(ddp_options: MutableMapping[str, Any]) -> None:
        if ddp_options.get("static_graph") is True:
            raise RuntimeError(
                "NeuronCluster conditional routing, growth, and pruning require "
                "static_graph=False under DDP because the set of used parameters "
                "can change between batches."
            )
        if ddp_options.get("skip_all_reduce_unused_params") is True:
            raise RuntimeError(
                "NeuronCluster conditional routing requires "
                "skip_all_reduce_unused_params=False under DDP because unused "
                "parameters can differ between batches and ranks."
            )
        if ddp_options.get("find_unused_parameters") is False:
            raise RuntimeError(
                "NeuronCluster conditional routing requires "
                "find_unused_parameters=True under DDP; explicitly disabling it "
                "causes repeated backward passes to fail when routes skip neurons."
            )


def average_post_wrap_gradients(
    module: nn.Module,
    optimizer: Optimizer,
    parameter_ids: set[int],
) -> None:
    """Average gradients for parameters registered after DDP wrapped the model."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    named_parameters = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if id(parameter) in parameter_ids and id(parameter) in optimizer_parameter_ids
    ]
    manifest = [
        (
            name,
            tuple(parameter.shape),
            str(parameter.dtype),
            parameter.device.type,
            parameter.requires_grad,
        )
        for name, parameter in named_parameters
    ]
    rank_manifests = [None] * world_size
    # Every rank participates even if its selected set is empty. Detect divergent
    # roles/layouts before differently shaped parameter collectives can hang.
    torch.distributed.all_gather_object(rank_manifests, manifest)
    if any(rank_manifest != manifest for rank_manifest in rank_manifests):
        raise RuntimeError(
            "Distributed Neuron post-wrap parameter manifests differ across ranks; "
            "ordered roles, shapes, dtypes and trainability must agree."
        )
    for _, parameter in named_parameters:
        _average_gradient(parameter, world_size)


def _average_gradient(parameter: nn.Parameter, world_size: int) -> None:
    gradient = parameter.grad
    sparse_gradient_rank_count = torch.tensor(
        int(gradient is not None and gradient.is_sparse),
        dtype=torch.int64,
        device=parameter.device,
    )
    torch.distributed.all_reduce(sparse_gradient_rank_count)
    if int(sparse_gradient_rank_count.item()) > 0:
        raise RuntimeError(
            "Distributed Neuron growth does not support sparse gradients."
        )
    gradient_rank_count = torch.tensor(
        int(gradient is not None),
        dtype=torch.int64,
        device=parameter.device,
    )
    torch.distributed.all_reduce(gradient_rank_count)
    if int(gradient_rank_count.item()) == 0:
        return
    averaged_gradient = (
        torch.zeros_like(parameter) if gradient is None else gradient.detach().clone()
    )
    averaged_gradient.div_(world_size)
    torch.distributed.all_reduce(averaged_gradient)
    if gradient is None:
        parameter.grad = averaged_gradient
    else:
        gradient.copy_(averaged_gradient)

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer


def configure_conditional_ddp_strategy(strategy: object) -> None:
    """Require unused-parameter discovery for conditionally routed clusters."""

    from lightning.pytorch.strategies import DDPStrategy

    if not isinstance(strategy, DDPStrategy):
        return
    try:
        ddp_kwargs = strategy._ddp_kwargs
    except AttributeError as error:
        raise RuntimeError(
            "The installed Lightning DDPStrategy does not expose the DDP "
            "configuration required by NeuronCluster conditional routing."
        ) from error
    if ddp_kwargs.get("static_graph") is True:
        raise RuntimeError(
            "NeuronCluster conditional routing, growth, and pruning require "
            "static_graph=False under DDP because the set of used parameters "
            "can change between batches."
        )
    if ddp_kwargs.get("skip_all_reduce_unused_params") is True:
        raise RuntimeError(
            "NeuronCluster conditional routing requires "
            "skip_all_reduce_unused_params=False under DDP because unused "
            "parameters can differ between batches and ranks."
        )
    if ddp_kwargs.get("find_unused_parameters") is False:
        raise RuntimeError(
            "NeuronCluster conditional routing requires "
            "find_unused_parameters=True under DDP; explicitly disabling it "
            "causes repeated backward passes to fail when routes skip neurons."
        )
    ddp_kwargs["find_unused_parameters"] = True


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
    for _, parameter in module.named_parameters():
        if (
            id(parameter) not in parameter_ids
            or id(parameter) not in optimizer_parameter_ids
        ):
            continue
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
    torch.distributed.all_reduce(averaged_gradient)
    averaged_gradient.div_(world_size)
    if gradient is None:
        parameter.grad = averaged_gradient
    else:
        gradient.copy_(averaged_gradient)

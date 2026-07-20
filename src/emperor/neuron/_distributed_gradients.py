from __future__ import annotations


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

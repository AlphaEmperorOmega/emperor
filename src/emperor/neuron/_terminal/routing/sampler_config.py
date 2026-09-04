import copy

from emperor.sampler import SamplerConfig


def derive_terminal_tree_sampler_config(
    template: SamplerConfig,
    *,
    input_dim: int,
    num_experts: int,
    top_k: int,
) -> SamplerConfig:
    """Derive one independent node config from a terminal sampler template."""

    derived_config = copy.deepcopy(template)
    derived_config.num_experts = num_experts
    derived_config.top_k = top_k
    if derived_config.num_topk_samples is not None:
        derived_config.num_topk_samples = min(
            derived_config.num_topk_samples,
            top_k,
        )

    if derived_config.router_config is None:
        raise ValueError(
            "Terminal routing trees require learned router_config values for "
            "both direction and connection sampler templates."
        )
    derived_config.router_config.input_dim = input_dim
    derived_config.router_config.num_experts = num_experts
    derived_config.router_config.noisy_topk_flag = derived_config.noisy_topk_flag
    return derived_config

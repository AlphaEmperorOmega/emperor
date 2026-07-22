from types import MappingProxyType, ModuleType
from typing import Any, Final

import models.experts.linear_adaptive.config as config
from models.experts.linear_adaptive._config_implementation import (
    _RuntimeDefaultsResolver,
)
from models.experts.linear_adaptive.runtime_options import RuntimeOptions


def builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> dict[str, Any]:
    del config_module
    return dict(flat_kwargs)


def _runtime_from_resolver(
    resolver: _RuntimeDefaultsResolver,
) -> RuntimeOptions:
    return RuntimeOptions(
        batch_size=resolver.batch_size,
        learning_rate=resolver.learning_rate,
        input_dim=resolver.input_dim,
        output_dim=resolver.output_dim,
        stack_options=resolver.stack_options,
        submodule_stack_options=resolver.submodule_stack_options,
        mixture_options=resolver.mixture_options,
        expert_stack_options=resolver.expert_stack_options,
        sampler_options=resolver.sampler_options,
        router_options=resolver.router_options,
        router_stack_options=resolver.router_stack_options,
        router_layer_controller_options=resolver.router_layer_controller_options,
        router_dynamic_memory_options=resolver.router_dynamic_memory_options,
        router_recurrent_controller_options=(
            resolver.router_recurrent_controller_options
        ),
        layer_controller_options=resolver.layer_controller_options,
        dynamic_memory_options=resolver.dynamic_memory_options,
        expert_layer_controller_options=resolver.expert_layer_controller_options,
        expert_dynamic_memory_options=resolver.expert_dynamic_memory_options,
        expert_recurrent_controller_options=(
            resolver.expert_recurrent_controller_options
        ),
        adaptive_generator_stack_options=resolver.adaptive_generator_stack_options,
        hidden_adaptive_weight_options=resolver.hidden_adaptive_weight_options,
        hidden_adaptive_bias_options=resolver.hidden_adaptive_bias_options,
        hidden_adaptive_diagonal_options=resolver.hidden_adaptive_diagonal_options,
        hidden_adaptive_mask_options=resolver.hidden_adaptive_mask_options,
        input_boundary_options=resolver.input_boundary_options,
        output_boundary_options=resolver.output_boundary_options,
        router_adaptive_weight_options=resolver.router_adaptive_weight_options,
        router_adaptive_bias_options=resolver.router_adaptive_bias_options,
        router_adaptive_diagonal_options=resolver.router_adaptive_diagonal_options,
        router_adaptive_mask_options=resolver.router_adaptive_mask_options,
        recurrent_controller_options=resolver.recurrent_controller_options,
        _resolved_state=MappingProxyType(dict(vars(resolver))),
    )


def runtime_from_flat(
    flat_kwargs: dict[str, Any] | None = None,
    config_module: ModuleType = config,
) -> RuntimeOptions:
    return _runtime_from_resolver(
        _RuntimeDefaultsResolver(
            **builder_kwargs_from_flat(flat_kwargs or {}, config_module)
        )
    )


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()


__all__ = [
    "DEFAULT_RUNTIME",
    "builder_kwargs_from_flat",
    "runtime_from_flat",
]

from collections.abc import Mapping
from types import ModuleType
from typing import Final

import models.experts.linear_adaptive.config as config
from models.experts.linear_adaptive._config_implementation import (
    _RuntimeDefaultsResolver,
    _RuntimeDefaultValues,
    resolve_runtime_defaults,
)
from models.experts.linear_adaptive.runtime_options import RuntimeOptions


def builder_kwargs_from_flat(
    flat_kwargs: Mapping[str, object],
    config_module: ModuleType,
) -> dict[str, object]:
    del config_module
    return dict(flat_kwargs)


def _runtime_from_resolver(
    resolver: _RuntimeDefaultsResolver,
) -> RuntimeOptions:
    core = resolver.core
    control = resolver.control
    adaptive = resolver.adaptive
    return RuntimeOptions(
        batch_size=core.batch_size,
        learning_rate=core.learning_rate,
        input_dim=core.input_dim,
        output_dim=core.output_dim,
        stack_options=core.stack_options,
        submodule_stack_options=core.submodule_stack_options,
        mixture_options=core.mixture_options,
        expert_stack_options=core.expert_stack_options,
        sampler_options=core.sampler_options,
        router_options=control.router_options,
        router_stack_options=control.router_stack_options,
        router_layer_controller_options=control.router_layer_controller_options,
        router_dynamic_memory_options=control.router_dynamic_memory_options,
        router_recurrent_controller_options=(
            control.router_recurrent_controller_options
        ),
        layer_controller_options=control.layer_controller_options,
        dynamic_memory_options=control.dynamic_memory_options,
        expert_layer_controller_options=control.expert_layer_controller_options,
        expert_dynamic_memory_options=control.expert_dynamic_memory_options,
        expert_recurrent_controller_options=(
            control.expert_recurrent_controller_options
        ),
        adaptive_generator_stack_options=adaptive.adaptive_generator_stack_options,
        hidden_adaptive_weight_options=adaptive.hidden_adaptive_weight_options,
        hidden_adaptive_bias_options=adaptive.hidden_adaptive_bias_options,
        hidden_adaptive_diagonal_options=adaptive.hidden_adaptive_diagonal_options,
        hidden_adaptive_mask_options=adaptive.hidden_adaptive_mask_options,
        input_boundary_options=adaptive.input_boundary_options,
        output_boundary_options=adaptive.output_boundary_options,
        router_adaptive_weight_options=adaptive.router_adaptive_weight_options,
        router_adaptive_bias_options=adaptive.router_adaptive_bias_options,
        router_adaptive_diagonal_options=adaptive.router_adaptive_diagonal_options,
        router_adaptive_mask_options=adaptive.router_adaptive_mask_options,
        recurrent_controller_options=control.recurrent_controller_options,
    )


def runtime_from_flat(
    flat_kwargs: Mapping[str, object] | None = None,
    config_module: ModuleType = config,
) -> RuntimeOptions:
    flat_values = builder_kwargs_from_flat(flat_kwargs or {}, config_module)
    try:
        values = _RuntimeDefaultValues(**flat_values)
    except TypeError as error:
        message = str(error).replace(
            "_RuntimeDefaultValues.__init__",
            "_RuntimeDefaultsResolver.__init__",
        )
        raise TypeError(message) from None
    return _runtime_from_resolver(resolve_runtime_defaults(values))


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()


__all__ = [
    "DEFAULT_RUNTIME",
    "builder_kwargs_from_flat",
    "runtime_from_flat",
]

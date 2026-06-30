from emperor.base.layer.config import LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from models.linears._controller_stack import (
    ControllerStackOptions,
    ControllerStackSource,
    build_controller_stack,
    resolve_controller_stack_options,
    resolve_enabled,
)


def build_linear_controller_stack(
    options: ControllerStackOptions,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    return build_controller_stack(
        options,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        layer_model_config=LinearLayerConfig(
            bias_flag=options.bias_flag,
        ),
    )


__all__ = [
    "ControllerStackOptions",
    "ControllerStackSource",
    "build_linear_controller_stack",
    "resolve_controller_stack_options",
    "resolve_enabled",
]

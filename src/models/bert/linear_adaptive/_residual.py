from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    ResidualConfig,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.linears import LinearLayerConfig

_WEIGHTED_RESIDUAL_CONFIGS = (
    WeightedResidualConfig,
    WeightedBlendResidualConfig,
)


class _SubmoduleStackDefaults(Protocol):
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class ResidualStackSource:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_pipeline_flag: bool | None
    bias_flag: bool | None


@dataclass(frozen=True, slots=True)
class ResidualStackOptions:
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool
    bias_flag: bool


def resolve_residual_stack_options(
    source: ResidualStackSource,
    defaults: _SubmoduleStackDefaults,
) -> ResidualStackOptions:
    """Resolve the residual coefficient stack from submodule-stack defaults."""

    if not source.independent_flag:
        return ResidualStackOptions(
            hidden_dim=defaults.hidden_dim,
            num_layers=defaults.num_layers,
            activation=defaults.activation,
            layer_norm_position=defaults.layer_norm_position,
            residual_connection_option=defaults.residual_connection_option,
            residual_model_flag=source.residual_model_flag,
            dropout_probability=defaults.dropout_probability,
            last_layer_bias_option=defaults.last_layer_bias_option,
            apply_output_pipeline_flag=defaults.apply_output_pipeline_flag,
            bias_flag=defaults.bias_flag,
        )
    return ResidualStackOptions(
        hidden_dim=(
            defaults.hidden_dim if source.hidden_dim is None else source.hidden_dim
        ),
        num_layers=(
            defaults.num_layers if source.num_layers is None else source.num_layers
        ),
        activation=(
            defaults.activation if source.activation is None else source.activation
        ),
        layer_norm_position=(
            defaults.layer_norm_position
            if source.layer_norm_position is None
            else source.layer_norm_position
        ),
        residual_connection_option=(
            defaults.residual_connection_option
            if source.residual_connection_option is None
            else source.residual_connection_option
        ),
        residual_model_flag=source.residual_model_flag,
        dropout_probability=(
            defaults.dropout_probability
            if source.dropout_probability is None
            else source.dropout_probability
        ),
        last_layer_bias_option=(
            defaults.last_layer_bias_option
            if source.last_layer_bias_option is None
            else source.last_layer_bias_option
        ),
        apply_output_pipeline_flag=(
            defaults.apply_output_pipeline_flag
            if source.apply_output_pipeline_flag is None
            else source.apply_output_pipeline_flag
        ),
        bias_flag=defaults.bias_flag if source.bias_flag is None else source.bias_flag,
    )


def build_residual_stack_config(
    options: ResidualStackOptions,
) -> LayerStackConfig:
    """Build the coefficient stack owned by a residual."""

    if options.residual_model_flag:
        raise ValueError(
            "RESIDUAL_STACK_RESIDUAL_MODEL_FLAG=True with "
            "RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION would recursively "
            "define the residual stack in itself."
        )

    return LayerStackConfig(
        hidden_dim=options.hidden_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_pipeline_flag=options.apply_output_pipeline_flag,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            residual_config=build_residual_config(
                options.residual_connection_option,
                False,
                selector_field="RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION",
                model_flag_field="RESIDUAL_STACK_RESIDUAL_MODEL_FLAG",
            ),
            dropout_probability=options.dropout_probability,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
        ),
    )


def build_residual_config(
    residual_connection_option: type[ResidualConfig] | None,
    residual_model_flag: bool,
    residual_stack_options: ResidualStackOptions | None = None,
    *,
    selector_field: str = "residual_connection_option",
    model_flag_field: str = "residual_model_flag",
) -> ResidualConfig | None:
    """Build one package-owned residual config from a selector/flag pair."""

    if type(residual_model_flag) is not bool:
        raise TypeError(f"{model_flag_field} must be bool.")
    if residual_connection_option is None:
        if residual_model_flag:
            _raise_incompatible_selector(
                residual_connection_option,
                selector_field=selector_field,
                model_flag_field=model_flag_field,
            )
        return None
    if not isinstance(residual_connection_option, type) or not issubclass(
        residual_connection_option,
        ResidualConfig,
    ):
        raise TypeError(
            f"{selector_field} must be a ResidualConfig type or None; "
            f"received {type(residual_connection_option).__name__}."
        )
    if not residual_model_flag:
        return residual_connection_option()
    if not issubclass(residual_connection_option, _WEIGHTED_RESIDUAL_CONFIGS):
        _raise_incompatible_selector(
            residual_connection_option,
            selector_field=selector_field,
            model_flag_field=model_flag_field,
        )
    if residual_stack_options is None:
        raise ValueError(
            f"{model_flag_field}=True with {selector_field} requires resolved "
            "RESIDUAL_STACK options."
        )
    return residual_connection_option(
        model_config=build_residual_stack_config(
            residual_stack_options,
        ),
    )


def _raise_incompatible_selector(
    residual_connection_option: type[ResidualConfig] | None,
    *,
    selector_field: str,
    model_flag_field: str,
) -> None:
    selected = (
        "None"
        if residual_connection_option is None
        else getattr(
            residual_connection_option,
            "__name__",
            type(residual_connection_option).__name__,
        )
    )
    raise ValueError(
        f"{model_flag_field}=True requires {selector_field} to select "
        "WeightedResidualConfig or WeightedBlendResidualConfig; "
        f"received {selected}."
    )

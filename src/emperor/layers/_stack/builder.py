from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import cast

from emperor.config import ConfigBase
from emperor.layers._config import LayerConfig, LayerStackConfig
from emperor.layers._layer import Layer
from emperor.layers._options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)


class LayerStackBuilder:
    """Build configured Layers from an ordered dimension plan."""

    def __init__(
        self,
        cfg: LayerStackConfig,
        supports_rectangular_gate: bool,
    ) -> None:
        self.layer_config = cfg.layer_config
        self.apply_output_postprocessing_flag = cfg.apply_output_postprocessing_flag
        self.last_layer_bias_option = cfg.last_layer_bias_option
        self.supports_rectangular_gate = supports_rectangular_gate

    def build_layer_stack(
        self,
        dimensions: Sequence[tuple[int, int]],
    ) -> list[Layer]:
        layers: list[Layer] = []
        layer_count = len(dimensions)
        for layer_number, (input_dim, output_dim) in enumerate(dimensions, start=1):
            is_last_layer = layer_number == layer_count
            stack_layer = self.__build_layer(
                input_dim,
                output_dim,
                is_last_layer=is_last_layer,
            )
            layers.append(stack_layer)
        return layers

    def __build_layer(
        self,
        input_dim: int,
        output_dim: int,
        is_last_layer: bool = False,
    ) -> Layer:
        has_stable_dimension = input_dim == output_dim
        resolved_layer_config = self.__resolve_layer_config(
            input_dim, output_dim, is_last_layer
        )
        self.__apply_compatibility_overrides(
            resolved_layer_config, is_last_layer, has_stable_dimension
        )
        layer = resolved_layer_config.build()
        if is_last_layer:
            layer._mark_as_last_layer()  # pyright: ignore[reportPrivateUsage]
        return layer

    def __resolve_layer_config(
        self,
        input_dim: int,
        output_dim: int,
        is_last_layer: bool,
    ) -> LayerConfig:
        dimension_overrides = self.__resolve_config_overrides(
            self.layer_config, input_dim, output_dim
        )
        if is_last_layer:
            output_layer_overrides = self.__resolve_output_layer_overrides()
            dimension_overrides = self.__override_config(
                dimension_overrides, output_layer_overrides
            )
        resolved_layer_config = self.__override_config(
            self.layer_config, dimension_overrides
        )
        return resolved_layer_config

    def __resolve_output_layer_overrides(self) -> LayerConfig | None:
        output_layer_overrides: LayerConfig | None = None
        if not self.apply_output_postprocessing_flag:
            output_layer_overrides = LayerConfig(
                activation=ActivationOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
            )
        last_layer_bias_override = self.__resolve_last_layer_bias_override()
        return self.__merge_layer_override(
            output_layer_overrides, last_layer_bias_override
        )

    def __resolve_last_layer_bias_override(self) -> LayerConfig | None:
        if self.last_layer_bias_option == LastLayerBiasOptions.DEFAULT:
            return None
        if not hasattr(self.layer_config.layer_model_config, "bias_flag"):
            return None

        last_layer_model_config = deepcopy(self.layer_config.layer_model_config)
        match self.last_layer_bias_option:
            case LastLayerBiasOptions.DISABLED:
                last_layer_model_config.bias_flag = False
            case LastLayerBiasOptions.ENABLED:
                last_layer_model_config.bias_flag = True
            case _:
                raise ValueError(
                    "Unsupported last layer bias option "
                    f"{self.last_layer_bias_option} for LayerStack."
                )
        return LayerConfig(
            layer_model_config=cast(ConfigBase, last_layer_model_config),
        )

    @staticmethod
    def __merge_layer_override(
        base_override: LayerConfig | None,
        additional_override: LayerConfig | None,
    ) -> LayerConfig | None:
        if additional_override is None:
            return base_override
        if base_override is None:
            return additional_override
        return LayerStackBuilder.__override_config(
            base_override,
            additional_override,
        )

    def __apply_compatibility_overrides(
        self,
        resolved_layer_config: LayerConfig,
        is_last_layer: bool,
        has_stable_dimension: bool,
    ) -> None:
        if self.__should_disable_residual(is_last_layer, has_stable_dimension):
            resolved_layer_config.residual_config = None
        if not has_stable_dimension and not self.supports_rectangular_gate:
            resolved_layer_config.gate_config = None

    def __should_disable_residual(
        self,
        is_last_layer: bool,
        has_stable_dimension: bool,
    ) -> bool:
        output_layer_postprocessing_is_disabled = (
            is_last_layer and not self.apply_output_postprocessing_flag
        )
        layer_dimensions_do_not_support_residual = not has_stable_dimension
        return (
            output_layer_postprocessing_is_disabled
            or layer_dimensions_do_not_support_residual
        )

    @staticmethod
    def __resolve_config_overrides(
        config: LayerConfig,
        input_dim: int,
        output_dim: int,
    ) -> LayerConfig:
        return type(config)(input_dim=input_dim, output_dim=output_dim)

    @staticmethod
    def __override_config(
        config: LayerConfig,
        overrides: LayerConfig | None = None,
    ) -> LayerConfig:
        if overrides is None:
            return config
        overridden_config = deepcopy(config)
        return overridden_config.update(overrides)

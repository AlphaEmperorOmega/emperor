from __future__ import annotations

from torch import Tensor


class _WeightedResidualValidationMixin:
    @classmethod
    def _validate_weighted_config(cls, config: object) -> None:
        model_config = config.model_config
        if model_config is None:
            return
        from emperor.layers import (
            LastLayerBiasOptions,
            LayerConfig,
            LayerStackConfig,
        )
        from emperor.linears import LinearLayerConfig

        if isinstance(model_config, LinearLayerConfig):
            if model_config.bias_flag is not True:
                raise ValueError(
                    f"{type(config).__name__}.model_config.bias_flag must be True "
                    "so the initial mixing coefficient can be represented."
                )
        elif isinstance(model_config, LayerStackConfig):
            layer_config = model_config.layer_config
            if type(layer_config) is not LayerConfig:
                raise TypeError(
                    f"{type(config).__name__}.model_config.layer_config must be "
                    f"exactly LayerConfig, got {type(layer_config).__name__}."
                )
            if not isinstance(layer_config.layer_model_config, LinearLayerConfig):
                raise TypeError(
                    f"{type(config).__name__}.model_config.layer_config."
                    "layer_model_config must be LinearLayerConfig, got "
                    f"{type(layer_config.layer_model_config).__name__}."
                )
            final_bias_enabled = (
                model_config.last_layer_bias_option == LastLayerBiasOptions.ENABLED
                or (
                    model_config.last_layer_bias_option == LastLayerBiasOptions.DEFAULT
                    and layer_config.layer_model_config.bias_flag is True
                )
            )
            if not final_bias_enabled:
                raise ValueError(
                    f"{type(config).__name__}.model_config must enable bias on "
                    "its final layer so the initial mixing coefficient can be "
                    "represented."
                )
            nested_configs = {
                "layer_config.gate_config": layer_config.gate_config,
                "layer_config.halting_config": layer_config.halting_config,
                "layer_config.memory_config": layer_config.memory_config,
                "shared_gate_config": model_config.shared_gate_config,
                "shared_halting_config": model_config.shared_halting_config,
                "shared_memory_config": model_config.shared_memory_config,
            }
            for path, nested_config in nested_configs.items():
                if nested_config is not None:
                    raise ValueError(
                        f"{type(config).__name__}.model_config.{path} must be None "
                        "for a residual coefficient model."
                    )
        else:
            raise TypeError(
                f"{type(config).__name__}.model_config must be a "
                "LayerStackConfig or LinearLayerConfig when provided, "
                f"got {type(model_config).__name__}."
            )
        if config.residual_dim is None:
            raise TypeError(
                f"residual_dim must be int for {type(config).__name__} when "
                "model_config is provided, got NoneType."
            )

    @staticmethod
    def validate_raw_mix_coefficient(raw_mix_coefficient: Tensor | None) -> None:
        if raw_mix_coefficient is None:
            raise RuntimeError(
                "weighted residual requires either raw_weight or a coefficient model."
            )

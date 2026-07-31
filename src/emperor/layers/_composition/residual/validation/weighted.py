from __future__ import annotations

from torch import Tensor


class _WeightedResidualValidationMixin:
    @classmethod
    def _validate_weighted_config(cls, config: object) -> None:
        model_config = config.model_config
        if model_config is None:
            return
        from emperor.linears import LinearLayerConfig

        if not isinstance(model_config, LinearLayerConfig):
            raise TypeError(
                f"{type(config).__name__}.model_config must be a "
                "LinearLayerConfig when provided, "
                f"got {type(model_config).__name__}."
            )
        if model_config.bias_flag is not True:
            raise ValueError(
                f"{type(config).__name__}.model_config.bias_flag must be True so "
                "the initial mixing coefficient can be represented."
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

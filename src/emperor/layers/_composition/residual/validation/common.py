from __future__ import annotations

from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.layers._composition.residual.validation.attention import (
    _AttentionResidualValidationMixin,
)
from emperor.layers._composition.residual.validation.weighted import (
    _WeightedResidualValidationMixin,
)

if TYPE_CHECKING:
    from emperor.layers._composition.residual.base import (
        ResidualConnectionAbstract,
    )
    from emperor.layers._composition.residual.config import ResidualConfig


class ResidualConnectionValidator(
    _AttentionResidualValidationMixin,
    _WeightedResidualValidationMixin,
    ValidatorBase,
):
    @classmethod
    def validate(cls, model: ResidualConnectionAbstract) -> None:
        from emperor.layers._composition.residual.config import (
            AttentionResidualConfig,
            ResidualConfig,
            WeightedBlendResidualConfig,
            WeightedResidualConfig,
        )

        config = model.cfg
        if not isinstance(config, ResidualConfig):
            raise TypeError(
                "residual connection cfg must be a ResidualConfig, "
                f"got {type(config).__name__}."
            )
        cls._validate_concrete_config(config, owner_name=type(model).__name__)
        expected_owner = config.registry_owner()
        if not isinstance(model, expected_owner):
            raise TypeError(
                f"{type(config).__name__} builds {expected_owner.__name__}, not "
                f"{type(model).__name__}."
            )
        if isinstance(config, AttentionResidualConfig):
            cls._validate_attention_config(config)
        else:
            cls._validate_optional_residual_dim(config.residual_dim)
        if isinstance(
            config,
            (WeightedResidualConfig, WeightedBlendResidualConfig),
        ):
            cls._validate_weighted_config(config)

    @classmethod
    def validate_residual_config(
        cls,
        residual_config: ResidualConfig | None,
        owner_name: str,
    ) -> None:
        if residual_config is None:
            return
        from emperor.layers._composition.residual.config import ResidualConfig

        if not isinstance(residual_config, ResidualConfig):
            raise TypeError(
                "residual_config must be an instance of ResidualConfig for "
                f"{owner_name}, got {type(residual_config).__name__}"
            )
        cls._validate_concrete_config(residual_config, owner_name=owner_name)

    @staticmethod
    def _validate_concrete_config(
        residual_config: ResidualConfig,
        *,
        owner_name: str,
    ) -> None:
        try:
            residual_config.registry_owner()
        except (NotImplementedError, ValueError) as exc:
            raise ValueError(
                f"residual_config must be a concrete residual config for {owner_name}"
            ) from exc

    @staticmethod
    def _validate_optional_residual_dim(residual_dim: int | None) -> None:
        if residual_dim is None:
            return
        if isinstance(residual_dim, bool) or not isinstance(residual_dim, int):
            raise TypeError(
                "residual_dim must be int for a residual config, "
                f"got {type(residual_dim).__name__}."
            )
        if residual_dim <= 0:
            raise ValueError(
                f"residual_dim must be greater than 0, received {residual_dim}"
            )

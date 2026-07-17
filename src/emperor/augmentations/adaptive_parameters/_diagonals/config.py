from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.layers import LayerStackConfig


@dataclass
class DynamicDiagonalConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    model_config: "LayerStackConfig | None" = optional_field(
        "Internal generator network config."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            "DynamicDiagonalConfig is abstract and has no registered "
            "DynamicDiagonal class; instantiate a concrete leaf config instead."
        )


@dataclass
class StandardDynamicDiagonalConfig(DynamicDiagonalConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._diagonals.variants.standard import (
            StandardDynamicDiagonal,
        )

        return StandardDynamicDiagonal


@dataclass
class AntiDynamicDiagonalConfig(DynamicDiagonalConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._diagonals.variants.anti import (
            AntiDynamicDiagonal,
        )

        return AntiDynamicDiagonal


@dataclass
class CombinedDynamicDiagonalConfig(DynamicDiagonalConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._diagonals.variants.combined import (
            CombinedDynamicDiagonal,
        )

        return CombinedDynamicDiagonal

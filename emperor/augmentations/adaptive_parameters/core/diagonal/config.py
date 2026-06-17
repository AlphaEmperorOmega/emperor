from dataclasses import dataclass
from emperor.base.layer import LayerStackConfig
from emperor.base.utils import ConfigBase, optional_field


@dataclass
class DynamicDiagonalConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    model_config: LayerStackConfig | None = optional_field(
        "Internal generator network config."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            f"DynamicDiagonalConfig is abstract and has no registered "
            f"DynamicDiagonal class; instantiate a concrete leaf config instead."
        )


@dataclass
class StandardDynamicDiagonalConfig(DynamicDiagonalConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.diagonal.variants.standard import (
            StandardDynamicDiagonal,
        )

        return StandardDynamicDiagonal


@dataclass
class AntiDynamicDiagonalConfig(DynamicDiagonalConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.diagonal.variants.anti import (
            AntiDynamicDiagonal,
        )

        return AntiDynamicDiagonal


@dataclass
class CombinedDynamicDiagonalConfig(DynamicDiagonalConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.diagonal.variants.combined import (
            CombinedDynamicDiagonal,
        )

        return CombinedDynamicDiagonal

from dataclasses import dataclass

from emperor.augmentations.adaptive_parameters._options import (
    BankExpansionFactorOptions,
    WeightDecayScheduleOptions,
)
from emperor.config import ConfigBase, optional_field
from emperor.layers import LayerStackConfig


@dataclass
class DynamicBiasConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    decay_schedule: WeightDecayScheduleOptions | None = optional_field(
        "Base bias decay schedule."
    )
    decay_rate: float | None = optional_field("Decay rate for the selected schedule.")
    decay_warmup_batches: int | None = optional_field(
        "Warmup batches before bias decay starts."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Internal generator network config."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            "DynamicBiasConfig is abstract and has no registered "
            "DynamicBias class; instantiate a concrete leaf config instead."
        )


@dataclass
class AffineTransformDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._biases.affine import (
            AffineTransformDynamicBias,
        )

        return AffineTransformDynamicBias


@dataclass
class AdditiveDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._biases.additive import (
            AdditiveDynamicBias,
        )

        return AdditiveDynamicBias


@dataclass
class MultiplicativeDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._biases.multiplicative import (
            MultiplicativeDynamicBias,
        )

        return MultiplicativeDynamicBias


@dataclass
class SigmoidGatedDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._biases.gated import (
            SigmoidGatedDynamicBias,
        )

        return SigmoidGatedDynamicBias


@dataclass
class TanhGatedDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._biases.gated import (
            TanhGatedDynamicBias,
        )

        return TanhGatedDynamicBias


@dataclass
class GeneratorDynamicBiasConfig(DynamicBiasConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._biases.generator import (
            GeneratorDynamicBias,
        )

        return GeneratorDynamicBias


@dataclass
class WeightedBankDynamicBiasConfig(DynamicBiasConfig):
    bank_expansion_factor: BankExpansionFactorOptions | None = optional_field(
        "Bias bank expansion factor."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._biases.weighted_bank import (
            WeightedBankDynamicBias,
        )

        return WeightedBankDynamicBias

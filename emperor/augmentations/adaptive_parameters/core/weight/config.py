from dataclasses import dataclass
from emperor.base.layer import LayerStackConfig
from emperor.base.utils import ConfigBase, optional_field
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)


@dataclass
class DynamicWeightConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    generator_depth: DynamicDepthOptions | None = optional_field(
        "Generator depth for dynamic weights."
    )
    decay_schedule: WeightDecayScheduleOptions | None = optional_field(
        "Base weight decay schedule."
    )
    decay_rate: float | None = optional_field("Decay rate for the selected schedule.")
    decay_warmup_batches: int | None = optional_field(
        "Warmup batches before weight decay starts."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Internal generator network config."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            f"DynamicWeightConfig is abstract and has no registered "
            f"DynamicWeight class; instantiate a concrete leaf config instead."
        )


@dataclass
class SingleModelDynamicWeightConfig(DynamicWeightConfig):
    normalization_option: WeightNormalizationOptions | None = optional_field(
        "Dynamic weight normalization method."
    )
    normalization_position_option: WeightNormalizationPositionOptions | None = (
        optional_field("Where dynamic weight normalization is applied.")
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.weight.variants.single_model import (
            SingleModelDynamicWeight,
        )

        return SingleModelDynamicWeight


@dataclass
class DualModelDynamicWeightConfig(DynamicWeightConfig):
    normalization_option: WeightNormalizationOptions | None = optional_field(
        "Dynamic weight normalization method."
    )
    normalization_position_option: WeightNormalizationPositionOptions | None = (
        optional_field("Where dynamic weight normalization is applied.")
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.weight.variants.dual_model import (
            DualModelDynamicWeight,
        )

        return DualModelDynamicWeight


@dataclass
class LowRankDynamicWeightConfig(DynamicWeightConfig):
    normalization_option: WeightNormalizationOptions | None = optional_field(
        "Dynamic weight normalization method."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.weight.variants.low_rank import (
            LowRankDynamicWeight,
        )

        return LowRankDynamicWeight


@dataclass
class HypernetworkDynamicWeightConfig(DynamicWeightConfig):
    normalization_option: WeightNormalizationOptions | None = optional_field(
        "Dynamic weight normalization method."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.weight.variants.hypernetwork import (
            HypernetworkDynamicWeight,
        )

        return HypernetworkDynamicWeight


@dataclass
class LayeredWeightedBankDynamicWeightConfig(DynamicWeightConfig):
    bank_expansion_factor: BankExpansionFactorOptions | None = optional_field(
        "Weight bank expansion factor."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.weight.variants.layered_weighted_bank import (
            LayeredWeightedBankDynamicWeight,
        )

        return LayeredWeightedBankDynamicWeight


@dataclass
class SoftWeightedBankDynamicWeightConfig(DynamicWeightConfig):
    bank_expansion_factor: BankExpansionFactorOptions | None = optional_field(
        "Weight bank expansion factor."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.weight.variants.soft_weighted_bank import (
            SoftWeightedBankDynamicWeight,
        )

        return SoftWeightedBankDynamicWeight

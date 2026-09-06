from dataclasses import dataclass

from emperor.augmentations.adaptive_parameters._grouping.options import (
    SummaryNormalizationOptions,
)
from emperor.augmentations.adaptive_parameters._options import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
)
from emperor.config import ConfigBase, optional_field


@dataclass(kw_only=True)
class GroupingConfig(ConfigBase):
    """Split flat token rows into groups and summarize each group."""

    scope: AdaptiveParameterGroupingScopeOptions | None = optional_field(
        "Grouping scope. Required when grouping is configured."
    )
    group_count: int | None = optional_field(
        "Number of groups across rows or within each sequence."
    )
    chunk_size: int | None = optional_field(
        "Maximum real tokens per chunk and internally padded chunk width; "
        "mutually exclusive with group_count."
    )
    sequence_length: int | None = optional_field(
        "Physical sequence length. Required for SEQUENCE grouping only."
    )
    input_order: AdaptiveParameterInputOrderOptions | None = optional_field(
        "Input order before flattening. Required for SEQUENCE grouping only."
    )
    feature_dim: int | None = optional_field("Group member feature dimension.")
    summary_normalization: SummaryNormalizationOptions | None = optional_field(
        "Normalization after grouping. None resolves to DISABLED."
    )
    rms_norm_epsilon: float | None = optional_field(
        "Positive RMS_NORM epsilon. None resolves to 1e-6 when selected."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            "GroupingConfig is abstract; instantiate a concrete "
            "grouping variant config instead."
        )


@dataclass(kw_only=True)
class SumGroupingConfig(GroupingConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._grouping.variants.sum import (
            SumGrouper,
        )

        return SumGrouper


@dataclass(kw_only=True)
class MeanGroupingConfig(GroupingConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._grouping.variants.mean import (
            MeanGrouper,
        )

        return MeanGrouper


@dataclass(kw_only=True)
class MeanStdGroupingConfig(GroupingConfig):
    model_config: ConfigBase | None = optional_field(
        "Required statistics projection model config: 2 * feature_dim inputs to feature_dim outputs."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._grouping.variants.mean_std import (
            MeanStdGrouper,
        )

        return MeanStdGrouper


@dataclass(kw_only=True)
class AttentionGroupingConfig(GroupingConfig):
    model_config: ConfigBase | None = optional_field(
        "Required token-scoring model config: feature_dim inputs to one score per token."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._grouping.variants.attention import (
            AttentionGrouper,
        )

        return AttentionGrouper


@dataclass(kw_only=True)
class RMSGroupingConfig(GroupingConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters._grouping.variants.rms import (
            RMSGrouper,
        )

        return RMSGrouper

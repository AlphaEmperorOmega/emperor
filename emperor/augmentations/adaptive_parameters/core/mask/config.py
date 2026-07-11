from dataclasses import dataclass
from emperor.base.layer import LayerStackConfig
from emperor.base.config import ConfigBase, optional_field
from emperor.augmentations.adaptive_parameters.options import MaskDimensionOptions


@dataclass
class AxisMaskConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    mask_threshold: float | None = optional_field(
        "Threshold for keeping rows or columns."
    )
    mask_surrogate_scale: float | None = optional_field(
        "Training-time mask surrogate scale. Use 0.0 to disable."
    )
    mask_floor: float | None = optional_field("Minimum value for dropped mask regions.")
    model_config: LayerStackConfig | None = optional_field(
        "Internal generator network config."
    )

    def _registry_owner(self) -> type:
        raise ValueError(
            f"AxisMaskConfig is abstract and has no registered "
            f"AxisMask class; instantiate a concrete leaf config instead."
        )


@dataclass
class WeightInformedScoreAxisMaskConfig(AxisMaskConfig):
    mask_dimension_option: MaskDimensionOptions | None = optional_field(
        "Mask rows or columns."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.mask.variants.weight_informed import (
            WeightInformedScoreAxisMask,
        )

        return WeightInformedScoreAxisMask


@dataclass
class PerAxisScoreMaskConfig(AxisMaskConfig):
    mask_dimension_option: MaskDimensionOptions | None = optional_field(
        "Mask rows or columns."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.mask.variants.per_axis import (
            PerAxisScoreMask,
        )

        return PerAxisScoreMask


@dataclass
class TopSliceAxisMaskConfig(AxisMaskConfig):
    mask_dimension_option: MaskDimensionOptions | None = optional_field(
        "Mask rows or columns."
    )
    mask_transition_width: float | None = optional_field(
        "Smooth transition width for top-slice masking."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.mask.variants.top_slice import (
            TopSliceAxisMask,
        )

        return TopSliceAxisMask


@dataclass
class OuterProductMaskConfig(AxisMaskConfig):
    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.mask.variants.outer_product import (
            OuterProductMask,
        )

        return OuterProductMask


@dataclass
class DiagonalAxisMaskConfig(AxisMaskConfig):
    mask_transition_width: float | None = optional_field(
        "Smooth transition width for diagonal masking."
    )

    def _registry_owner(self) -> type:
        from emperor.augmentations.adaptive_parameters.core.mask.variants.diagonal import (
            DiagonalAxisMask,
        )

        return DiagonalAxisMask

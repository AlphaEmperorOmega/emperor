from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    DynamicWeightOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
    MaskDimensionOptions,
    RowMaskOptions,
    WeightNormalizationOptions,
)

@dataclass
class AdaptiveParameterAugmentationConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the linear layer"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the linear layer"},
    )
    weight_option: DynamicWeightOptions | None = field(
        default=None,
        metadata={
            "help": "Selects the weight handler type for input-dependent weight adjustments."
        },
    )
    weight_normalization: WeightNormalizationOptions | None = field(
        default=None,
        metadata={
            "help": "Normalization applied to vectors before the outer product computation."
        },
    )
    generator_depth: DynamicDepthOptions | None = field(
        default=None,
        metadata={
            "help": "Depth of the generator network that produces input-dependent weight adjustments."
        },
    )
    diagonal_option: DynamicDiagonalOptions | None = field(
        default=None,
        metadata={"help": "Input-dependent adjustment of the weight matrix diagonal."},
    )
    bias_flag: bool | None = field(
        default=None,
        metadata={"help": "Whether the linear layer has a bias parameter."},
    )
    bias_option: DynamicBiasOptions | None = field(
        default=None,
        metadata={"help": "Input-dependent adjustment of the bias vector."},
    )
    bias_bank_size: int | None = field(
        default=None,
        metadata={"help": "Size of the weight bank for WEIGHTED_BANK bias option."},
    )
    weight_bank_size: int | None = field(
        default=None,
        metadata={"help": "Size of the weight bank for WEIGHTED_BANK weight option."},
    )
    row_mask_option: RowMaskOptions | None = field(
        default=None,
        metadata={
            "help": "Input-dependent row masking of the weight matrix after weight updates."
        },
    )
    mask_dimension_option: MaskDimensionOptions | None = field(
        default=None,
        metadata={"help": "Whether to mask rows or columns of the weight matrix."},
    )
    memory_option: LinearMemoryOptions | None = field(
        default=None,
        metadata={
            "help": "Blends a learned memory representation with the linear layer input or output."
        },
    )
    memory_size_option: LinearMemorySizeOptions | None = field(
        default=None,
        metadata={"help": "Size of the learned memory representation."},
    )
    memory_position_option: LinearMemoryPositionOptions | None = field(
        default=None,
        metadata={"help": "Controls when memory is applied in the computation."},
    )

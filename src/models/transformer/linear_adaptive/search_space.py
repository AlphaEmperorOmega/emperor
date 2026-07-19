from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
    DiagonalAxisMaskConfig,
    DualModelDynamicWeightConfig,
    GeneratorDynamicBiasConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MultiplicativeDynamicBiasConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    SigmoidGatedDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TanhGatedDynamicBiasConfig,
    TopSliceAxisMaskConfig,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.layers import LayerNormPositionOptions

SEARCH_SPACE_LEARNING_RATE = [0.5, 1.0, 2.0]
SEARCH_SPACE_MODEL_DIM = [64, 128, 256]
SEARCH_SPACE_ENCODER_NUM_LAYERS = [2, 3, 4]
SEARCH_SPACE_DECODER_NUM_LAYERS = [2, 3, 4]
SEARCH_SPACE_ATTN_NUM_HEADS = [2, 4, 8]
SEARCH_SPACE_FF_STACK_HIDDEN_DIM = [256, 512, 1024]
SEARCH_SPACE_ENCODER_LAYER_NORM_POSITION = [
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]
SEARCH_SPACE_DECODER_LAYER_NORM_POSITION = SEARCH_SPACE_ENCODER_LAYER_NORM_POSITION
SEARCH_SPACE_WEIGHT_OPTION = [
    None,
    SingleModelDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
]
SEARCH_SPACE_BIAS_OPTION = [
    None,
    AffineTransformDynamicBiasConfig,
    AdditiveDynamicBiasConfig,
    GeneratorDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
]
SEARCH_SPACE_DIAGONAL_OPTION = [
    None,
    StandardDynamicDiagonalConfig,
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
]
SEARCH_SPACE_ROW_MASK_OPTION = [
    None,
    DiagonalAxisMaskConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
]

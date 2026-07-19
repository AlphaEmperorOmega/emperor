from .config import *  # noqa: F401,F403

SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM: list = [16, 32, 64, 128, 256, 512]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [2, 4, 8, 16, 32]
SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEARCH_SPACE_STACK_LAYER_NORM_POSITION: list = [
    LayerNormPositionOptions.DISABLED,
    LayerNormPositionOptions.DEFAULT,
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]
SEARCH_SPACE_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.LEAKY_RELU,
    ActivationOptions.ELU,
    ActivationOptions.GELU,
    ActivationOptions.TANH,
]
SEARCH_SPACE_WEIGHT_OPTION: list = [
    None,
    SingleModelDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
]
SEARCH_SPACE_WEIGHT_GENERATOR_DEPTH: list = [
    DynamicDepthOptions.DEPTH_OF_ONE,
    DynamicDepthOptions.DEPTH_OF_TWO,
    DynamicDepthOptions.DEPTH_OF_FOUR,
    DynamicDepthOptions.DEPTH_OF_SIX,
    DynamicDepthOptions.DEPTH_OF_EIGHT,
    DynamicDepthOptions.DEPTH_OF_TEN,
]
SEARCH_SPACE_WEIGHT_DECAY_SCHEDULE: list = [
    WeightDecayScheduleOptions.DISABLED,
    WeightDecayScheduleOptions.EXPONENTIAL,
    WeightDecayScheduleOptions.LINEAR,
    WeightDecayScheduleOptions.MULTIPLICATIVE,
]
SEARCH_SPACE_WEIGHT_DECAY_RATE: list = [1e-5, 1e-4, 1e-3, 1e-2]
SEARCH_SPACE_WEIGHT_DECAY_WARMUP_BATCHES: list = [0, 100, 500, 1000]
SEARCH_SPACE_WEIGHT_NORMALIZATION_OPTION: list = [
    WeightNormalizationOptions.DISABLED,
    WeightNormalizationOptions.CLAMP,
    WeightNormalizationOptions.L2_SCALE,
    WeightNormalizationOptions.SOFT_CLAMP,
    WeightNormalizationOptions.RMS,
    WeightNormalizationOptions.SIGMOID_SCALE,
]
SEARCH_SPACE_WEIGHT_NORMALIZATION_POSITION_OPTION: list = [
    WeightNormalizationPositionOptions.DISABLED,
    WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
    WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
]
SEARCH_SPACE_WEIGHT_BANK_EXPANSION_FACTOR: list = [
    BankExpansionFactorOptions.FACTOR_OF_ONE,
    BankExpansionFactorOptions.FACTOR_OF_TWO,
    BankExpansionFactorOptions.FACTOR_OF_THREE,
    BankExpansionFactorOptions.FACTOR_OF_FOUR,
]
SEARCH_SPACE_BIAS_OPTION: list = [
    None,
    AffineTransformDynamicBiasConfig,
    AdditiveDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
]
SEARCH_SPACE_BIAS_DECAY_SCHEDULE: list = [
    WeightDecayScheduleOptions.DISABLED,
    WeightDecayScheduleOptions.EXPONENTIAL,
    WeightDecayScheduleOptions.LINEAR,
    WeightDecayScheduleOptions.MULTIPLICATIVE,
]
SEARCH_SPACE_BIAS_DECAY_RATE: list = [1e-5, 1e-4, 1e-3, 1e-2]
SEARCH_SPACE_BIAS_DECAY_WARMUP_BATCHES: list = [0, 100, 500, 1000]
SEARCH_SPACE_BIAS_BANK_EXPANSION_FACTOR: list = [
    BankExpansionFactorOptions.FACTOR_OF_ONE,
    BankExpansionFactorOptions.FACTOR_OF_TWO,
    BankExpansionFactorOptions.FACTOR_OF_THREE,
    BankExpansionFactorOptions.FACTOR_OF_FOUR,
]
SEARCH_SPACE_DIAGONAL_OPTION: list = [
    None,
    StandardDynamicDiagonalConfig,
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
]
SEARCH_SPACE_ROW_MASK_OPTION: list = [
    None,
    DiagonalAxisMaskConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
]
SEARCH_SPACE_MASK_THRESHOLD: list = [0.1, 0.3, 0.5, 0.7, 0.9]
SEARCH_SPACE_MASK_SURROGATE_SCALE: list = [1.0, 5.0, 10.0, 20.0]
SEARCH_SPACE_MASK_FLOOR: list = [0.0, 0.1, 0.25, 0.5]
SEARCH_SPACE_MASK_TRANSITION_WIDTH: list = [0.05, 0.1, 0.2, 0.5]
SEARCH_SPACE_MASK_DIMENSION_OPTION: list = [
    MaskDimensionOptions.ROW,
    MaskDimensionOptions.COLUMN,
]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: list = [1, 2, 3]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: list = [64, 128, 256]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.SILU,
    ActivationOptions.GELU,
    ActivationOptions.MISH,
]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: list = [
    LayerNormPositionOptions.DISABLED,
    LayerNormPositionOptions.DEFAULT,
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]

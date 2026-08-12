from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    AdditiveDynamicBiasConfig,  # noqa: F401
    AffineTransformDynamicBiasConfig,  # noqa: F401
    AntiDynamicDiagonalConfig,  # noqa: F401
    AxisMaskConfig,
    BankExpansionFactorOptions,
    CombinedDynamicDiagonalConfig,  # noqa: F401
    DiagonalAxisMaskConfig,  # noqa: F401
    DualModelDynamicWeightConfig,  # noqa: F401
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GeneratorDynamicBiasConfig,  # noqa: F401
    HypernetworkDynamicWeightConfig,  # noqa: F401
    LayeredWeightedBankDynamicWeightConfig,  # noqa: F401
    LowRankDynamicWeightConfig,  # noqa: F401
    MaskDimensionOptions,
    MultiplicativeDynamicBiasConfig,  # noqa: F401
    OuterProductMaskConfig,  # noqa: F401
    PerAxisScoreMaskConfig,  # noqa: F401
    SigmoidGatedDynamicBiasConfig,  # noqa: F401
    SingleModelDynamicWeightConfig,  # noqa: F401
    SoftWeightedBankDynamicWeightConfig,  # noqa: F401
    StandardDynamicDiagonalConfig,  # noqa: F401
    TanhGatedDynamicBiasConfig,  # noqa: F401
    TopSliceAxisMaskConfig,  # noqa: F401
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,  # noqa: F401
    WeightInformedScoreAxisMaskConfig,  # noqa: F401
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.embedding.absolute import (
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,  # noqa: F401
    AttentionResidualConfig,  # noqa: F401
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConfig,
    WeightedBlendResidualConfig,  # noqa: F401
    WeightedResidualConfig,  # noqa: F401
)
from emperor.memory import (
    AttentionDynamicMemoryConfig,  # noqa: F401
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,  # noqa: F401
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,  # noqa: F401
)

# Global
BATCH_SIZE = 64
LEARNING_RATE = 1.0
VOCAB_SIZE = 8192
MODEL_DIM = 128
SOURCE_SEQUENCE_LENGTH = 64
TARGET_SEQUENCE_LENGTH = 64
SEQUENCE_LENGTH = 64
DROPOUT_PROBABILITY = 0.1
POSITIONAL_EMBEDDING_OPTION = TextSinusoidalPositionalEmbeddingConfig

# Attention Options
ATTN_NUM_HEADS: int = 4
ATTN_ADD_KEY_VALUE_BIAS_FLAG: bool = False
ATTN_ZERO_ATTENTION_FLAG: bool = False

## Attention Projection Stack Options
ATTN_NUM_LAYERS: int = 1
ATTN_BIAS_FLAG: bool = True
ATTN_STACK_HIDDEN_DIM: int = MODEL_DIM
ATTN_STACK_ACTIVATION: ActivationOptions = ActivationOptions.DISABLED
ATTN_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_STACK_DROPOUT_PROBABILITY: float = 0.0
ATTN_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
ATTN_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

### Attention Projection Gate Options
ATTN_STACK_GATE_FLAG: bool = False
ATTN_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
ATTN_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

#### Attention Projection Gate Stack Options
ATTN_GATE_STACK_INDEPENDENT_FLAG: bool = False
ATTN_GATE_STACK_HIDDEN_DIM: int | None = None
ATTN_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_GATE_STACK_NUM_LAYERS: int | None = None
ATTN_GATE_STACK_ACTIVATION: ActivationOptions | None = ActivationOptions.TANH
ATTN_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = True
ATTN_GATE_STACK_BIAS_FLAG: bool | None = True

### Attention Projection Halting Options
ATTN_STACK_HALTING_FLAG: bool = False
ATTN_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
ATTN_HALTING_THRESHOLD: float = 0.999
ATTN_HALTING_DROPOUT: float = 0.0
ATTN_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)

#### Attention Projection Halting Stack Options
ATTN_HALTING_STACK_INDEPENDENT_FLAG: bool = False
ATTN_HALTING_STACK_HIDDEN_DIM: int | None = None
ATTN_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    LayerNormPositionOptions.DISABLED
)
ATTN_HALTING_STACK_NUM_LAYERS: int | None = None
ATTN_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    LastLayerBiasOptions.DISABLED
)
ATTN_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ATTN_HALTING_STACK_BIAS_FLAG: bool | None = None

### Attention Projection Memory Options
ATTN_MEMORY_FLAG: bool = False
ATTN_MEMORY_OPTION: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
ATTN_MEMORY_POSITION_OPTION: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
ATTN_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
ATTN_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None

#### Attention Projection Memory Stack Options
ATTN_MEMORY_STACK_INDEPENDENT_FLAG: bool = False
ATTN_MEMORY_STACK_HIDDEN_DIM: int | None = None
ATTN_MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_MEMORY_STACK_NUM_LAYERS: int | None = None
ATTN_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_MEMORY_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ATTN_MEMORY_STACK_BIAS_FLAG: bool | None = None

### Attention Projection Recurrent Layer Options
ATTN_RECURRENT_FLAG: bool = False
ATTN_RECURRENT_MAX_STEPS: int = 2
ATTN_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)

#### Attention Projection Recurrent Gate Options
ATTN_RECURRENT_STACK_GATE_FLAG: bool = False
ATTN_RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
ATTN_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

##### Attention Projection Recurrent Gate Stack Options
ATTN_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
ATTN_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
ATTN_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
ATTN_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ATTN_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

#### Attention Projection Recurrent Halting Options
ATTN_RECURRENT_STACK_HALTING_FLAG: bool = False
ATTN_RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
ATTN_RECURRENT_HALTING_THRESHOLD: float = 0.999
ATTN_RECURRENT_HALTING_DROPOUT: float = 0.0
ATTN_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)

##### Attention Projection Recurrent Halting Stack Options
ATTN_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
ATTN_RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
ATTN_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
ATTN_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = (
    None
)
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ATTN_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None


# Feed-Forward Stack Options
FF_NUM_LAYERS: int = 1
FF_BIAS_FLAG: bool = True
FF_STACK_HIDDEN_DIM: int = 512
FF_STACK_ACTIVATION: ActivationOptions = ActivationOptions.RELU
FF_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_STACK_DROPOUT_PROBABILITY: float = DROPOUT_PROBABILITY
FF_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
FF_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

## Feed-Forward Gate Options
FF_STACK_GATE_FLAG: bool = False
FF_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
FF_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

### Feed-Forward Gate Stack Options
FF_GATE_STACK_INDEPENDENT_FLAG: bool = False
FF_GATE_STACK_HIDDEN_DIM: int | None = None
FF_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_GATE_STACK_NUM_LAYERS: int | None = None
FF_GATE_STACK_ACTIVATION: ActivationOptions | None = ActivationOptions.TANH
FF_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
FF_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = True
FF_GATE_STACK_BIAS_FLAG: bool | None = True

## Feed-Forward Halting Options
FF_STACK_HALTING_FLAG: bool = False
FF_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
FF_HALTING_THRESHOLD: float = 0.999
FF_HALTING_DROPOUT: float = 0.0
FF_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)

### Feed-Forward Halting Stack Options
FF_HALTING_STACK_INDEPENDENT_FLAG: bool = False
FF_HALTING_STACK_HIDDEN_DIM: int | None = None
FF_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    LayerNormPositionOptions.DISABLED
)
FF_HALTING_STACK_NUM_LAYERS: int | None = None
FF_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
FF_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
FF_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    LastLayerBiasOptions.DISABLED
)
FF_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
FF_HALTING_STACK_BIAS_FLAG: bool | None = None

## Feed-Forward Memory Options
FF_MEMORY_FLAG: bool = False
FF_MEMORY_OPTION: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
FF_MEMORY_POSITION_OPTION: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
FF_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
FF_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None

### Feed-Forward Memory Stack Options
FF_MEMORY_STACK_INDEPENDENT_FLAG: bool = False
FF_MEMORY_STACK_HIDDEN_DIM: int | None = None
FF_MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_MEMORY_STACK_NUM_LAYERS: int | None = None
FF_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
FF_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_MEMORY_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
FF_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
FF_MEMORY_STACK_BIAS_FLAG: bool | None = None

## Feed-Forward Recurrent Layer Options
FF_RECURRENT_FLAG: bool = False
FF_RECURRENT_MAX_STEPS: int = 2
FF_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)

### Feed-Forward Recurrent Gate Options
FF_RECURRENT_STACK_GATE_FLAG: bool = False
FF_RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
FF_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

#### Feed-Forward Recurrent Gate Stack Options
FF_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
FF_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
FF_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
FF_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
FF_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
FF_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
FF_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

### Feed-Forward Recurrent Halting Options
FF_RECURRENT_STACK_HALTING_FLAG: bool = False
FF_RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
FF_RECURRENT_HALTING_THRESHOLD: float = 0.999
FF_RECURRENT_HALTING_DROPOUT: float = 0.0
FF_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)

#### Feed-Forward Recurrent Halting Stack Options
FF_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
FF_RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
FF_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
FF_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
FF_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = (
    None
)
FF_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
FF_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
FF_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

# Transformer Options
ENCODER_NUM_LAYERS = 3
DECODER_NUM_LAYERS = 3
ENCODER_LAYER_NORM_POSITION = LayerNormPositionOptions.BEFORE
DECODER_LAYER_NORM_POSITION = LayerNormPositionOptions.BEFORE
ENCODER_ATTN_NUM_HEADS = ATTN_NUM_HEADS
DECODER_SELF_ATTN_NUM_HEADS = ATTN_NUM_HEADS
DECODER_CROSS_ATTN_NUM_HEADS = ATTN_NUM_HEADS
ENCODER_FEED_FORWARD_HIDDEN_DIM = FF_STACK_HIDDEN_DIM
DECODER_FEED_FORWARD_HIDDEN_DIM = FF_STACK_HIDDEN_DIM
ENCODER_FEED_FORWARD_NUM_LAYERS = FF_NUM_LAYERS
DECODER_FEED_FORWARD_NUM_LAYERS = FF_NUM_LAYERS

# Controller Options
STACK_GATE_FLAG = False
STACK_HALTING_FLAG = False
HALTING_THRESHOLD: float | None = None
MEMORY_FLAG = False
RECURRENT_FLAG = False
RECURRENT_STACK_GATE_FLAG = False
RECURRENT_STACK_HALTING_FLAG = False
RECURRENT_HALTING_THRESHOLD: float | None = None
RECURRENT_MAX_STEPS = 2
RECURRENT_INITIAL_ITERATIONS: int = 2
RECURRENT_GRADIENT_TRANSITION_COUNT: int | None = None
RECURRENT_ITERATION_INCREMENT: int = 1
RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT: int = 1
RECURRENT_RESIDUAL_CONNECTION_OPTION = None
RECURRENT_RESIDUAL_MODEL_FLAG: bool = False

## Residual Options
# - False uses the residual variant's learned coefficient parameters.
# - True uses the residual stack for data-dependent coefficients.
STACK_RESIDUAL_CONNECTION_OPTION = None
STACK_RESIDUAL_MODEL_FLAG: bool = False
### Residual Stack Options
# - If False, residual stack options inherit layer stack submodule options.
RESIDUAL_STACK_INDEPENDENT_FLAG: bool = False
RESIDUAL_STACK_HIDDEN_DIM: int | None = None
RESIDUAL_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
RESIDUAL_STACK_NUM_LAYERS: int | None = None
RESIDUAL_STACK_ACTIVATION: ActivationOptions | None = None
RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
RESIDUAL_STACK_RESIDUAL_MODEL_FLAG: bool = False
RESIDUAL_STACK_DROPOUT_PROBABILITY: float | None = None
RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
RESIDUAL_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
RESIDUAL_STACK_BIAS_FLAG: bool | None = None


# Adaptive Parameter Options
_CONFIG_FIELD_METADATA_ALIASES: dict[str, str] = {}

GROUPING_SCOPE: AdaptiveParameterGroupingScopeOptions = (
    AdaptiveParameterGroupingScopeOptions.DISABLED
)
GROUP_COUNT: int = 1
WEIGHT_OPTION_FLAG: bool = True
WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_ONE
WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
WEIGHT_DECAY_RATE: float = 0.0
WEIGHT_DECAY_WARMUP_BATCHES: int = 0
WEIGHT_NORMALIZATION_OPTION: WeightNormalizationOptions = (
    WeightNormalizationOptions.DISABLED
)
WEIGHT_NORMALIZATION_POSITION_OPTION: WeightNormalizationPositionOptions = (
    WeightNormalizationPositionOptions.DISABLED
)
WEIGHT_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_ONE
)
BIAS_OPTION_FLAG: bool = True
BIAS_OPTION: type[DynamicBiasConfig] | None = None
BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
BIAS_DECAY_RATE: float = 0.0
BIAS_DECAY_WARMUP_BATCHES: int = 0
BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_ONE
)
DIAGONAL_OPTION_FLAG: bool = True
DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
MASK_OPTION_FLAG: bool = True
ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
MASK_THRESHOLD: float = 0.5
MASK_SURROGATE_SCALE: float = 1.0
MASK_FLOOR: float = 0.0
MASK_DIMENSION_OPTION: MaskDimensionOptions = MaskDimensionOptions.ROW
MASK_TRANSITION_WIDTH: float = 0.1

# Shared adaptive generator stack. Component-specific stacks inherit this unless
# their INDEPENDENT_FLAG is enabled.
ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: int = 64
ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: int = 1
ADAPTIVE_GENERATOR_STACK_ACTIVATION: ActivationOptions = ActivationOptions.RELU
ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ADAPTIVE_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: float = 0.0
ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
ADAPTIVE_GENERATOR_STACK_BIAS_FLAG: bool = True

_COMPONENT_GENERATOR_DEFAULTS = {
    "INDEPENDENT_FLAG": False,
    "HIDDEN_DIM": None,
    "NUM_LAYERS": None,
    "ACTIVATION": None,
    "LAYER_NORM_POSITION": None,
    "RESIDUAL_CONNECTION_OPTION": None,
    "RESIDUAL_MODEL_FLAG": False,
    "DROPOUT_PROBABILITY": None,
    "LAST_LAYER_BIAS_OPTION": None,
    "APPLY_OUTPUT_PIPELINE_FLAG": None,
    "BIAS_FLAG": None,
}
_annotations = globals().setdefault("__annotations__", {})
for _component in ("WEIGHT", "BIAS", "DIAGONAL", "MASK"):
    for _suffix, _value in _COMPONENT_GENERATOR_DEFAULTS.items():
        _target = f"{_component}_GENERATOR_STACK_{_suffix}"
        globals()[_target] = _value
        _source_suffix = "HIDDEN_DIM" if _suffix == "INDEPENDENT_FLAG" else _suffix
        _source = f"ADAPTIVE_GENERATOR_STACK_{_source_suffix}"
        _CONFIG_FIELD_METADATA_ALIASES[_target] = _source
        if _suffix == "INDEPENDENT_FLAG":
            _annotations[_target] = bool
        elif _source in _annotations:
            _annotations[_target] = (
                _annotations[_source]
                if _value is not None
                else _annotations[_source] | None
            )


_ADAPTIVE_DEFAULT_NAMES = {
    "GROUPING_SCOPE": "GROUPING_SCOPE",
    "GROUP_COUNT": "GROUP_COUNT",
    "WEIGHT_OPTION_FLAG": "WEIGHT_OPTION_FLAG",
    "WEIGHT_OPTION": "WEIGHT_OPTION",
    "GENERATOR_DEPTH": "GENERATOR_DEPTH",
    "WEIGHT_DECAY_SCHEDULE": "WEIGHT_DECAY_SCHEDULE",
    "WEIGHT_DECAY_RATE": "WEIGHT_DECAY_RATE",
    "WEIGHT_DECAY_WARMUP_BATCHES": "WEIGHT_DECAY_WARMUP_BATCHES",
    "WEIGHT_NORMALIZATION_OPTION": "WEIGHT_NORMALIZATION_OPTION",
    "WEIGHT_NORMALIZATION_POSITION_OPTION": ("WEIGHT_NORMALIZATION_POSITION_OPTION"),
    "WEIGHT_BANK_EXPANSION_FACTOR": "WEIGHT_BANK_EXPANSION_FACTOR",
    "BIAS_OPTION_FLAG": "BIAS_OPTION_FLAG",
    "BIAS_OPTION": "BIAS_OPTION",
    "BIAS_DECAY_SCHEDULE": "BIAS_DECAY_SCHEDULE",
    "BIAS_DECAY_RATE": "BIAS_DECAY_RATE",
    "BIAS_DECAY_WARMUP_BATCHES": "BIAS_DECAY_WARMUP_BATCHES",
    "BIAS_BANK_EXPANSION_FACTOR": "BIAS_BANK_EXPANSION_FACTOR",
    "DIAGONAL_OPTION_FLAG": "DIAGONAL_OPTION_FLAG",
    "DIAGONAL_OPTION": "DIAGONAL_OPTION",
    "MASK_OPTION_FLAG": "MASK_OPTION_FLAG",
    "ROW_MASK_OPTION": "ROW_MASK_OPTION",
    "MASK_THRESHOLD": "MASK_THRESHOLD",
    "MASK_SURROGATE_SCALE": "MASK_SURROGATE_SCALE",
    "MASK_FLOOR": "MASK_FLOOR",
    "MASK_DIMENSION_OPTION": "MASK_DIMENSION_OPTION",
    "MASK_TRANSITION_WIDTH": "MASK_TRANSITION_WIDTH",
}
for _name in tuple(globals()):
    if _name.startswith("ADAPTIVE_GENERATOR_STACK_"):
        _ADAPTIVE_DEFAULT_NAMES[_name.removeprefix("ADAPTIVE_")] = _name
    elif any(
        _name.startswith(f"{_component}_GENERATOR_STACK_")
        for _component in ("WEIGHT", "BIAS", "DIAGONAL", "MASK")
    ):
        _ADAPTIVE_DEFAULT_NAMES[_name] = _name


def _copy_adaptive_runtime_defaults(target_prefix: str) -> None:
    annotations = globals().setdefault("__annotations__", {})
    for suffix, source in _ADAPTIVE_DEFAULT_NAMES.items():
        target = f"{target_prefix}{suffix}"
        globals()[target] = globals()[source]
        _CONFIG_FIELD_METADATA_ALIASES[target] = source
        if source in annotations:
            annotations[target] = annotations[source]


for _adaptive_target_prefix in (
    "PROJECTION_ADAPTIVE_",
    "FEED_FORWARD_ADAPTIVE_",
    "ATTN_",
    "FF_",
    "ENCODER_ATTN_ADAPTIVE_",
    "DECODER_SELF_ATTN_ADAPTIVE_",
    "DECODER_CROSS_ATTN_ADAPTIVE_",
    "ENCODER_FF_ADAPTIVE_",
    "DECODER_FF_ADAPTIVE_",
):
    _copy_adaptive_runtime_defaults(_adaptive_target_prefix)

del (
    _adaptive_target_prefix,
    _annotations,
    _component,
    _name,
    _source,
    _source_suffix,
    _suffix,
    _target,
    _value,
)

# Trainer
NUM_EPOCHS = 30
TRAINER_ACCELERATOR: str = "auto"
TRAINER_DEVICES: str | int = "auto"
TRAINER_ACCUMULATE_GRAD_BATCHES: int = 1
TRAINER_PRECISION: str = "32-true"
TRAINER_DETERMINISTIC = True
TRAINER_BENCHMARK: bool = False
TRAINER_MAX_STEPS: int = -1
TRAINER_MAX_TIME: str | None = None
TRAINER_VAL_CHECK_INTERVAL: float = 1.0
TRAINER_LIMIT_TRAIN_BATCHES: float = 1.0
TRAINER_LIMIT_VAL_BATCHES: float = 1.0
TRAINER_OVERFIT_BATCHES: int | float = 0.0
TRAINER_NUM_SANITY_VAL_STEPS: int = 2
TRAINER_GRADIENT_CLIP_VAL = 1.0
TRAINER_GRADIENT_CLIP_ALGORITHM = "norm"
TRAINER_LOG_EVERY_N_STEPS = 10
TRAINER_ENABLE_PROGRESS_BAR: bool = False
TRAINER_ENABLE_CHECKPOINTING: bool = False
TRAINER_ENABLE_MODEL_SUMMARY: bool = False
TRAINER_PROFILER: str | None = None

MONITOR_LOG_EVERY_N_STEPS: int = 100

# Callback
CALLBACK_EARLY_STOPPING_PATIENCE = 5
CALLBACK_EARLY_STOPPING_METRIC = "validation/loss"
CALLBACK_EARLY_STOPPING_MIN_DELTA = 0.0
CALLBACK_EARLY_STOPPING_STRICT = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE = True
CALLBACK_CHECKPOINT_FLAG = False

# Run
DATA_NUM_WORKERS = 0
RUN_TEST_AFTER_FIT = True
SEED: int | None = 0


def _copy_path_runtime_defaults(source_prefix: str, target_prefix: str) -> None:
    annotations = globals().setdefault("__annotations__", {})
    for name, value in tuple(globals().items()):
        if not name.isupper() or not name.startswith(source_prefix):
            continue
        target = f"{target_prefix}{name.removeprefix(source_prefix)}"
        globals()[target] = value
        _CONFIG_FIELD_METADATA_ALIASES[target] = name
        if name in annotations:
            annotations[target] = annotations[name]


for _target_prefix in (
    "ENCODER_ATTN_",
    "DECODER_SELF_ATTN_",
    "DECODER_CROSS_ATTN_",
):
    _copy_path_runtime_defaults("ATTN_", _target_prefix)
for _target_prefix in ("ENCODER_FF_", "DECODER_FF_"):
    _copy_path_runtime_defaults("FF_", _target_prefix)

ENCODER_FF_STACK_HIDDEN_DIM = ENCODER_FEED_FORWARD_HIDDEN_DIM
DECODER_FF_STACK_HIDDEN_DIM = DECODER_FEED_FORWARD_HIDDEN_DIM
ENCODER_FF_NUM_LAYERS = ENCODER_FEED_FORWARD_NUM_LAYERS
DECODER_FF_NUM_LAYERS = DECODER_FEED_FORWARD_NUM_LAYERS

del _target_prefix

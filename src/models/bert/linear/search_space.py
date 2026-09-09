from emperor.layers import LayerNormPositionOptions, NormalizationOptions

from .config import *  # noqa: F401,F403

SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM: list = [16, 32, 64, 128]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [1, 2, 4, 8]
SEARCH_SPACE_LAYER_NORM_POSITION: list = [
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]
SEARCH_SPACE_NORMALIZATION: list = list(NormalizationOptions)
SEARCH_SPACE_ATTN_NUM_HEADS: list = [1, 2, 4]

# Embedding and output normalization
SEARCH_SPACE_ENCODER_OUTPUT_NORMALIZATION: list = list(NormalizationOptions)
SEARCH_SPACE_EMBEDDING_NORMALIZATION: list = list(NormalizationOptions)
SEARCH_SPACE_MLM_NORMALIZATION: list = list(NormalizationOptions)

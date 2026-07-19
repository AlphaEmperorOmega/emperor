from emperor.base.options import LayerNormPositionOptions

from .config import *  # noqa: F401,F403

SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM: list = [16, 32, 64, 128]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [1, 2, 4, 8]
SEARCH_SPACE_LAYER_NORM_POSITION: list = [
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]
SEARCH_SPACE_STACK_LAYER_NORM_POSITION: list = SEARCH_SPACE_LAYER_NORM_POSITION
SEARCH_SPACE_IMAGE_PATCH_SIZE: list = [4, 2, 1]
SEARCH_SPACE_ATTN_NUM_HEADS: list = [1, 2, 4]

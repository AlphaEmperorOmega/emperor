from emperor.layers import LayerNormPositionOptions

from .config import *  # noqa: F401,F403

# Optimization
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]

# Model
SEARCH_SPACE_HIDDEN_DIM: list = [16, 32, 64, 128]

# Image patches
SEARCH_SPACE_IMAGE_PATCH_SIZE: list = [4, 8, 16]

# Mixer blocks
SEARCH_SPACE_STACK_NUM_LAYERS: list = [1, 2, 4, 8]
SEARCH_SPACE_LAYER_NORM_POSITION: list = [
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]

# Token mixer
SEARCH_SPACE_TOKEN_MIXER_STACK_HIDDEN_DIM: list = [32, 64, 128]

# Channel mixer
SEARCH_SPACE_CHANNEL_MIXER_STACK_HIDDEN_DIM: list = [64, 128, 256]

# ruff: noqa: F405

from .config import *  # noqa: F401,F403

SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]

SEARCH_SPACE_HIDDEN_DIM: list = [16, 32, 64, 128, 256, 512]

SEARCH_SPACE_STACK_NUM_LAYERS: list = [2, 4, 8, 16, 32]

SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

SEARCH_SPACE_LAYER_NORM_POSITION: list = [
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


SEARCH_SPACE_CLUSTER_MAX_STEPS: list = [1, 2, 4, 6]

SEARCH_SPACE_CLUSTER_TERMINAL_TOP_K: list = [1, 2]

SEARCH_SPACE_CLUSTER_GROWTH_THRESHOLD: list = [100, 250, 500, None]

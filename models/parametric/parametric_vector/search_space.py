# ruff: noqa: F405

from .config import *  # noqa: F401,F403

SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3]

SEARCH_SPACE_HIDDEN_DIM: list = [32, 64]

SEARCH_SPACE_STACK_NUM_LAYERS: list = [1, 2]

SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1]

SEARCH_SPACE_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.GELU,
]

SEARCH_SPACE_ADAPTIVE_MIXTURE_NUM_EXPERTS: list = [2, 3]

from models.trainer_config import *
from emperor.datasets.image.classification.mnist import Mnist
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.linears.core.monitor import LinearMonitorCallback
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)

# Global
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 2
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_GRADIENT_CLIP_VAL: float = 1.0
CALLBACK_EARLY_STOPPING_PATIENCE: int = 5
# CALLBACK_LINEAR_MONITOR = LinearMonitorCallback(log_every_n_steps=100)

# Model
INPUT_DIM: int = 28**2
OUTPUT_DIM: int = 10
BIAS_FLAG: bool = True

#########################################################################
# LAYER STACK OPTIONS
HIDDEN_DIM: int = 256
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_NUM_LAYERS: int = 5
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_FLAG: bool = True
STACK_DROPOUT_PROBABILITY: float = 0.2
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

#########################################################################
# GATE STACK OPTIONS
# If `GATE_FLAG` is False, the gate-specific parameters below are ignored.
GATE_FLAG: bool = False
GATE_HIDDEN_DIM: int = 256
GATE_LAYER_NORM_POSITION: LayerNormPositionOptions = LAYER_NORM_POSITION
GATE_STACK_NUM_LAYERS: int = 2
GATE_STACK_ACTIVATION: ActivationOptions = STACK_ACTIVATION
GATE_STACK_RESIDUAL_FLAG: bool = STACK_RESIDUAL_FLAG
GATE_STACK_DROPOUT_PROBABILITY: float = STACK_DROPOUT_PROBABILITY
GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = STACK_LAST_LAYER_BIAS_OPTION
GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
GATE_BIAS_FLAG: bool = BIAS_FLAG

#########################################################################
# Halting options
# HALTING OPTIONS
# If `HALTING_FLAG` is False, the halting-specific parameters below are ignored.
HALTING_FLAG: bool = False
HALTING_THRESHOLD: float = 0.99
HALTING_DROPOUT: float = 0.0
HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
HALTING_GATE_HIDDEN_DIM: int = HIDDEN_DIM
HALTING_GATE_OUTPUT_DIM: int = 2
HALTING_GATE_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
HALTING_GATE_STACK_NUM_LAYERS: int = STACK_NUM_LAYERS
HALTING_GATE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.DISABLED
HALTING_GATE_STACK_RESIDUAL_FLAG: bool = STACK_RESIDUAL_FLAG
HALTING_GATE_STACK_DROPOUT_PROBABILITY: float = STACK_DROPOUT_PROBABILITY
HALTING_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DISABLED
)
HALTING_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
HALTING_GATE_BIAS_FLAG: bool = BIAS_FLAG

#########################################################################
# HYPERPARAMETER SEARCH SPACE
# These values define the parameter ranges explored when search mode is enabled.
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM: list = [64, 128, 256]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [3, 6, 9]
SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2]
SEARCH_SPACE_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.SILU,
    ActivationOptions.GELU,
    ActivationOptions.MISH,
]

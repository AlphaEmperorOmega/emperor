from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.base.layer import LayerStackConfig
from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST

# Global
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]

# Trainer
GRADIENT_CLIP_VAL: float = 1.0
GRADIENT_CLIP_ALGORITHM: str = "norm"
ACCUMULATE_GRAD_BATCHES: int = 1
PRECISION: str = "32-true"
DETERMINISTIC: bool = False
BENCHMARK: bool = True
MAX_STEPS: int = -1
MAX_TIME: str | None = None
VAL_CHECK_INTERVAL: float = 1.0
LIMIT_TRAIN_BATCHES: float = 1.0
LIMIT_VAL_BATCHES: float = 1.0
OVERFIT_BATCHES: int | float = 0.0
NUM_SANITY_VAL_STEPS: int = 2
LOG_EVERY_N_STEPS: int = 50
ENABLE_PROGRESS_BAR: bool = True
PROFILER: str | None = None
EARLY_STOPPING_PATIENCE: int = 0
EARLY_STOPPING_METRIC: str = "validation_loss"
CHECKPOINT_FLAG: bool = False

# Model
INPUT_DIM: int = 28**2
OUTPUT_DIM: int = 10
BIAS_FLAG: bool = True

# Layer stack options
HIDDEN_DIM: int = 256
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_NUM_LAYERS: int = 3
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_FLAG: bool = True
STACK_DROPOUT_PROBABILITY: float = 0.2

# Hyperparameter search space
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM: list = [64, 128, 256]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [3, 6, 9]
# SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2]
# SEARCH_SPACE_STACK_ACTIVATION: list = [
#     ActivationOptions.RELU,
#     ActivationOptions.SILU,
#     ActivationOptions.GELU,
#     ActivationOptions.MISH,
# ]


@dataclass
class ExperimentConfig(ConfigBase):
    model_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )

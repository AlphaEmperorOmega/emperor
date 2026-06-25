from emperor.base.layer.residual import ResidualConnectionOptions

from emperor.base.layer.monitor import LayerControllerMonitorCallback
from emperor.base.options import ActivationOptions
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.datasets.image.classification.mnist import Mnist
from emperor.parametric.core.mixtures.config import MatrixBiasMixtureConfig
from emperor.parametric.core.mixtures.options import ClipParameterOptions
from emperor.experiments.monitors import MonitorOption
from emperor.parametric.core.monitor import ParametricLayerMonitorCallback
from models.trainer_config import *

# Global
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 2
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]
MONITOR_OPTIONS: list[MonitorOption] = [
    MonitorOption(
        name="parametric",
        label="Parametric layers",
        description=(
            "Logs generated parameter norms, affine deltas, router entropy, "
            "mixture utilization, skip/drop fraction, auxiliary loss, and "
            "utilization visual summaries."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: ParametricLayerMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="layer-controller",
        label="Layer controllers",
        description=(
            "Logs Layer gate, residual, dropout, layer-norm, and activation "
            "controller statistics without duplicating memory metrics."
        ),
        kinds=["scalar"],
        callback_factory=lambda: LayerControllerMonitorCallback(log_every_n_steps=100),
    ),
]

# Model
INPUT_DIM: int = 28**2
OUTPUT_DIM: int = 10

# Layer stack options
STACK_HIDDEN_DIM: int = 64
STACK_NUM_LAYERS: int = 1
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
STACK_DROPOUT_PROBABILITY: float = 0.0

# Parametric matrix mixture
ADAPTIVE_MIXTURE_TOP_K: int = 1
ADAPTIVE_MIXTURE_NUM_EXPERTS: int = 2
ADAPTIVE_MIXTURE_WEIGHTED_PARAMETERS_FLAG: bool = False
ADAPTIVE_MIXTURE_CLIP_PARAMETER_OPTION: ClipParameterOptions = (
    ClipParameterOptions.DISABLED
)
ADAPTIVE_MIXTURE_CLIP_RANGE: float = 5.0
ADAPTIVE_BIAS_OPTION: type[MatrixBiasMixtureConfig] | None = None

# Sampler
SAMPLER_THRESHOLD: float = 0.0
SAMPLER_FILTER_ABOVE_THRESHOLD: bool = False
SAMPLER_NUM_TOPK_SAMPLES: int = 0
SAMPLER_NORMALIZE_PROBABILITIES_FLAG: bool = False
SAMPLER_NOISY_TOPK_FLAG: bool = False
SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
SAMPLER_SWITCH_LOSS_WEIGHT: float = 0.0
SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: float = 0.0
SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

# Hyperparameter search space
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3]
SEARCH_SPACE_STACK_HIDDEN_DIM: list = [32, 64]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [1, 2]
SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1]
SEARCH_SPACE_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.GELU,
]
SEARCH_SPACE_ADAPTIVE_MIXTURE_NUM_EXPERTS: list = [2, 3]
SEARCH_SPACE_ADAPTIVE_BIAS_OPTION: list = [None, MatrixBiasMixtureConfig]

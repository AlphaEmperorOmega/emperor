from emperor.base.layer.residual import ResidualConnectionOptions

from emperor.base.options import ActivationOptions
from emperor.parametric.core.mixtures.config import GeneratorBiasMixtureConfig
from emperor.parametric.core.mixtures.options import ClipParameterOptions
from models.trainer_config import *

# Global
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 2

# Model
INPUT_DIM: int = 28**2
HIDDEN_DIM: int = 32
OUTPUT_DIM: int = 10

# Layer Stack Options
# - hidden_dim comes from the global HIDDEN_DIM field above.
STACK_NUM_LAYERS: int = 1
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
STACK_DROPOUT_PROBABILITY: float = 0.0

# Parametric Generator Mixture Options
ADAPTIVE_MIXTURE_TOP_K: int = 1
ADAPTIVE_MIXTURE_NUM_EXPERTS: int = 2
ADAPTIVE_MIXTURE_WEIGHTED_PARAMETERS_FLAG: bool = False
ADAPTIVE_MIXTURE_CLIP_PARAMETER_OPTION: ClipParameterOptions = (
    ClipParameterOptions.DISABLED
)
ADAPTIVE_MIXTURE_CLIP_RANGE: float = 5.0
ADAPTIVE_BIAS_OPTION: type[GeneratorBiasMixtureConfig] | None = None

# Sampler Model Options
SAMPLER_THRESHOLD: float = 0.0
SAMPLER_FILTER_ABOVE_THRESHOLD: bool = False
SAMPLER_NUM_TOPK_SAMPLES: int = 0
SAMPLER_NORMALIZE_PROBABILITIES_FLAG: bool = False
SAMPLER_NOISY_TOPK_FLAG: bool = False
SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
SAMPLER_SWITCH_LOSS_WEIGHT: float = 0.0
SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: float = 0.0
SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

# Generator Expert Stack Options
GENERATOR_STACK_NUM_LAYERS: int = 2
GENERATOR_STACK_HIDDEN_DIM: int = 32
GENERATOR_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
GENERATOR_STACK_DROPOUT_PROBABILITY: float = 0.0

# Hyperparameter search space

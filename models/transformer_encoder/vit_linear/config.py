from models.trainer_config import *
from emperor.attention.core.monitor import AttentionMonitorCallback
from emperor.base.layer.monitor import LayerControllerMonitorCallback
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.datasets.image.classification.mnist import Mnist
from emperor.embedding.absolute.core.config import (
    AbsolutePositionalEmbeddingConfig,
    ImageLearnedPositionalEmbeddingConfig,
)
from emperor.experiments.monitors import MonitorOption

# Global
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]
MONITOR_OPTIONS: list[MonitorOption] = [
    MonitorOption(
        name="attention",
        label="Attention",
        description=(
            "Logs Q/K/V norms, attention entropy, max probability, dropout and "
            "mask coverage, auxiliary loss, and attention head visual summaries."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: AttentionMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="layer-controller",
        label="Layer controllers",
        description=(
            "Logs Layer gate, residual, dropout, layer-norm, and activation "
            "controller statistics without duplicating memory metrics."
        ),
        kinds=["scalar"],
        callback_factory=lambda: LayerControllerMonitorCallback(
            log_every_n_steps=100
        ),
    ),
]
CONFIG_OVERRIDE_SKIP_KEYS: set[str] = {
    "BIAS_FLAG",
    "SEQUENCE_LENGTH",
}

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_DEVICES: int = 1
TRAINER_GRADIENT_CLIP_VAL: float = 1.0
CALLBACK_EARLY_STOPPING_PATIENCE: int = 5
CALLBACK_EARLY_STOPPING_METRIC: str = "validation/loss"

# Model
INPUT_DIM: int = 32 * 32 * 3
OUTPUT_DIM: int = 10
SEQUENCE_LENGTH: int = 65
HIDDEN_DIM: int = 32
BIAS_FLAG: bool = True

#########################################################################
# IMAGE PATCHES
IMAGE_PATCH_SIZE: int = 4
INPUT_CHANNELS: int = 3
IMAGE_HEIGHT: int = 32
PATCH_DROPOUT_PROBABILITY: float = 0.0
PATCH_BIAS_FLAG: bool = True

#########################################################################
# MAIN ENCODER STACK (transformer encoder of self-attention + feed-forward)
STACK_NUM_LAYERS: int = 1
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_DROPOUT_PROBABILITY: float = 0.1
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE

#########################################################################
# POSITIONAL EMBEDDING (added to patch embeddings before the encoder)
POSITIONAL_EMBEDDING_OPTION: type[AbsolutePositionalEmbeddingConfig] = (
    ImageLearnedPositionalEmbeddingConfig
)
# None (not 0): images have no padding token, so the [CLS] slot at position 0 must
# learn its positional embedding rather than being zero-frozen by an nn.Embedding
# padding_idx (the text idiom). See proper_vit.md.
POSITIONAL_EMBEDDING_PADDING_IDX: int | None = None
POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG: bool = False

#########################################################################
# ATTENTION SUB-STACK (per encoder layer)
ATTN_NUM_HEADS: int = 4
ATTN_NUM_LAYERS: int = 1
ATTN_BIAS_FLAG: bool = False
ATTN_ADD_KEY_VALUE_BIAS_FLAG: bool = False

#########################################################################
# FEED-FORWARD SUB-STACK (per encoder layer)
FF_NUM_LAYERS: int = 2
FF_BIAS_FLAG: bool = True

#########################################################################
# OUTPUT HEAD (maps [CLS] hidden_dim to class logits)
OUTPUT_NUM_LAYERS: int = 2
OUTPUT_BIAS_FLAG: bool = True

#########################################################################
# HYPERPARAMETER SEARCH SPACE
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM: list = [16, 32, 64, 128, 256, 512]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [1, 2, 4, 8]
SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2, 0.3]
SEARCH_SPACE_LAYER_NORM_POSITION: list = [
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]
SEARCH_SPACE_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.GELU,
    ActivationOptions.SILU,
]

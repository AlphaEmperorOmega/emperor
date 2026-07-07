from models.trainer_config import *
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.embedding.absolute.core.config import (
    AbsolutePositionalEmbeddingConfig,
    ImageLearnedPositionalEmbeddingConfig,
)

# Global
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10
CONFIG_OVERRIDE_SKIP_KEYS: set[str] = {
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

#########################################################################
# IMAGE PATCHES
IMAGE_PATCH_SIZE: int = 4
INPUT_CHANNELS: int = 3
IMAGE_HEIGHT: int = 32
PATCH_DROPOUT_PROBABILITY: float = 0.0
PATCH_BIAS_FLAG: bool = True

#########################################################################
# Layer Stack Options
# - hidden_dim comes from the global HIDDEN_DIM field above.
STACK_NUM_LAYERS: int = 1
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_DROPOUT_PROBABILITY: float = 0.1
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE

#########################################################################
# POSITIONAL EMBEDDING (added to patch embeddings before the encoder)
POSITIONAL_EMBEDDING_OPTION: type[AbsolutePositionalEmbeddingConfig] = (
    ImageLearnedPositionalEmbeddingConfig
)
# Padding index note: use None, not 0. Images have no padding token, so the
# [CLS] slot at position 0 must learn its positional embedding rather than
# being zero-frozen by an nn.Embedding padding_idx. See proper_vit.md.
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

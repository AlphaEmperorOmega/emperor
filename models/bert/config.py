from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.base.layer import LayerStackConfig
from emperor.base.options import ActivationOptions
from emperor.datasets.text.language_modeling.penn_treebank import PennTreebank
from emperor.datasets.text.language_modeling.wiki_text_2 import WikiText2
from emperor.transformer import TransformerEncoderStackConfig
from emperor.embedding.absolute.core.config import AbsolutePositionalEmbeddingConfig
from models.trainer_config import *

# Global
BATCH_SIZE: int = 64
NUM_EPOCHS: int = 10
LEARNING_RATE: float = 1e-3
DATASET_OPTIONS: list = [PennTreebank, WikiText2]
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]

# Trainer
# ACCELERATOR: str = "cpu"
GRADIENT_CLIP_VAL: float = 1.0
EARLY_STOPPING_PATIENCE: int = 5

# Model
SEQUENCE_LENGTH: int = 35
HIDDEN_DIM: int = 128
ACTIVATION_FUNCTION: ActivationOptions = ActivationOptions.GELU

# Shared
DROPOUT_PROBABILITY: float = 0.1
TRANSFORMER_NUM_LAYERS: int = 2
ATTN_NUM_LAYERS: int = 1
FF_NUM_LAYERS: int = 2

# Preset
ATTN_BIAS_FLAG: bool = False
ATTN_NUM_HEADS: int = 4
FF_BIAS_FLAG: bool = True
OUTPUT_NUM_LAYERS: int = 1
OUTPUT_BIAS_FLAG: bool = True


@dataclass
class ExperimentConfig(ConfigBase):
    positional_embedding_config: "AbsolutePositionalEmbeddingConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    encoder_config: "TransformerEncoderStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    output_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )

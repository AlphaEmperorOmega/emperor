from emperor.base.layer.residual import ResidualConnectionOptions
from models.trainer_config import *
from emperor.attention.core.monitor import AttentionMonitorCallback
from emperor.base.layer.monitor import LayerControllerMonitorCallback
from emperor.datasets.text.bert_pretraining import (
    BERT_PRETRAINING_TARGET_VOCAB_SIZE,
    PennTreebankBertPretraining,
    WikiText2BertPretraining,
)
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.embedding.absolute.core.config import (
    AbsolutePositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
)
from emperor.experiments.monitors import MonitorOption

# Global
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10
DATASET_OPTIONS: list = [PennTreebankBertPretraining, WikiText2BertPretraining]
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
        callback_factory=lambda: LayerControllerMonitorCallback(log_every_n_steps=100),
    ),
]
CONFIG_OVERRIDE_SKIP_KEYS: set[str] = {
    "BERT_PRETRAINING_TARGET_VOCAB_SIZE",
    "BIAS_FLAG",
    "GATE_BIAS_FLAG",
    "GATE_FLAG",
    "GATE_HIDDEN_DIM",
    "GATE_LAYER_NORM_POSITION",
    "GATE_STACK_ACTIVATION",
    "GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG",
    "GATE_STACK_DROPOUT_PROBABILITY",
    "GATE_STACK_LAST_LAYER_BIAS_OPTION",
    "GATE_STACK_NUM_LAYERS",
    "GATE_STACK_RESIDUAL_CONNECTION_OPTION",
    "HALTING_BIAS_FLAG",
    "HALTING_DROPOUT",
    "HALTING_FLAG",
    "HALTING_HIDDEN_DIM",
    "HALTING_HIDDEN_STATE_MODE",
    "HALTING_LAYER_NORM_POSITION",
    "HALTING_OUTPUT_DIM",
    "HALTING_STACK_ACTIVATION",
    "HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG",
    "HALTING_STACK_DROPOUT_PROBABILITY",
    "HALTING_STACK_LAST_LAYER_BIAS_OPTION",
    "HALTING_STACK_NUM_LAYERS",
    "HALTING_STACK_RESIDUAL_CONNECTION_OPTION",
    "HALTING_THRESHOLD",
    "RECURRENT_FLAG",
    "RECURRENT_GATE_FLAG",
    "RECURRENT_HALTING_FLAG",
    "RECURRENT_MAX_STEPS",
}

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_DEVICES: int = 1
TRAINER_GRADIENT_CLIP_VAL: float = 1.0
CALLBACK_EARLY_STOPPING_PATIENCE: int = 5

# Callback
CALLBACK_EARLY_STOPPING_METRIC: str = "validation/loss"

# Model
INPUT_DIM: int = BERT_PRETRAINING_TARGET_VOCAB_SIZE
OUTPUT_DIM: int = BERT_PRETRAINING_TARGET_VOCAB_SIZE
SEQUENCE_LENGTH: int = 35
HIDDEN_DIM: int = 128
BIAS_FLAG: bool = True

#########################################################################
# MAIN ENCODER STACK (transformer encoder of self-attention + feed-forward)
# STACK_* names map onto the searchable transformer-stack axes below.
STACK_NUM_LAYERS: int = 2
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_DROPOUT_PROBABILITY: float = 0.1
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.DEFAULT
CAUSAL_ATTENTION_MASK_FLAG: bool = False

#########################################################################
# POSITIONAL EMBEDDING (added to token embeddings before the encoder)
POSITIONAL_EMBEDDING_OPTION: type[AbsolutePositionalEmbeddingConfig] = (
    TextLearnedPositionalEmbeddingConfig
)
POSITIONAL_EMBEDDING_PADDING_IDX: int = 0
POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG: bool = False

#########################################################################
# EMBEDDING PIPELINE (applied to the summed token+positional+segment embedding)
EMBEDDING_DROPOUT_PROBABILITY: float = STACK_DROPOUT_PROBABILITY

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
# GATE STACK OPTIONS (per encoder-layer block; recurrent gate when recurrent)
# If GATE_FLAG is False, the gate-specific parameters below are ignored.
GATE_FLAG: bool = False
GATE_HIDDEN_DIM: int = HIDDEN_DIM
GATE_LAYER_NORM_POSITION: LayerNormPositionOptions = LAYER_NORM_POSITION
GATE_STACK_NUM_LAYERS: int = 2
GATE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.TANH
GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
GATE_STACK_DROPOUT_PROBABILITY: float = 0.0
GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
GATE_BIAS_FLAG: bool = True

#########################################################################
# HALTING OPTIONS (per encoder-layer block; recurrent halting when recurrent)
# If HALTING_FLAG is False, the halting-specific parameters below are ignored.
HALTING_FLAG: bool = False
HALTING_THRESHOLD: float = 0.99
HALTING_DROPOUT: float = 0.0
HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
HALTING_HIDDEN_DIM: int = HIDDEN_DIM
HALTING_OUTPUT_DIM: int = 2
HALTING_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
HALTING_STACK_NUM_LAYERS: int = 2
HALTING_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
HALTING_STACK_DROPOUT_PROBABILITY: float = 0.0
HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DISABLED
)
HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
HALTING_BIAS_FLAG: bool = BIAS_FLAG

#########################################################################
# RECURRENT LAYER OPTIONS
# If RECURRENT_FLAG is False, the recurrent-specific parameters below are ignored.
RECURRENT_FLAG: bool = False
RECURRENT_MAX_STEPS: int = 4
RECURRENT_GATE_FLAG: bool = False
RECURRENT_HALTING_FLAG: bool = False

#########################################################################
# HYPERPARAMETER SEARCH SPACE
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

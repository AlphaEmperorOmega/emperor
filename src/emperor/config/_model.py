from dataclasses import dataclass, field

from emperor.config._base import ConfigBase

_BATCH_SIZE = 10
_SEQUENCE_LENGTH = 5
_LEARNING_RATE = 1e-3
_INPUT_DIM = 16
_HIDDEN_DIM = 32
_OUTPUT_DIM = 64


@dataclass
class ModelConfig(ConfigBase):
    batch_size: int = field(
        default=_BATCH_SIZE,
        metadata={"help": "Batch size for training and inference"},
    )
    learning_rate: float = field(
        default=_LEARNING_RATE,
        metadata={"help": "Learning rate for training"},
    )
    sequence_length: int = field(
        default=_SEQUENCE_LENGTH,
        metadata={"help": "Number of tokens for each sequence in the input batch."},
    )
    input_dim: int = field(
        default=_INPUT_DIM,
        metadata={"help": "Dimension of the input features"},
    )
    hidden_dim: int = field(
        default=_HIDDEN_DIM,
        metadata={"help": "Dimension of the hidden features"},
    )
    output_dim: int = field(
        default=_OUTPUT_DIM,
        metadata={"help": "Dimension of the output features"},
    )
    experiment_config: ConfigBase | None = field(
        default=None,
        metadata={"help": "Config used to build the model module within the layer"},
    )

from dataclasses import dataclass, field

from emperor.base.config import ConfigBase

# MODEL WISE CONFIG
BATCH_SIZE = 10
SEQUENCE_LENGTH = 5
LEARNING_RATE = 1e-3
INPUT_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 64
GATHER_FREQUENCY_FLAG = False


@dataclass
class ModelConfig(ConfigBase):
    batch_size: int = field(
        default=BATCH_SIZE,
        metadata={"help": "Batch size for training and inference"},
    )
    learning_rate: float = field(
        default=LEARNING_RATE,
        metadata={"help": "Learning rate for training"},
    )
    sequence_length: int = field(
        default=SEQUENCE_LENGTH,
        metadata={"help": "Number of tokens for each sequence in the input batch."},
    )
    input_dim: int = field(
        default=INPUT_DIM,
        metadata={"help": "Dimension of the input features"},
    )
    hidden_dim: int = field(
        default=HIDDEN_DIM,
        metadata={"help": "Dimension of the hidden features"},
    )
    output_dim: int = field(
        default=OUTPUT_DIM,
        metadata={"help": "Dimension of the output features"},
    )
    gather_frequency_flag: bool = field(
        default=GATHER_FREQUENCY_FLAG,
        metadata={
            "help": (
                "Flag to control frequency of gathering operations for the purpose "
                "of visualization"
            )
        },
    )
    experiment_config: ConfigBase | None = field(
        default=None,
        metadata={"help": "Config used to build the model module within the layer"},
    )
